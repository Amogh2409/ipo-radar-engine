"""
Deterministic fact-checking interceptor between Ollama and the scoring engine.

A 4B model is useful for reading prose and useless as an authority on
arithmetic it was handed. Observed failures on real RHPs from this very
pipeline:

  * "Insider exit multiple of 1.38, which is below the typical range for
     Indian SMEs" - filed under red_flags. 1.38x is REASSURING (holders are
     selling near what they paid), and the issue was mainboard, not SME.
  * "High RoNW (43.51%) but this is a positive" - listed as a red flag while
     the sentence itself disowns the classification.

Both are polarity inversions: the model identified the right fact and
attached the wrong sign. Neither is fixable by prompting, because the failure
is in the model's reasoning, not its instructions.

So nothing the model says about a number reaches the score without being
checked against the number Python already computed. Where they disagree,
Python wins. The design rule is: the LLM may contribute JUDGEMENT, never
FACTS, and never the SIGN of a fact.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal

from ..models import IPO
from ..util import clamp

log = logging.getLogger("ipo_radar.guardrails")

# Pydantic gives the strictest enforcement, but the interceptor must never be
# the reason the pipeline stops, so an equivalent stdlib validator stands in
# when it is absent. Both enforce the same enums and ranges.
try:                                             # pragma: no cover
    from pydantic import BaseModel, Field, ValidationError, field_validator
    HAVE_PYDANTIC = True
except ImportError:                              # pragma: no cover
    BaseModel = object                           # type: ignore[assignment,misc]
    HAVE_PYDANTIC = False

Severity = Literal["LOW", "MEDIUM", "CRITICAL"]
MoatStrength = Literal["none", "weak", "moderate", "strong"]
ProceedsQuality = Literal["growth", "debt repayment", "mixed",
                          "mostly exit", "unclear"]

# How many corrected claims before we stop trusting the model's judgement on
# this issue entirely. A couple of inversions is a bad day; five is a model
# that has not understood the document.
FALLBACK_THRESHOLD = 5

SEVERITIES: tuple[str, ...] = ("LOW", "MEDIUM", "CRITICAL")
MOATS: tuple[str, ...] = ("none", "weak", "moderate", "strong")
PROCEEDS: tuple[str, ...] = ("growth", "debt repayment", "mixed",
                             "mostly exit", "unclear")


# ------------------------------------------------------------ ground truth
@dataclass
class GroundTruth:
    """
    Everything Python knows for certain. The LLM is never allowed to
    contradict any of it.
    """
    symbol: str
    name: str
    board: str = "MAINBOARD"
    insider_exit_multiple: float | None = None
    ronw_pct: float | None = None
    eps_cagr_pct: float | None = None
    pe: float | None = None
    pb: float | None = None
    benchmark_pe: float | None = None
    loss_making: bool = False
    ofs_share: float | None = None
    anchor_marquee_hits: int | None = None
    anchor_investor_count: int | None = None
    gmp_pct: float | None = None
    quality_score: float = 50.0
    valuation_score: float = 50.0
    has_peers: bool = True

    @classmethod
    def build(cls, ipo: IPO, metrics: dict[str, Any],
              valuation: dict[str, Any],
              gmp_pct: float | None = None) -> "GroundTruth":
        ofs = None
        fresh, sale = ipo.fresh_issue_cr, ipo.ofs_cr
        if fresh is not None and sale is not None and (fresh + sale) > 0:
            ofs = sale / (fresh + sale)
        peers = metrics.get("peers")
        return cls(
            symbol=ipo.symbol, name=ipo.name, board=ipo.board or "MAINBOARD",
            insider_exit_multiple=metrics.get("insider_exit_multiple"),
            ronw_pct=valuation.get("ronw_pct"),
            eps_cagr_pct=valuation.get("eps_cagr_pct"),
            pe=valuation.get("pe"), pb=valuation.get("pb"),
            benchmark_pe=valuation.get("benchmark_pe"),
            loss_making=bool(valuation.get("loss_making")),
            ofs_share=ofs,
            anchor_marquee_hits=metrics.get("anchor_marquee_hits"),
            anchor_investor_count=metrics.get("anchor_investor_count"),
            gmp_pct=gmp_pct,
            quality_score=float(valuation.get("quality_score", 50.0)),
            valuation_score=float(valuation.get("valuation_score", 50.0)),
            has_peers=not (isinstance(peers, list) and len(peers) == 0),
        )


@dataclass
class Violation:
    kind: str            # inversion | metadata | schema | contradiction | range
    field: str
    detail: str
    removed: str = ""
    severity: str = "MEDIUM"


@dataclass
class GuardrailResult:
    cleaned: dict[str, Any]
    violations: list[Violation] = field(default_factory=list)
    used_fallback: bool = False
    schema_ok: bool = True
    fallback_reason: str = ""

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def summary(self) -> dict[str, Any]:
        by: dict[str, int] = {}
        for v in self.violations:
            by[v.kind] = by.get(v.kind, 0) + 1
        return {"violations": self.violation_count, "by_kind": by,
                "used_fallback": self.used_fallback, "schema_ok": self.schema_ok,
                "fallback_reason": self.fallback_reason,
                "details": [asdict(v) for v in self.violations[:12]]}


# --------------------------------------------------------- polarity rules
# A quantitative assertion: a number, or a magnitude word. Used to separate
# "the offer is largely an OFS" (fair qualitative comment) from "the insider
# exit multiple is HIGH" (a claim about a magnitude, which needs a magnitude).
MAGNITUDE_RX = re.compile(
    r"\d|\bhigh(?:er|ly)?\b|\blow(?:er)?\b|\bsignificant(?:ly)?\b|"
    r"\bexcessive\b|\bsteep\b|\bmassive\b|\bsubstantial(?:ly)?\b|"
    r"\belevated\b|\brich\b|\bmodest\b", re.I)


@dataclass
class PolarityRule:
    """
    A fact whose SIGN Python knows. If the model files a favourable fact under
    risks (or an adverse one under positives), the entry is deleted.

    `requires_fact` additionally forbids the model from asserting a MAGNITUDE
    for this metric when Python could not compute it at all. Observed live: a
    4B model read a 12x price-to-NAV and reported it as a "high insider exit
    multiple" on an issue whose RHP never disclosed a WACA. Sign inversion is
    one failure mode; inventing the metric outright is another.
    """
    name: str
    attr: str
    favourable: Callable[[float], bool]
    adverse: Callable[[float], bool]
    patterns: tuple[str, ...]
    requires_fact: bool = True

    def matches(self, text: str) -> bool:
        low = text.lower()
        return any(re.search(p, low) for p in self.patterns)


POLARITY_RULES: tuple[PolarityRule, ...] = (
    PolarityRule(
        # Selling holders exiting near their own cost is reassuring. The model
        # reliably gets this backwards.
        name="insider_exit_multiple", attr="insider_exit_multiple",
        favourable=lambda v: v < 2.0, adverse=lambda v: v >= 4.0,
        patterns=(r"insider exit", r"exit multiple", r"\bwaca\b",
                  r"weighted average cost", r"cost of acquisition",
                  r"selling shareholders?.{0,40}(paid|cost|acquir)")),
    PolarityRule(
        name="ronw", attr="ronw_pct",
        favourable=lambda v: v >= 20.0, adverse=lambda v: v < 8.0,
        patterns=(r"\bronw\b", r"return on net worth", r"return on equity",
                  r"\broe\b")),
    PolarityRule(
        name="eps_growth", attr="eps_cagr_pct",
        favourable=lambda v: v >= 20.0, adverse=lambda v: v < 0.0,
        patterns=(r"eps (growth|cagr|trajectory)", r"earnings growth",
                  r"profit growth", r"growth (in|of) (eps|earnings)")),
    PolarityRule(
        name="ofs_share", attr="ofs_share",
        favourable=lambda v: v < 0.40, adverse=lambda v: v > 0.85,
        patterns=(r"offer for sale", r"\bofs\b",
                  r"no (money|proceeds|funds) (reach|go|flow)"),
        requires_fact=False),
)


class FactChecker:
    """Runs the interceptor. Stateless; safe to share."""

    # ------------------------------------------------------------ public
    def check(self, raw: dict[str, Any] | None, gt: GroundTruth
              ) -> GuardrailResult:
        if not raw:
            return self._fallback(
                gt, [Violation("schema", "*", "model returned nothing",
                               severity="CRITICAL")],
                schema_ok=False, reason="empty model output")
        violations: list[Violation] = []
        data = dict(raw)

        data, schema_ok = self._enforce_schema(data, violations)
        data = self._enforce_metadata(data, gt, violations)
        data = self._enforce_polarity(data, gt, violations)
        data = self._enforce_numeric_claims(data, gt, violations)
        violations.extend(self._detect_contradictions(data, gt))

        critical = [v for v in violations if v.severity == "CRITICAL"]
        if not schema_ok or critical or len(violations) >= FALLBACK_THRESHOLD:
            reason = ("schema invalid" if not schema_ok else
                      "critical violation" if critical else
                      f"{len(violations)} violations >= {FALLBACK_THRESHOLD}")
            log.warning("%s: falling back to rule-based scoring (%s)",
                        gt.symbol, reason)
            return self._fallback(gt, violations, schema_ok=schema_ok,
                                  reason=reason)

        if violations:
            log.info("%s: guardrails corrected %d LLM claim(s): %s",
                     gt.symbol, len(violations),
                     ", ".join(sorted({v.kind for v in violations})))
        return GuardrailResult(cleaned=data, violations=violations,
                               schema_ok=True)

    # ------------------------------------------------------ 3. schema/enums
    def _enforce_schema(self, data: dict[str, Any],
                        violations: list[Violation]) -> tuple[dict[str, Any], bool]:
        """Enums, ranges and types. Pydantic when present, stdlib otherwise."""
        if HAVE_PYDANTIC:
            try:
                model = QualitativeAssessment(**_coerce(data))
                return model.model_dump(), True
            except ValidationError as exc:
                for err in exc.errors()[:6]:
                    violations.append(Violation(
                        "schema", ".".join(str(x) for x in err["loc"]) or "*",
                        err["msg"], severity="MEDIUM"))
                # one repair attempt: drop offending keys, revalidate
                repaired = _coerce(data)
                for err in exc.errors():
                    if err["loc"]:
                        repaired.pop(str(err["loc"][0]), None)
                try:
                    return QualitativeAssessment(**repaired).model_dump(), True
                except ValidationError:
                    return data, False
        return _stdlib_validate(data, violations)

    # ----------------------------------------------- 2. ground-truth facts
    @staticmethod
    def _enforce_metadata(data: dict[str, Any], gt: GroundTruth,
                          violations: list[Violation]) -> dict[str, Any]:
        """
        Strip claims that contradict exchange metadata. A mainboard issue
        cannot carry "SME liquidity risk"; SME benchmarks do not apply to it.
        """
        out = dict(data)
        if (gt.board or "").upper() == "MAINBOARD":
            bad = re.compile(r"\bsme\b|small and medium enterprise", re.I)
            for key in ("key_risks", "red_flags", "positives"):
                items = out.get(key)
                if not isinstance(items, list):
                    continue
                kept = []
                for it in items:
                    if bad.search(str(it)):
                        violations.append(Violation(
                            "metadata", key,
                            "SME claim on a MAINBOARD issue", removed=str(it)[:160]))
                    else:
                        kept.append(it)
                out[key] = kept
            for key in ("business_summary", "competitive_moat",
                        "use_of_proceeds"):
                text = out.get(key)
                if isinstance(text, str) and bad.search(text):
                    out[key] = re.sub(r"[^.]*\bSME\b[^.]*\.", "", text,
                                      flags=re.I).strip() or text
                    violations.append(Violation("metadata", key,
                                                "SME reference on MAINBOARD",
                                                severity="LOW"))
        if gt.gmp_pct is None:
            gmp_rx = re.compile(r"grey market|\bgmp\b", re.I)
            for key in ("key_risks", "red_flags", "positives"):
                items = out.get(key)
                if isinstance(items, list):
                    kept = [i for i in items if not gmp_rx.search(str(i))]
                    if len(kept) != len(items):
                        violations.append(Violation(
                            "metadata", key, "GMP claim with no GMP observed",
                            severity="LOW"))
                    out[key] = kept
        return out

    # ------------------------------------------------- 1. sign / polarity
    @staticmethod
    def _enforce_polarity(data: dict[str, Any], gt: GroundTruth,
                          violations: list[Violation]) -> dict[str, Any]:
        out = dict(data)
        neg_keys = ("key_risks", "red_flags")
        pos_keys = ("positives",)

        for rule in POLARITY_RULES:
            value = getattr(gt, rule.attr, None)
            if not isinstance(value, (int, float)):
                # Python has no such figure, so the model cannot have one
                # either. Drop any assertion about its magnitude.
                if not rule.requires_fact:
                    continue
                for key in neg_keys + pos_keys:
                    items = out.get(key)
                    if not isinstance(items, list):
                        continue
                    kept = []
                    for it in items:
                        text = str(it)
                        if rule.matches(text) and MAGNITUDE_RX.search(text):
                            violations.append(Violation(
                                "unsupported", key,
                                f"claims a magnitude for {rule.name}, which "
                                f"was not disclosed in the RHP and never "
                                f"computed", removed=text[:160]))
                        else:
                            kept.append(it)
                    out[key] = kept
                continue
            good, bad = rule.favourable(float(value)), rule.adverse(float(value))
            if not (good or bad):
                continue                     # neutral: the model may say either
            drop_from = neg_keys if good else pos_keys
            for key in drop_from:
                items = out.get(key)
                if not isinstance(items, list):
                    continue
                kept = []
                for it in items:
                    if rule.matches(str(it)):
                        violations.append(Violation(
                            "inversion", key,
                            f"{rule.name}={value:.2f} is "
                            f"{'favourable' if good else 'adverse'}; "
                            f"model classified it as "
                            f"{'a risk' if good else 'a positive'}",
                            removed=str(it)[:160]))
                    else:
                        kept.append(it)
                out[key] = kept

        # profitable issuers cannot carry a loss-making risk
        if not gt.loss_making:
            rx = re.compile(r"loss[- ]making|is (currently )?loss|negative (pat|profit)", re.I)
            for key in neg_keys:
                items = out.get(key)
                if not isinstance(items, list):
                    continue
                kept = []
                for it in items:
                    if rx.search(str(it)):
                        violations.append(Violation(
                            "inversion", key,
                            "issuer is profitable per restated financials",
                            removed=str(it)[:160]))
                    else:
                        kept.append(it)
                out[key] = kept
        return out

    # ------------------------------------------ numeric claim verification
    @staticmethod
    def _enforce_numeric_claims(data: dict[str, Any], gt: GroundTruth,
                                violations: list[Violation]) -> dict[str, Any]:
        """
        Catch invented figures. Where the model quotes a P/E or RoNW, it must
        match what Python computed from the RHP within tolerance.
        """
        out = dict(data)
        checks = ((r"p/?e\s*(?:ratio\s*)?(?:of|at|is)?\s*~?([\d.]+)\s*x?", gt.pe, 0.25, "P/E"),
                  (r"ro(?:nw|e)\s*(?:of|at|is)?\s*~?([\d.]+)\s*%", gt.ronw_pct, 0.15, "RoNW"))
        for key in ("key_risks", "red_flags", "positives"):
            items = out.get(key)
            if not isinstance(items, list):
                continue
            kept = []
            for it in items:
                text = str(it)
                bad = False
                for pattern, truth, tol, label in checks:
                    if not isinstance(truth, (int, float)) or truth == 0:
                        continue
                    m = re.search(pattern, text, re.I)
                    if not m:
                        continue
                    try:
                        claimed = float(m.group(1))
                    except ValueError:
                        continue
                    if abs(claimed - truth) / abs(truth) > tol:
                        violations.append(Violation(
                            "contradiction", key,
                            f"model quoted {label} {claimed:g}, computed "
                            f"{truth:.2f}", removed=text[:160]))
                        bad = True
                        break
                if not bad:
                    kept.append(it)
            out[key] = kept
        return out

    # -------------------------------------------- internal contradictions
    @staticmethod
    def _detect_contradictions(data: dict[str, Any],
                               gt: GroundTruth) -> list[Violation]:
        out: list[Violation] = []
        score = data.get("qualitative_score")
        sev = str(data.get("litigation_severity") or "").upper()
        if isinstance(score, (int, float)):
            if sev == "CRITICAL" and score >= 65:
                out.append(Violation("contradiction", "qualitative_score",
                                     f"score {score:.0f} with CRITICAL litigation",
                                     severity="MEDIUM"))
            if data.get("governance_flag") is True and score >= 75:
                out.append(Violation("contradiction", "qualitative_score",
                                     f"score {score:.0f} with a governance flag",
                                     severity="MEDIUM"))
        pos = {str(x).strip().lower() for x in (data.get("positives") or [])}
        neg = {str(x).strip().lower() for x in (data.get("red_flags") or [])}
        both = pos & neg
        if both:
            out.append(Violation("contradiction", "positives/red_flags",
                                 f"{len(both)} identical item(s) on both sides",
                                 removed=next(iter(both))[:160]))
        return out

    # ------------------------------------------------- rule-based fallback
    @staticmethod
    def _fallback(gt: GroundTruth, violations: list[Violation],
                  schema_ok: bool = False,
                  reason: str = "") -> GuardrailResult:
        """
        Zero-hallucination path. Every sentence is generated from a number
        Python computed, so this can be wrong only if the RHP was.
        """
        return GuardrailResult(cleaned=rule_based_assessment(gt),
                               violations=violations, used_fallback=True,
                               schema_ok=schema_ok, fallback_reason=reason)


# ---------------------------------------------------- pure-python analysis
def rule_based_assessment(gt: GroundTruth) -> dict[str, Any]:
    """Deterministic stand-in for the qualitative pass. No model involved."""
    positives: list[str] = []
    risks: list[str] = []
    score = 50.0

    if isinstance(gt.ronw_pct, (int, float)):
        if gt.ronw_pct >= 25:
            score += 12
            positives.append(f"RoNW of {gt.ronw_pct:.1f}% - strong returns on capital")
        elif gt.ronw_pct >= 15:
            score += 6
            positives.append(f"RoNW of {gt.ronw_pct:.1f}% is respectable")
        elif gt.ronw_pct < 8:
            score -= 10
            risks.append(f"RoNW of only {gt.ronw_pct:.1f}%")
    if isinstance(gt.eps_cagr_pct, (int, float)):
        if gt.eps_cagr_pct >= 30:
            score += 12
            positives.append(f"earnings compounding at {gt.eps_cagr_pct:.0f}% a year")
        elif gt.eps_cagr_pct < 0:
            score -= 12
            risks.append(f"earnings shrinking at {abs(gt.eps_cagr_pct):.0f}% a year")
    if gt.loss_making:
        score -= 18
        risks.append("loss-making in the latest reported year")
    if isinstance(gt.pe, (int, float)) and isinstance(gt.benchmark_pe, (int, float)) \
            and gt.benchmark_pe > 0:
        rel = gt.pe / gt.benchmark_pe
        if rel > 1.5:
            score -= 10
            risks.append(f"priced at {gt.pe:.1f}x against a {gt.benchmark_pe:.0f}x benchmark")
        elif rel < 0.8:
            score += 8
            positives.append(f"priced at {gt.pe:.1f}x, below the {gt.benchmark_pe:.0f}x benchmark")
    if isinstance(gt.insider_exit_multiple, (int, float)):
        if gt.insider_exit_multiple >= 4:
            score -= 10
            risks.append(f"selling holders exiting at {gt.insider_exit_multiple:.1f}x their cost")
        elif gt.insider_exit_multiple < 2:
            score += 6
            positives.append(f"offer priced at {gt.insider_exit_multiple:.2f}x recent insider cost")
    if isinstance(gt.ofs_share, (int, float)) and gt.ofs_share > 0.85:
        score -= 8
        risks.append(f"{gt.ofs_share:.0%} of the issue is an exit, not fresh capital")
    if isinstance(gt.anchor_marquee_hits, (int, float)) and gt.anchor_marquee_hits >= 4:
        score += 8
        positives.append(f"{int(gt.anchor_marquee_hits)} marquee institutions in the anchor book")
    if not gt.has_peers:
        risks.append("no comparable listed peers, so the multiple is hard to benchmark")

    return {
        "business_summary": (
            f"{gt.name} ({gt.board.lower()} issue). Assessed from restated RHP "
            f"figures only - the language model was bypassed because its output "
            f"failed validation."),
        "revenue_model": "not assessed without the language model",
        "competitive_moat": "not assessed without the language model",
        "moat_strength": "moderate",
        "key_risks": risks[:5] or ["no quantitative red flags in the parsed figures"],
        "red_flags": [r for r in risks if "loss-making" in r or "exiting at" in r][:5],
        "positives": positives[:5],
        "use_of_proceeds": "not assessed without the language model",
        "proceeds_quality": "unclear",
        "litigation_severity": "LOW",
        "governance_flag": False,
        "qualitative_score": round(clamp(score, 0.0, 100.0), 1),
        "confidence": 0.45,
        "_engine": "rule_based_fallback",
    }


# ------------------------------------------------------------- validation
def _coerce(data: dict[str, Any]) -> dict[str, Any]:
    """Normalise the sloppy-but-recoverable shapes small models emit."""
    out = dict(data)
    score = out.get("qualitative_score")
    if isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
        out["qualitative_score"] = float(score) * 100.0   # 0-1 scale slip
    sev = out.get("litigation_severity")
    if isinstance(sev, str):
        s = sev.strip().upper()
        out["litigation_severity"] = s if s in SEVERITIES else (
            "CRITICAL" if s.startswith("HIGH") else "LOW")
    moat = out.get("moat_strength")
    if isinstance(moat, str):
        m = moat.strip().lower()
        out["moat_strength"] = m if m in MOATS else "moderate"
    pq = out.get("proceeds_quality")
    if isinstance(pq, str):
        p = pq.strip().lower()
        out["proceeds_quality"] = p if p in PROCEEDS else "unclear"
    gf = out.get("governance_flag")
    if isinstance(gf, str):
        out["governance_flag"] = gf.strip().lower() in ("true", "yes", "1")
    for key in ("key_risks", "red_flags", "positives"):
        v = out.get(key)
        if v is None:
            out[key] = []
        elif isinstance(v, str):
            out[key] = [v]
        elif isinstance(v, list):
            out[key] = [str(x) for x in v if str(x).strip()]
    return out


def _stdlib_validate(data: dict[str, Any],
                     violations: list[Violation]) -> tuple[dict[str, Any], bool]:
    """Same enums and ranges as the pydantic model, without the dependency."""
    out = _coerce(data)
    ok = True
    enums = (("litigation_severity", SEVERITIES, "LOW"),
             ("moat_strength", MOATS, "moderate"),
             ("proceeds_quality", PROCEEDS, "unclear"))
    for key, allowed, default in enums:
        val = out.get(key)
        if val is None:
            out[key] = default
        elif val not in allowed:
            violations.append(Violation("range", key,
                                        f"{val!r} not in {allowed}",
                                        severity="LOW"))
            out[key] = default
    score = out.get("qualitative_score")
    if not isinstance(score, (int, float)):
        violations.append(Violation("schema", "qualitative_score",
                                    "missing or non-numeric",
                                    severity="CRITICAL"))
        ok = False
    else:
        out["qualitative_score"] = clamp(float(score), 0.0, 100.0)
    conf = out.get("confidence")
    out["confidence"] = clamp(float(conf), 0.0, 1.0) if isinstance(
        conf, (int, float)) else 0.5
    if not isinstance(out.get("governance_flag"), bool):
        out["governance_flag"] = False
    if not isinstance(out.get("business_summary"), str) or \
            not out["business_summary"].strip():
        violations.append(Violation("schema", "business_summary", "empty",
                                    severity="CRITICAL"))
        ok = False
    return out, ok


if HAVE_PYDANTIC:                                # pragma: no cover
    class QualitativeAssessment(BaseModel):
        """Strict contract for anything the model contributes to the score."""
        business_summary: str = Field(min_length=1)
        revenue_model: str = ""
        competitive_moat: str = ""
        moat_strength: MoatStrength = "moderate"
        key_risks: list[str] = Field(default_factory=list, max_length=8)
        red_flags: list[str] = Field(default_factory=list, max_length=8)
        positives: list[str] = Field(default_factory=list, max_length=8)
        use_of_proceeds: str = ""
        proceeds_quality: ProceedsQuality = "unclear"
        litigation_severity: Severity = "LOW"
        governance_flag: bool = False
        qualitative_score: float = Field(ge=0.0, le=100.0)
        confidence: float = Field(default=0.5, ge=0.0, le=1.0)

        @field_validator("qualitative_score", mode="before")
        @classmethod
        def _rescale(cls, v: Any) -> Any:
            if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
                return float(v) * 100.0
            return v
