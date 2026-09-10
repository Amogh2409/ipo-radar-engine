"""
Bayesian projection of final subscription.

The deterministic projector answers "what will the book close at?" with a
single number. That is the wrong shape of answer. On day one a QIB reading of
1.2x is compatible with a final book anywhere between 5x and 80x, and a point
estimate launders that ignorance into false conviction.

So the final subscription multiple X is treated as a random variable with

    log X ~ Normal(m, P)

log-space because subscription is strictly positive and heavily right-skewed:
a 60x book is unremarkable, a -3x book is impossible. The prior comes from
structural facts known before bidding opens, and each hourly snapshot updates
it through a scalar Kalman recursion.

The observation model is where the real work happens. Under the power law
x(t) = X * t^alpha, one snapshot implies

    z(t) = log x(t) - alpha * log t        (an estimate of log X)

and because alpha is itself uncertain, that estimate carries variance

    Var[z(t)] ~= (log t)^2 * Var[alpha] + floor

which falls to the noise floor as t -> 1. That single line is the whole point:
the model *derives* "early readings are nearly worthless, closing readings are
nearly exact" from the geometry of the curve, instead of asserting it with a
hand-tuned confidence ramp.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..config import ScoreWeights
from ..models import (BNII, EMPLOYEE, IPO, NII, QIB, RETAIL, SHAREHOLDER,
                      SNII, SubscriptionSnapshot)
from ..store import Store
from ..util import clamp

log = logging.getLogger("ipo_radar.bayes")

# --------------------------------------------------------------- constants
# Prior median subscription (log-space) for a typical mainboard issue, with
# the prior spread that goes with it. QIB and NII are wide because they are
# genuinely bimodal - institutions either show up in force or not at all.
PRIOR_LOG_MEDIAN: dict[str, float] = {
    QIB: math.log(8.0), NII: math.log(10.0), SNII: math.log(12.0),
    BNII: math.log(9.0), RETAIL: math.log(4.0), EMPLOYEE: math.log(2.0),
    SHAREHOLDER: math.log(2.5),
}
PRIOR_LOG_SD: dict[str, float] = {
    QIB: 1.45, NII: 1.50, SNII: 1.50, BNII: 1.55, RETAIL: 1.05,
    EMPLOYEE: 0.90, SHAREHOLDER: 1.10,
}
# Uncertainty in the power-law exponent itself. This is what makes early
# observations weak: Var[z] scales with (log t)^2 * VAR_ALPHA.
VAR_ALPHA: dict[str, float] = {
    QIB: 0.55 ** 2, BNII: 0.45 ** 2, NII: 0.40 ** 2, SNII: 0.35 ** 2,
    RETAIL: 0.16 ** 2, EMPLOYEE: 0.20 ** 2, SHAREHOLDER: 0.20 ** 2,
}
ALPHA: dict[str, float] = {
    QIB: 3.4, BNII: 2.8, NII: 2.4, SNII: 2.0, RETAIL: 1.15,
    EMPLOYEE: 1.0, SHAREHOLDER: 1.1,
}
_Z90 = 1.2815515655446004   # standard normal 90th percentile
OBS_FLOOR = 0.02          # irreducible measurement noise on any reading
# Only Retail Individual Investors may revise down or withdraw before close,
# so retail's committed book is a soft floor, not the hard one that binds QIB
# and NII under the SEBI ICDR rules.
RETAIL_WITHDRAWAL_FLOOR = 0.85
PROCESS_VAR = 0.004       # per-hour drift: demand regimes really do change
CORR_INFLATE = 2.4        # cumulative readings are not independent samples
MIN_T = 0.04              # below this the book carries no information at all

# ------------------------------------------------- structural break filter
# The t**alpha model assumes bidding is a continuous, back-loaded process. It
# is not robust to a regime break: when the macro tape turns and the grey
# market gaps down mid-window, institutions revoke bids that were already
# placed, and the curve stops describing the process at all.
#
# Treating that as ordinary observational noise is the specific failure mode
# this guards against - the filter would keep tightening around a trajectory
# the market has abandoned. On detection the observation covariance is
# inflated and the mean is pulled back toward the size-bracket prior, which
# is the only component of the prior that a demand shock does not invalidate.
SHOCK_GMP_DROP_PCT = -10.0     # day-2 -> day-3 grey market move
SHOCK_MACRO_STRESS = 0.35      # regime stress above which the tape is hostile
SHOCK_VAR_MULTIPLIER = 3.0     # P <- P * 3
SHOCK_PRIOR_PULL = 0.45        # weight pulled back onto the conservative prior


@dataclass
class ShockSignal:
    active: bool = False
    gmp_change_pct: float | None = None
    macro_stress: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"active": self.active,
                "gmp_change_pct": self.gmp_change_pct,
                "macro_stress": round(self.macro_stress, 3),
                "reason": self.reason}


def gmp_change_pct(series: list[tuple[str, float]],
                   window_hours: float = 24.0) -> float | None:
    """
    Percentage move in grey-market premium over the trailing window.

    Percentage, not absolute: a Rs 5 fall means very different things on a
    Rs 20 premium and a Rs 200 one.
    """
    pts = [(t, v) for t, v in (series or []) if isinstance(v, (int, float))]
    if len(pts) < 2:
        return None
    try:
        latest_ts = datetime.fromisoformat(pts[-1][0])
    except Exception:
        return None
    latest = pts[-1][1]
    baseline = None
    for ts, v in reversed(pts[:-1]):
        try:
            age = (latest_ts - datetime.fromisoformat(ts)).total_seconds() / 3600.0
        except Exception:
            continue
        baseline = v
        if age >= window_hours:
            break
    if baseline is None or abs(baseline) < 1e-9:
        return None
    return (latest - baseline) / abs(baseline) * 100.0


def detect_structural_break(gmp_series: list[tuple[str, float]] | None,
                            macro_stress: float = 0.0,
                            bid_progress: float = 0.0) -> ShockSignal:
    """
    A dislocation needs BOTH legs: the grey market gapping down AND a hostile
    macro tape. Either alone is ordinary variance the Kalman filter already
    handles; together they mark the point where the power-law model stops
    describing the process.
    """
    change = gmp_change_pct(gmp_series)
    sig = ShockSignal(gmp_change_pct=None if change is None else round(change, 1),
                      macro_stress=float(macro_stress or 0.0))
    if change is None:
        sig.reason = "no GMP history to measure velocity"
        return sig
    if bid_progress < 0.30:
        sig.reason = "too early in the window to call a structural break"
        return sig
    gmp_broken = change <= SHOCK_GMP_DROP_PCT
    macro_hostile = sig.macro_stress >= SHOCK_MACRO_STRESS
    if gmp_broken and macro_hostile:
        sig.active = True
        sig.reason = (f"GMP {change:+.0f}% into the close with macro stress "
                      f"{sig.macro_stress:.2f} - institutional bids at risk of "
                      f"revocation; power-law continuity no longer assumed")
    elif gmp_broken:
        sig.reason = (f"GMP {change:+.0f}% but the macro tape is calm - treated "
                      f"as ordinary variance")
    elif macro_hostile:
        sig.reason = (f"macro stress {sig.macro_stress:.2f} but the grey market "
                      f"is holding - no break")
    else:
        sig.reason = "no dislocation"
    return sig


def _z(q: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        s = math.sqrt(-2 * math.log(q))
        return (((((c[0]*s+c[1])*s+c[2])*s+c[3])*s+c[4])*s+c[5]) / \
               ((((d[0]*s+d[1])*s+d[2])*s+d[3])*s+1)
    if q > phigh:
        s = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*s+c[1])*s+c[2])*s+c[3])*s+c[4])*s+c[5]) / \
                ((((d[0]*s+d[1])*s+d[2])*s+d[3])*s+1)
    s = q - 0.5
    r = s * s
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*s / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------- posterior
@dataclass
class Posterior:
    """Log-normal belief about one category's final subscription multiple."""
    category: str
    log_mean: float
    log_var: float
    observed: float = 0.0          # the book right now - a hard lower bound
    n_updates: int = 0
    prior_log_mean: float = 0.0
    prior_log_var: float = 0.0
    drivers: dict[str, float] = field(default_factory=dict)

    @property
    def log_sd(self) -> float:
        return math.sqrt(max(self.log_var, 1e-9))

    @property
    def median(self) -> float:
        return max(self.observed, math.exp(self.log_mean))

    @property
    def mean(self) -> float:
        return max(self.observed, math.exp(self.log_mean + self.log_var / 2.0))

    def quantile(self, q: float) -> float:
        return max(self.observed, math.exp(self.log_mean + self.log_sd * _z(q)))

    def interval(self, level: float = 0.80) -> tuple[float, float]:
        tail = (1.0 - level) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    def p_above(self, x: float) -> float:
        """P(final subscription > x). The number that decides applications."""
        if x <= 0:
            return 1.0
        if x <= self.observed:
            return 1.0
        return 1.0 - _norm_cdf((math.log(x) - self.log_mean) / self.log_sd)

    @property
    def confidence(self) -> float:
        """0-1, monotone in posterior precision. Feeds scoring and sizing."""
        return clamp(1.0 / (1.0 + 2.0 * self.log_var), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.interval(0.80)
        return {"category": self.category, "median": round(self.median, 2),
                "mean": round(self.mean, 2), "p10": round(self.quantile(0.10), 2),
                "p90": round(self.quantile(0.90), 2),
                "ci80": [round(lo, 2), round(hi, 2)],
                "log_sd": round(self.log_sd, 3),
                "confidence": round(self.confidence, 3),
                "observed": round(self.observed, 2),
                "n_updates": self.n_updates,
                "prior_median": round(math.exp(self.prior_log_mean), 2),
                "drivers": {k: round(v, 3) for k, v in self.drivers.items()}}


# ------------------------------------------------------------------ priors
def size_bracket(issue_size_cr: float | None) -> str:
    if issue_size_cr is None:
        return "unknown"
    if issue_size_cr < 500:
        return "small"
    if issue_size_cr <= 1500:
        return "mid"
    return "large"


# Small issues clear far more easily than large ones: the same rupee of retail
# appetite is spread over a much smaller book.
SIZE_ADJ = {"small": 0.48, "mid": 0.0, "large": -0.42, "unknown": 0.0}


def anchor_quality(metrics: dict[str, Any] | None) -> float | None:
    """
    0-1 quality of the anchor book: what share of it is marquee long-only or
    sovereign money rather than opportunistic allocation. None when no anchor
    letter has been parsed - absence of evidence is not a zero.
    """
    if not metrics:
        return None
    count = metrics.get("anchor_investor_count")
    hits = metrics.get("anchor_marquee_hits")
    if not isinstance(count, (int, float)) or count <= 0:
        return None
    if not isinstance(hits, (int, float)):
        return None
    return clamp(float(hits) / float(count), 0.0, 1.0)


class BayesianProjectionEngine:
    """
    Posterior over final subscription per category.

    Usage mirrors the deterministic projector so it can sit behind the same
    call sites:  engine.project(ipo, snapshot) -> {category: multiple}
    with posteriors(), confidence() and demand weighting available alongside.
    """

    # When macro conditions deteriorate, institutional bids become far less
    # predictable from the shape of the book so far: anchors and QIBs pull or
    # trim at the last hour. The curve model is exactly as wrong as before,
    # but we should be less sure of it - so observation noise is inflated
    # rather than the central estimate being nudged.
    STRESS_NOISE_GAIN = {QIB: 1.8, BNII: 1.5, NII: 1.4, SNII: 1.1,
                         RETAIL: 0.35, EMPLOYEE: 0.2, SHAREHOLDER: 0.4}

    def __init__(self, store: Store | None = None,
                 macro_stress: float = 0.0) -> None:
        self.store = store
        self.macro_stress = clamp(macro_stress, 0.0, 1.0)
        self._last: dict[str, Posterior] = {}
        self.shock = ShockSignal()

    def set_macro_stress(self, stress: float) -> None:
        self.macro_stress = clamp(float(stress or 0.0), 0.0, 1.0)

    def assess_shock(self, ipo: IPO, now: datetime | None = None) -> ShockSignal:
        """Run the structural-break test for one issue and cache the verdict."""
        series = (self.store.gmp_series(ipo.symbol, limit=60)
                  if self.store else [])
        self.shock = detect_structural_break(series, self.macro_stress,
                                             ipo.bid_progress(now))
        if self.shock.active:
            log.warning("%s: structural break detected - %s",
                        ipo.symbol, self.shock.reason)
        return self.shock

    def _conservative_prior(self, ipo: IPO, category: str) -> float:
        """
        Size-bracket prior only.

        A demand shock invalidates the anchor-quality and momentum components
        of the prior - both are readings of the very enthusiasm that just
        evaporated. Issue size does not change, so it is the only anchor left
        that the shock has not contaminated.
        """
        base = PRIOR_LOG_MEDIAN.get(category, math.log(5.0))
        return base + SIZE_ADJ[size_bracket(ipo.issue_size_cr)]

    # ------------------------------------------------------------- prior
    def build_prior(self, ipo: IPO, category: str, *,
                    metrics: dict[str, Any] | None = None,
                    day1_retail: float | None = None) -> tuple[float, float, dict]:
        base = PRIOR_LOG_MEDIAN.get(category, math.log(5.0))
        sd = PRIOR_LOG_SD.get(category, 1.3)
        drivers: dict[str, float] = {}

        # 1. issue size bracket
        bracket = size_bracket(ipo.issue_size_cr)
        adj_size = SIZE_ADJ[bracket]
        drivers["size_" + bracket] = adj_size

        # 2. anchor book quality - strongest signal for the institutional
        #    tranche, weaker but real for everyone else (retail follows QIB)
        adj_anchor = 0.0
        q = anchor_quality(metrics)
        if q is not None:
            weight = 1.05 if category in (QIB, NII, BNII) else 0.45
            adj_anchor = (q - 0.30) * weight
            drivers["anchor_quality"] = round(q, 3)
            drivers["anchor_adj"] = adj_anchor
            sd *= 0.92                       # a parsed anchor book sharpens us

        # 3. day-1 retail momentum: the earliest honest read on general demand.
        #    Retail bids early and broadly, so it leads the other categories.
        adj_mom = 0.0
        if day1_retail is not None and day1_retail >= 0:
            ratio = (day1_retail + 0.20) / 1.10        # ~1.0 is a normal day 1
            adj_mom = clamp(math.log(max(ratio, 0.05)), -0.95, 1.35)
            if category != RETAIL:
                adj_mom *= 0.75
            drivers["day1_retail"] = round(day1_retail, 3)
            drivers["momentum_adj"] = adj_mom
            sd *= 0.90

        mu = base + adj_size + adj_anchor + adj_mom
        return mu, sd ** 2, drivers

    # -------------------------------------------------------------- update
    def posterior(self, ipo: IPO, category: str, series: list[tuple[str, float]],
                  *, metrics: dict[str, Any] | None = None,
                  day1_retail: float | None = None,
                  now: datetime | None = None) -> Posterior:
        mu, var, drivers = self.build_prior(ipo, category, metrics=metrics,
                                            day1_retail=day1_retail)
        post = Posterior(category=category, log_mean=mu, log_var=var,
                         prior_log_mean=mu, prior_log_var=var, drivers=drivers)
        if not series:
            return post

        alpha = ALPHA.get(category, 1.8)
        var_alpha = VAR_ALPHA.get(category, 0.30 ** 2)
        obs = self._hourly(series)
        post.observed = obs[-1][1] if obs else 0.0

        prev_t: float | None = None
        for ts, x in obs:
            try:
                t = ipo.bid_progress(datetime.fromisoformat(ts))
            except Exception:
                continue
            if t < MIN_T or x <= 0:
                continue
            if t >= 1.0:
                # The book is final. Collapse the posterior onto the truth.
                post.log_mean, post.log_var = math.log(x), OBS_FLOOR / 8.0
                post.n_updates += 1
                post.observed = x
                prev_t = t
                continue

            # curve observation: z = log x(t) - alpha*log t
            z = math.log(x) - alpha * math.log(t)
            r = (math.log(t) ** 2) * var_alpha + OBS_FLOOR
            r *= CORR_INFLATE          # successive cumulative reads overlap
            # retail behaviour is sticky under stress; institutions are not
            r *= 1.0 + self.macro_stress * self.STRESS_NOISE_GAIN.get(category, 0.8)

            if prev_t is not None:
                post.log_var += PROCESS_VAR * max(0.0, (t - prev_t)) * 21.0
            post.log_mean, post.log_var = self._kalman(post.log_mean,
                                                       post.log_var, z, r)
            post.n_updates += 1
            prev_t = t

        # velocity observation: extrapolate the recent fill rate to the bell
        vel = self._velocity_observation(ipo, obs, now)
        if vel is not None:
            z_v, r_v = vel
            r_v *= 1.0 + self.macro_stress * self.STRESS_NOISE_GAIN.get(
                category, 0.8)
            post.log_mean, post.log_var = self._kalman(post.log_mean,
                                                       post.log_var, z_v, r_v)
            post.n_updates += 1
            drivers["velocity_proj"] = round(math.exp(z_v), 2)

        # --- structural break: widen, and step back toward the safe prior
        # A closed book has no remaining flow to be uncertain about. The
        # t >= 1.0 branch above collapses the posterior onto the final number,
        # and re-inflating its variance afterwards would manufacture doubt
        # about a fact.
        window_shut = ipo.bid_progress(now) >= 1.0
        if self.shock.active and not window_shut:
            # Widening alone would make the outcome MORE optimistic: for a
            # log-normal, E[X] = exp(m + v/2), so tripling v raises both the
            # expectation and P(blowout). Subtracting half the ADDED variance
            # from the mean widens the distribution without inflating it - a
            # dislocation must never improve the odds.
            v_old = post.log_var
            added = v_old * (SHOCK_VAR_MULTIPLIER - 1.0)
            post.log_var += added
            # Shift by the amount that holds the UPPER quantile still, not
            # merely the mean. Compensating by added/2 preserves E[X] but the
            # p90 and the top of the credible interval still rise, so a
            # declared dislocation would read as a larger upside than before.
            post.log_mean -= _Z90 * (math.sqrt(post.log_var) - math.sqrt(v_old))
            safe = self._conservative_prior(ipo, category)
            # only ever pulls DOWNWARD - a dislocation is not a reason to
            # raise an estimate that the live book has already exceeded
            if safe < post.log_mean:
                post.log_mean = ((1.0 - SHOCK_PRIOR_PULL) * post.log_mean
                                 + SHOCK_PRIOR_PULL * safe)
            drivers["structural_break"] = 1.0
            drivers["shock_gmp_change_pct"] = self.shock.gmp_change_pct or 0.0

        # The book already in hand is a hard floor on the final number, and
        # it deliberately outranks the shock pull above.
        #
        # This is a SEBI constraint, not a modelling preference: QIB and NII
        # bids cannot be withdrawn or revised downward once submitted. Only
        # retail may revise down or withdraw, and then only until the closing
        # date. So a late dislocation cannot shrink an institutional book that
        # is already committed - it can only change how much MORE arrives.
        #
        # Consequence worth understanding: on an issue whose book is already
        # large, the shock's downward pull is fully absorbed by this floor and
        # only the variance inflation survives. That is the correct outcome -
        # the central estimate is pinned by committed bids while uncertainty
        # about the remaining flow widens.
        if post.observed > 0:
            # RETAIL is the exception the rule above describes: RIIs may revise
            # down or withdraw until the closing date, so their book is not a
            # hard floor. A soft floor keeps it anchored without pretending
            # committed-bid rules apply to a category they do not cover.
            floor = (post.observed * RETAIL_WITHDRAWAL_FLOOR
                     if category == RETAIL else post.observed)
            post.log_mean = max(post.log_mean, math.log(floor))
        if self.macro_stress > 0.05:
            drivers["macro_stress"] = round(self.macro_stress, 3)
        return post

    @staticmethod
    def _kalman(m: float, p: float, z: float, r: float) -> tuple[float, float]:
        """Scalar Kalman/Bayesian update for a Gaussian state and observation."""
        k = p / (p + r)
        return m + k * (z - m), (1.0 - k) * p

    @staticmethod
    def _hourly(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """One reading per clock hour - the last of each - so a fast poll
        cadence does not masquerade as extra independent evidence."""
        buckets: dict[str, tuple[str, float]] = {}
        for ts, x in series:
            key = ts[:13]                    # yyyy-mm-ddThh
            buckets[key] = (ts, x)
        return [buckets[k] for k in sorted(buckets)]

    def _velocity_observation(self, ipo: IPO, obs: list[tuple[str, float]],
                              now: datetime | None) -> tuple[float, float] | None:
        if len(obs) < 3:
            return None
        try:
            recent = obs[-min(5, len(obs)):]
            ta = datetime.fromisoformat(recent[0][0])
            tb = datetime.fromisoformat(recent[-1][0])
            hours = (tb - ta).total_seconds() / 3600.0
            if hours <= 0.2:
                return None
            rate = (recent[-1][1] - recent[0][1]) / hours
        except Exception:
            return None
        if rate <= 0:
            return None
        left = hours_remaining(ipo, now)
        if left is None or left <= 0:
            return None
        t = ipo.bid_progress(now)
        surge = 1.0 + 2.2 * max(0.0, t - 0.55)      # bids pile up at the close
        projected = recent[-1][1] + rate * left * surge
        if projected <= 0:
            return None
        # trust extrapolation more the less there is left to extrapolate
        r = 0.35 + 2.6 * (1.0 - t) ** 2
        return math.log(projected), r

    # ----------------------------------------------------------- interface
    def posteriors(self, ipo: IPO, sub: SubscriptionSnapshot | None,
                   *, metrics: dict[str, Any] | None = None,
                   now: datetime | None = None) -> dict[str, Posterior]:
        if not sub:
            return {}
        day1 = self._day1_retail(ipo)
        out: dict[str, Posterior] = {}
        for cat in sub.categories:
            series = (self.store.subscription_series(ipo.symbol, cat)
                      if self.store else [])
            if not series:
                series = [(sub.ts, sub.times(cat))]
            out[cat] = self.posterior(ipo, cat, series, metrics=metrics,
                                      day1_retail=day1, now=now)
        self._last = out
        return out

    def prior_projection(self, ipo: IPO,
                         metrics: dict[str, Any] | None = None
                         ) -> dict[str, Posterior]:
        """
        Prior-only posteriors for an issue whose book has not opened.

        Capital has to be reserved for a forthcoming issue BEFORE any
        subscription data exists, and the prior already encodes what is
        knowable then: issue size, anchor-book quality and day-one retail
        momentum from comparable issues. The posterior is simply the prior,
        so its confidence is correspondingly low and everything downstream
        widens accordingly.
        """
        out: dict[str, Posterior] = {}
        for cat in (QIB, NII, SNII, BNII, RETAIL):
            mu, var, drivers = self.build_prior(ipo, cat, metrics=metrics)
            drivers["prior_only"] = 1.0
            out[cat] = Posterior(category=cat, log_mean=mu, log_var=var,
                                 prior_log_mean=mu, prior_log_var=var,
                                 drivers=drivers)
        self._last = out
        return out

    def project(self, ipo: IPO, sub: SubscriptionSnapshot | None,
                now: datetime | None = None,
                metrics: dict[str, Any] | None = None) -> dict[str, float]:
        """Drop-in replacement for the deterministic projector's `project`.

        Returns posterior MEDIANS - the central estimate - while the full
        distribution stays available through `posteriors()`."""
        return {c: p.median
                for c, p in self.posteriors(ipo, sub, metrics=metrics,
                                            now=now).items()}

    def confidence(self, ipo: IPO, now: datetime | None = None) -> float:
        """Posterior-derived confidence, weighted towards the categories that
        actually drive decisions."""
        if not self._last:
            from .subscription import projection_confidence
            return projection_confidence(ipo.bid_progress(now))
        weights = {QIB: 0.45, RETAIL: 0.30, NII: 0.15, SNII: 0.10, BNII: 0.10}
        num = den = 0.0
        for cat, post in self._last.items():
            w = weights.get(cat, 0.05)
            num += w * post.confidence
            den += w
        return clamp(num / den, 0.0, 1.0) if den else 0.35

    def _day1_retail(self, ipo: IPO) -> float | None:
        """Retail multiple as at the end of day 1 - the momentum prior."""
        if not self.store:
            return None
        series = self.store.subscription_series(ipo.symbol, RETAIL)
        if not series:
            return None
        best: float | None = None
        for ts, x in series:
            try:
                t = ipo.bid_progress(datetime.fromisoformat(ts))
            except Exception:
                continue
            if t <= 0.42:                    # roughly through the first day
                best = x
        return best

    def describe(self, ipo: IPO) -> dict[str, Any]:
        if not self._last:
            return {}
        return {c: p.to_dict() for c, p in self._last.items()}


# ------------------------------------------------------- dynamic weighting
def hours_remaining(ipo: IPO, now: datetime | None = None) -> float | None:
    """Bidding hours left (10:00-17:00 windows only)."""
    now = now or datetime.now()
    _, close = ipo.days()
    if not close:
        return None
    end = datetime.combine(close, datetime.min.time()).replace(hour=17)
    if now >= end:
        return 0.0
    hours, cur = 0.0, now.date()
    while cur <= close:
        d_start = datetime.combine(cur, datetime.min.time()).replace(hour=10)
        d_end = datetime.combine(cur, datetime.min.time()).replace(hour=17)
        lo = max(now, d_start)
        if lo < d_end:
            hours += (d_end - lo).total_seconds() / 3600.0
        cur = cur.fromordinal(cur.toordinal() + 1)
    return hours


DEMAND_W_MIN = 5.0        # day 1: the book says almost nothing
DEMAND_W_MAX = 25.0       # day 3 afternoon: the book is nearly the whole story
FINAL_PUSH_HOUR = 14      # 2 PM on the closing day


def demand_weight(ipo: IPO, now: datetime | None = None) -> float:
    """
    How much the composite score should lean on demand right now.

    Early in the window the book is noise and the score should rest on
    fundamentals; by the closing afternoon the book is the single most
    informative thing in existence. Ramps 5 -> 25.
    """
    now = now or datetime.now()
    _, close = ipo.days()
    if close and now.date() == close and now.hour >= FINAL_PUSH_HOUR:
        return DEMAND_W_MAX
    if close and now.date() > close:
        return DEMAND_W_MAX
    t = ipo.bid_progress(now)
    ramp = clamp((t - 0.10) / 0.80, 0.0, 1.0) ** 0.85
    return DEMAND_W_MIN + (DEMAND_W_MAX - DEMAND_W_MIN) * ramp


def dynamic_weights(base: ScoreWeights, ipo: IPO,
                    now: datetime | None = None) -> tuple[ScoreWeights, float]:
    """
    Rebalance the score weights for where we are in the bid window.

    Whatever weight demand gives up early is handed to valuation and
    financials - the factors that are fully knowable on day one because they
    come out of the RHP, not the order book. Total stays at 100.
    """
    from dataclasses import replace
    target = demand_weight(ipo, now)
    freed = base.demand - target
    val, fin = base.valuation, base.financials
    denom = val + fin
    if denom <= 0:
        return base, target
    scaled = replace(base, demand=target,
                     valuation=val + freed * (val / denom),
                     financials=fin + freed * (fin / denom))
    return scaled, target
