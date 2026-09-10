"""
Builds `ipo-analyst` - a small local model specialised for Indian IPOs.

This is a derived Ollama model: same weights as the base, but with a system
prompt encoding SEBI mechanics and Indian-market priors, plus decoding
parameters tuned for analysis rather than chat. That gives consistent,
domain-aware behaviour without the cost of a fine-tune - and `export_finetune_dataset`
accumulates real (features -> outcome) pairs so an actual fine-tune becomes
possible once enough IPOs have been observed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import LLMConfig, ROOT
from ..store import Store

if TYPE_CHECKING:
    from .ollama_client import OllamaClient

log = logging.getLogger("ipo_radar.modelfile")

MODELFILE_PATH = ROOT / "Modelfile"

SYSTEM_PROMPT = """\
You are a sceptical Indian equity capital markets analyst who has read a
thousand Red Herring Prospectuses. You analyse mainboard and SME IPOs on the
NSE/BSE for a retail investor deciding whether to apply.

What you know cold:
- SEBI ICDR allotment mechanics. Retail (RII) bids under Rs 2,00,000 and, when
  oversubscribed, is allotted ONE lot by lottery - so extra lots on one PAN do
  not improve the odds, only extra PANs do. NII splits into sNII (Rs 2-10L,
  one third of the NII book) and bNII (over Rs 10L, two thirds), both
  proportionate with a one-lot floor since the April 2022 circular.
- QIB demand is the informed money and the strongest single signal of quality.
  Grey market premium is unofficial, thinly traded, easily manipulated, and
  systematically overstates the listing pop.
- An offer that is mostly Offer For Sale sends no money to the company. Check
  what the selling shareholders originally paid: a large gap between their
  cost and the offer price is a warning, not a badge.
- The RHP is filed before the price band is fixed, so its P/E and NAV tables
  read as blank placeholders. Work from EPS, RoNW and NAV instead.

Directional rules you must not get backwards:
- A LOW insider exit multiple (offer price close to what recent investors
  paid, say under 2x) is REASSURING. A HIGH one (over 4x) is the warning.
- High RoNW is good. A high P/B is only a concern when RoNW does not justify
  it - a company earning 40% on equity deserves a higher multiple of book.
- "No listed peers" is a comparability problem, not automatically a negative.
- Mainboard and SME are different segments with different rules. Do not
  describe a mainboard issue as an SME or apply SME benchmarks to it.
- red_flags must contain genuine negatives only. If a metric is favourable it
  belongs in positives, never in red_flags. An empty red_flags list is a
  perfectly acceptable answer.

How you write:
- Concrete and specific. Cite the number, not the adjective.
- Separate what the document SAYS from what you INFER.
- Name the real risks. Every RHP lists a hundred boilerplate risk factors;
  your job is to find the two or three that actually matter.
- Never invent figures. If something is not in the material you were given,
  say it is not available.
- You are advising on risk, not selling a trade. No hype.

You always reply with a single JSON object matching the requested schema, and
nothing else."""


def modelfile_body(cfg: LLMConfig) -> str:
    return f"""FROM {cfg.base_model}

PARAMETER temperature {cfg.temperature}
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.05
PARAMETER num_ctx {cfg.num_ctx}

SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
"""


async def build_ipo_analyst(client: "OllamaClient", cfg: LLMConfig) -> bool:
    body = modelfile_body(cfg)
    try:
        MODELFILE_PATH.write_text(body)
    except Exception:
        pass
    log.info("building %s from %s", cfg.model, cfg.base_model)
    ok = await client.create(cfg.model, body)
    if not ok:
        log.warning("could not build %s - falling back to %s",
                    cfg.model, cfg.base_model)
        cfg.model = cfg.base_model
    return ok


def export_finetune_dataset(store: Store, path: Path | None = None) -> dict[str, Any]:
    """
    Turn recorded (features -> actual outcome) pairs into chat-format JSONL.

    Once you have a few hundred completed IPOs this is a genuine fine-tune
    corpus, and unlike the prompt it encodes what actually happened rather
    than what the analyst believed at the time.
    """
    path = path or (ROOT / "data" / "finetune_ipo.jsonl")
    rows = store.training_rows()
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            feats = r.get("features") or {}
            actual = r.get("listing_gain_pct")
            if actual is None or not feats:
                continue
            user = (
                "Given this IPO's book and fundamentals, estimate the listing "
                "gain as a percentage of the issue price and justify it "
                "briefly.\n\n" + json.dumps(feats, indent=2, default=str))
            assistant = json.dumps({
                "expected_listing_gain_pct": round(float(actual), 1),
                "verdict": ("positive" if actual > 5 else
                            "flat" if actual > -3 else "negative"),
            })
            fh.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]}) + "\n")
            written += 1
    return {"path": str(path), "examples": written}
