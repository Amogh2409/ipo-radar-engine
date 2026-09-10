"""JSON Schemas passed to Ollama's `format` so decoding is constrained."""
from __future__ import annotations

BUSINESS_ANALYSIS = {
    "type": "object",
    "properties": {
        "business_summary": {"type": "string"},
        "revenue_model": {"type": "string"},
        "competitive_moat": {"type": "string"},
        "moat_strength": {"type": "string",
                          "enum": ["none", "weak", "moderate", "strong"]},
        "key_risks": {"type": "array", "items": {"type": "string"},
                      "minItems": 2, "maxItems": 5},
        "red_flags": {"type": "array", "items": {"type": "string"},
                      "maxItems": 5},
        "positives": {"type": "array", "items": {"type": "string"},
                      "maxItems": 5},
        "use_of_proceeds": {"type": "string"},
        "proceeds_quality": {"type": "string",
                             "enum": ["growth", "debt repayment", "mixed",
                                      "mostly exit", "unclear"]},
        # Categorical judgements the guardrail layer range-checks. Constrained
        # to enums so a small model cannot invent a severity scale of its own.
        "litigation_severity": {"type": "string",
                                "enum": ["LOW", "MEDIUM", "CRITICAL"]},
        "governance_flag": {"type": "boolean"},
        # 0-100. Small models love to answer on a 0-1 scale here, so the
        # analyst normalises anything <= 1 before use.
        "qualitative_score": {"type": "number", "minimum": 0, "maximum": 100},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    # Everything the report renders is required; optional fields get silently
    # dropped by small models, leaving a hollow analysis.
    "required": ["business_summary", "revenue_model", "competitive_moat",
                 "moat_strength", "key_risks", "red_flags", "positives",
                 "use_of_proceeds", "proceeds_quality", "litigation_severity",
                 "governance_flag", "qualitative_score", "confidence"],
}

NEWS_SENTIMENT = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "sentiment": {"type": "number", "minimum": -1, "maximum": 1},
                    "relevant": {"type": "boolean"},
                },
                "required": ["index", "sentiment", "relevant"],
            },
        },
        "overall_sentiment": {"type": "number", "minimum": -1, "maximum": 1},
        "summary": {"type": "string"},
    },
    "required": ["items", "overall_sentiment", "summary"],
}

CRITIQUE = {
    "type": "object",
    "properties": {
        "strongest_bear_case": {"type": "string"},
        "overlooked_risks": {"type": "array", "items": {"type": "string"},
                             "maxItems": 4},
        "is_score_too_high": {"type": "boolean"},
        "suggested_score_adjustment": {"type": "number",
                                       "minimum": -20, "maximum": 10},
        "reasoning": {"type": "string"},
    },
    "required": ["strongest_bear_case", "overlooked_risks", "is_score_too_high",
                 "suggested_score_adjustment", "reasoning"],
}

SYNTHESIS = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "recommendation": {
            "type": "string",
            "enum": ["STRONG APPLY", "APPLY", "NEUTRAL", "AVOID",
                     "STRONG AVOID", "SPECULATIVE FLIP"]},
        "listing_gain_view": {"type": "string"},
        "long_term_view": {"type": "string"},
        "apply_in_category": {
            "type": "string",
            "enum": ["retail 1 lot", "retail max lots", "sNII", "bNII",
                     "do not apply"]},
        "one_line_reason": {"type": "string"},
        "conviction": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["headline", "recommendation", "listing_gain_view",
                 "long_term_view", "apply_in_category", "one_line_reason",
                 "conviction"],
}

# Fallback numeric extraction, used only when the deterministic table parsers
# in sources/rhp_tables.py come up empty. Nullable everywhere: "not stated" is
# a correct and useful answer, and far better than an invented figure.
METRIC_EXTRACTION = {
    "type": "object",
    "properties": {
        "eps_basic": {"type": ["number", "null"]},
        "eps_diluted": {"type": ["number", "null"]},
        "eps_fiscal_year": {"type": ["string", "null"]},
        "ronw_pct": {"type": ["number", "null"]},
        "nav_per_share": {"type": ["number", "null"]},
        "peer_pe_average": {"type": ["number", "null"]},
        # Enterprise-value inputs. Units are captured separately because RHP
        # tables switch between Rs million, crore and lakh without warning.
        "pat_latest_fy": {"type": ["number", "null"]},
        "ebitda_latest_fy": {"type": ["number", "null"]},
        "total_borrowings": {"type": ["number", "null"]},
        "cash_and_equivalents": {"type": ["number", "null"]},
        "amounts_unit": {"type": ["string", "null"],
                         "enum": ["million", "crore", "lakh", None]},
        "found": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["eps_basic", "eps_diluted", "ronw_pct", "nav_per_share",
                 "found", "notes"],
}
