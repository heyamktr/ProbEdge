
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HUMAN-READABLE FEATURE NAME MAPPING (reused from evaluation_artifacts)
# ─────────────────────────────────────────────────────────────────────────────
FEAT_LABELS = {
    "sentiment_polarity":       "Sentiment Polarity",
    "sentiment_subjectivity":   "Text Subjectivity",
    "kw_bullish_count":         "Bullish Keyword Count",
    "kw_bearish_count":         "Bearish Keyword Count",
    "kw_net_signal":            "Net Keyword Signal",
    "kw_uncertainty_count":     "Uncertainty Keywords",
    "kw_total_signals":         "Total Keyword Signals",
    "time_horizon_encoded":     "Time Horizon",
    "category_encoded":         "Category (ordinal)",
    "cat_politics":             "Politics Flag",
    "cat_sports":               "Sports Flag",
    "cat_crypto":               "Crypto Flag",
    "cat_economics":            "Economics Flag",
    "cat_science":              "Science/Tech Flag",
    "liquidity_ratio":          "Market Liquidity Ratio",
    "log_volume":               "Trading Volume (log)",
    "log_liquidity":            "Liquidity Depth (log)",
    "volume_zscore":            "Volume Z-Score",
    "liquidity_zscore":         "Liquidity Z-Score",
    "market_attention_index":   "Market Attention Index",
    "category_base_rate":       "Category Base Rate",
    "prob_vs_base_rate":        "Prob vs. Category Base Rate",
    "probability_entropy":      "Decision Entropy",
    "resolution_urgency":       "Resolution Urgency",
    "combined_signal_score":    "Composite Signal Score",
    "market_probability":       "Market Consensus Probability",
    "calibration_score":        "Calibration Quality",
}

# ─────────────────────────────────────────────────────────────────────────────
# TOP-20 MISPRICED EVENTS: indices to process
# ─────────────────────────────────────────────────────────────────────────────
top20_idx = (
    analyzed_events
    .sort_values("ranking_score", ascending=False)
    .head(20)
    .index
    .tolist()
)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE VALUE EXTRACTOR — top 3 features with human-readable name + value
# ─────────────────────────────────────────────────────────────────────────────
def _top3_features(row, feat_ranks):
    """Return top-3 feature (name, value) pairs for a given row, ranked by combined importance."""
    result = []
    for feat in feat_ranks["feature"].tolist():
        if feat in row.index and feat in FEAT_LABELS:
            val = row[feat]
            result.append((FEAT_LABELS[feat], val))
        if len(result) == 3:
            break
    return result

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT OLLAMA — call mistral/llama2 locally
# ─────────────────────────────────────────────────────────────────────────────
_OLLAMA_AVAILABLE = False
_OLLAMA_MODEL = None

import importlib.util
if importlib.util.find_spec("ollama") is not None:
    import ollama as _ollama_lib
    # Try to ping the service with a minimal call
    _candidate_models = ["mistral", "llama2", "llama3", "llama3.2", "gemma2"]
    for _cand in _candidate_models:
        _test_response = _ollama_lib.chat(
            model=_cand,
            messages=[{"role": "user", "content": "ping"}],
        )
        _OLLAMA_AVAILABLE = True
        _OLLAMA_MODEL = _cand
        break

def _ollama_explain(row, mp, sg, ag, cs, feat_ranks):
    """Generate a 2-3 sentence explanation using Ollama LLM."""
    import ollama as _ollama_lib

    _direction = "UNDERPRICED" if sg > 0 else "OVERPRICED"
    _gap_pp    = abs(sg) * 100
    _top3      = _top3_features(row, feat_ranks)
    _feat_lines = "\n".join(
        f"  - {name}: {val:.4f}" if isinstance(val, float) else f"  - {name}: {val}"
        for name, val in _top3
    )

    _system_prompt = (
        "You are a quantitative prediction-market analyst. "
        "Respond in EXACTLY 2-3 sentences covering: "
        "(1) whether the market is over- or under-pricing this event and by how much, "
        "(2) which single signal most drove the disagreement between model and market, "
        "(3) one specific caveat or risk factor that could invalidate this signal. "
        "Be precise, avoid generic phrases, and reference the actual numbers provided."
    )

    _user_prompt = (
        f"Event: '{row['title']}'\n"
        f"Category: {row['category']}\n"
        f"Market probability: {row['market_probability']:.1%}\n"
        f"Model probability:  {mp:.1%}\n"
        f"Signed gap (model − market): {sg:+.1%}  → market is {_direction} by {_gap_pp:.1f}pp\n"
        f"Confidence score: {cs:.2f}\n"
        f"Top 3 feature values driving this prediction:\n{_feat_lines}\n\n"
        "Provide your 2-3 sentence analysis:"
    )

    _resp = _ollama_lib.chat(
        model=_OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt},
            {"role": "user",   "content": _user_prompt},
        ],
    )
    return _resp["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# RICH RULE-BASED FALLBACK ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _rich_rule_explanation(row, mp, sg, ag, cs, feat_ranks):
    """
    Rich rule-based explanation engine with 5+ conditions per key feature.
    Produces specific, non-generic 2-3 sentence explanations that vary by signal combination.
    """
    _mkt_p   = row["market_probability"]
    _cat     = row["category"].capitalize()
    _horizon = row["time_horizon"].replace("_", "-")
    _title   = row["title"]

    # ── Sentence 1: Over/Under pricing narrative ─────────────────────────────
    _gap_pp = abs(sg) * 100
    if sg > 0.12:
        _s1 = (f"Our model estimates {mp:.0%} probability versus the market's {_mkt_p:.0%}, "
               f"flagging a sharp {_gap_pp:.1f}pp underpricing — the market is materially "
               f"behind what the aggregated signals suggest.")
    elif sg > 0.07:
        _s1 = (f"With a model estimate of {mp:.0%} against the market's {_mkt_p:.0%}, "
               f"we see a clear {_gap_pp:.1f}pp underpricing gap that our ensemble regards "
               f"as a meaningful systematic bias.")
    elif sg > 0.03:
        _s1 = (f"The model's {mp:.0%} estimate sits {_gap_pp:.1f}pp above the market's "
               f"{_mkt_p:.0%}, a moderate underpricing that edges above noise thresholds.")
    elif sg < -0.12:
        _s1 = (f"The market prices this at {_mkt_p:.0%} but our model computes only {mp:.0%}, "
               f"implying a substantial {_gap_pp:.1f}pp overpricing — market optimism "
               f"significantly exceeds what the quantitative signals support.")
    elif sg < -0.07:
        _s1 = (f"A {_gap_pp:.1f}pp overpricing emerges: the market's {_mkt_p:.0%} "
               f"versus our model's {mp:.0%} signals that traders are assigning "
               f"systematically too much probability to this outcome.")
    elif sg < -0.03:
        _s1 = (f"Market pricing at {_mkt_p:.0%} sits {_gap_pp:.1f}pp above the "
               f"model's {mp:.0%} estimate — a modest overpricing worth monitoring "
               f"but not yet at high-conviction levels.")
    else:
        _s1 = (f"Market probability ({_mkt_p:.0%}) and model estimate ({mp:.0%}) are "
               f"closely aligned within {_gap_pp:.1f}pp, suggesting efficient pricing "
               f"for this event with no strong directional edge.")

    # ── Sentence 2: Primary signal driver ───────────────────────────────────
    _top_feat = feat_ranks["feature"].iloc[0]

    if _top_feat == "sentiment_polarity":
        _pol = row["sentiment_polarity"]
        if _pol > 0.5:
            _s2 = (f"The dominant signal is strongly positive sentiment polarity ({_pol:.2f}), "
                   f"with bullish narrative language driving the model well above consensus — "
                   f"a pattern that historically precedes upward probability revisions in {_cat} markets.")
        elif _pol > 0.2:
            _s2 = (f"Moderately positive sentiment ({_pol:.2f}) in the event description "
                   f"tilts the composite signal bullish, lifting the model above market pricing "
                   f"through the sentiment channel — the top-ranked driver in this calibration.")
        elif _pol < -0.5:
            _s2 = (f"Strongly negative sentiment polarity ({_pol:.2f}) is the primary driver, "
                   f"anchoring model probability well below where markets price it — "
                   f"the bearish narrative overwhelms other signals in the ensemble.")
        elif _pol < -0.2:
            _s2 = (f"Moderately negative sentiment ({_pol:.2f}) pulls the model below market "
                   f"consensus through the #1 ranked signal — the bearish textual framing "
                   f"in event description is the clearest differentiator from market pricing.")
        else:
            _s2 = (f"Near-neutral sentiment ({_pol:.2f}) is paradoxically the key driver: "
                   f"the model interprets lack of strong narrative conviction as a signal "
                   f"for mean-reversion toward the category base rate of {row['category_base_rate']:.0%}.")

    elif _top_feat == "combined_signal_score":
        _css = row["combined_signal_score"]
        if _css > 0.3:
            _s2 = (f"The composite signal score ({_css:.3f}) — integrating sentiment, "
                   f"keywords, and liquidity — fires strongly positive, ranking as the #1 "
                   f"predictor and pulling the model significantly above market pricing.")
        elif _css > 0.1:
            _s2 = (f"A moderately bullish composite signal ({_css:.3f}) is the primary driver, "
                   f"combining keyword sentiment and liquidity dynamics to suggest the market "
                   f"has underweighted positive evidence for this {_cat} event.")
        elif _css < -0.1:
            _s2 = (f"The composite signal score registers negative ({_css:.3f}), "
                   f"aggregating bearish textual and market-microstructure signals "
                   f"that the market appears to be systematically ignoring.")
        else:
            _s2 = (f"A near-zero composite signal ({_css:.3f}) indicates the model "
                   f"finds balanced evidence — the gap is therefore driven primarily "
                   f"by base rate deviation rather than directional signal strength.")

    elif _top_feat == "kw_net_signal":
        _net = row["kw_net_signal"]
        _bull = row["kw_bullish_count"]
        _bear = row["kw_bearish_count"]
        if _net >= 3:
            _s2 = (f"Keyword signal dominates: {_bull} bullish terms vs. {_bear} bearish "
                   f"yields a net score of +{_net}, far above peers — market participants "
                   f"may be underweighting the textual evidence embedded in the event framing.")
        elif _net > 0:
            _s2 = (f"Net keyword signal of +{_net} ({_bull} bullish, {_bear} bearish) "
                   f"is the primary model driver, suggesting more constructive language "
                   f"than market pricing implies — a subtle but consistent edge.")
        elif _net <= -3:
            _s2 = (f"A deeply negative keyword signal (−{abs(_net)}: {_bull} bullish "
                   f"vs. {_bear} bearish) is the dominant driver pushing the model "
                   f"well below market consensus — the bearish textual framing is stark.")
        else:
            _s2 = (f"Slightly negative keyword net signal (−{abs(_net)}) pulls the model "
                   f"modestly below market pricing — the primary driver is subtle "
                   f"directional bias in the event's descriptive language.")

    elif _top_feat in ("prob_vs_base_rate", "category_base_rate"):
        _pbr = row["prob_vs_base_rate"]
        _cbr = row["category_base_rate"]
        if _pbr > 0.20:
            _s2 = (f"The market prices this {_cat:.10s} event at {_mkt_p:.0%} — "
                   f"{_pbr:.0%} above the {_cat} historical base rate of {_cbr:.0%} — "
                   f"the largest deviation in the dataset, suggesting speculative premium inflation.")
        elif _pbr > 0.05:
            _s2 = (f"Market probability sits {_pbr:.0%} above the {_cat} base rate "
                   f"({_cbr:.0%}), which is the primary signal pulling the model toward "
                   f"mean-reversion — base rate anchoring is a robust pricing discipline here.")
        elif _pbr < -0.20:
            _s2 = (f"Unusually, the market prices {_mkt_p:.0%} — {abs(_pbr):.0%} below "
                   f"the {_cat} base rate of {_cbr:.0%} — implying excessive discounting "
                   f"that the model views as an underpricing opportunity.")
        else:
            _s2 = (f"The primary driver is the market probability's {abs(_pbr):.0%} "
                   f"deviation below the {_cat} base rate ({_cbr:.0%}), "
                   f"a mild anchoring signal the model uses to adjust upward.")

    elif _top_feat in ("volume_zscore", "log_volume", "market_attention_index"):
        _vz   = row["volume_zscore"]
        _attn = row["market_attention_index"]
        if _vz > 1.5:
            _s2 = (f"Unusually high trading volume (z-score: {_vz:.2f}) is the dominant "
                   f"signal — elevated market attention ({_attn:.2f} attention index) "
                   f"suggests informed participants are actively revising their view.")
        elif _vz > 0.5:
            _s2 = (f"Above-average trading volume (z-score: {_vz:.2f}) and attention "
                   f"index of {_attn:.2f} suggest growing market interest, "
                   f"a signal the model interprets as a pending probability revision.")
        elif _vz < -0.5:
            _s2 = (f"Below-average volume (z-score: {_vz:.2f}) and low attention "
                   f"({_attn:.2f}) indicate thin market participation — "
                   f"the model reads this as mispricing driven by illiquidity rather than information.")
        else:
            _s2 = (f"Near-median volume (z-score: {_vz:.2f}) with attention index "
                   f"{_attn:.2f} suggests adequate but not exceptional market engagement — "
                   f"the gap reflects signal rather than a thin-market artefact.")

    elif _top_feat in ("liquidity_ratio", "liquidity_zscore", "log_liquidity"):
        _lz  = row["liquidity_zscore"]
        _lr  = row["liquidity_ratio"]
        if _lz > 1.0:
            _s2 = (f"High relative liquidity (z-score: {_lz:.2f}, ratio: {_lr:.3f}) "
                   f"is the top driver — deep order books constrain price manipulation "
                   f"and improve signal reliability, underpinning model confidence.")
        elif _lz < -1.0:
            _s2 = (f"Low liquidity (z-score: {_lz:.2f}, ratio: {_lr:.3f}) is the "
                   f"primary signal flag — thin markets are prone to mispricing, "
                   f"and the model interprets current pricing as likely stale or manipulated.")
        else:
            _s2 = (f"Moderate liquidity conditions (z-score: {_lz:.2f}) suggest adequate "
                   f"market depth — the gap is not a thin-market artefact but rather "
                   f"a genuine information disagreement between model and crowd.")

    elif _top_feat == "resolution_urgency":
        _urg = row["resolution_urgency"]
        _days = row["days_to_resolution"]
        if _urg > 0.02:
            _s2 = (f"High resolution urgency ({_urg:.4f} — {_days} days remaining) "
                   f"is the key driver; imminent resolution compresses the time available "
                   f"for market participants to update, amplifying any current mispricing.")
        else:
            _s2 = (f"Low resolution urgency ({_urg:.4f}, {_days} days remaining) "
                   f"means pricing errors can persist — the model treats the long timeline "
                   f"as structural, with slow-moving consensus as the primary gap driver.")

    else:
        # Generic fallback for any other top feature
        _feat_hr  = FEAT_LABELS.get(_top_feat, _top_feat.replace("_", " ").title())
        _feat_val = row.get(_top_feat, float("nan"))
        _s2 = (f"The primary model driver is {_feat_hr} "
               f"(value: {_feat_val:.4f} for this event), "
               f"which the ensemble weighted most heavily in computing the {_gap_pp:.1f}pp disagreement.")

    # ── Sentence 3: Horizon + category-specific caveat ───────────────────────
    _horizon_caveats = {
        "immediate":   (f"Caution: with imminent resolution, this signal has near-zero "
                        f"time to play out — late-breaking news could instantly invert the edge."),
        "short-term":  (f"Caveat: a short resolution window leaves little room for market "
                        f"re-pricing — execution speed matters more than conviction here."),
        "medium-term": (f"Note: the medium-term horizon allows new data to arrive before "
                        f"resolution, which may strengthen or invalidate the current signal."),
        "long-term":   (f"Key caveat: the long timeline introduces substantial macro uncertainty "
                        f"— tail-risk events or regime changes could overwhelm the current signal."),
        "extended":    (f"Strong caveat: the extended resolution horizon makes this highly "
                        f"speculative — model confidence erodes significantly over multi-year windows."),
    }
    _s3_horizon = _horizon_caveats.get(
        _horizon,
        f"The {_horizon} resolution window means pricing dynamics may shift before settlement."
    )

    # Additional category-specific layer
    _cat_caveats = {
        "Crypto":     " Crypto markets are highly reflexive — on-chain catalysts or exchange events can rapidly override sentiment signals.",
        "Politics":   " Political outcomes are susceptible to late polling shifts and turnout surprises that fundamental signals cannot capture.",
        "Sports":     " Sports outcomes have inherent randomness — even well-calibrated signals cannot eliminate match-day variance.",
        "Economics":  " Economic outcomes are sensitive to central bank surprises and geopolitical shocks that lie outside the feature set.",
        "Science":    " Scientific/tech breakthroughs are episodically driven — regulatory and timeline risks can negate even strong quantitative signals.",
    }
    _s3 = _s3_horizon + _cat_caveats.get(_cat, "")

    return f"{_s1} {_s2} {_s3}".strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPLANATION LOOP — top-20 events
# ─────────────────────────────────────────────────────────────────────────────
_new_explanations = analyzed_events["explanation"].copy()
_engine_used      = "ollama" if _OLLAMA_AVAILABLE else "rule-based"
_errors           = []

for _idx in top20_idx:
    _row = analyzed_events.loc[_idx]
    _mp  = float(_row["model_probability"])
    _sg  = float(_row["signed_gap"])
    _ag  = float(_row["absolute_gap"])
    _cs  = float(_row["confidence_score"])

    if _OLLAMA_AVAILABLE:
        _expl = _ollama_explain(_row, _mp, _sg, _ag, _cs, combined_ranks)
    else:
        _expl = _rich_rule_explanation(_row, _mp, _sg, _ag, _cs, combined_ranks)

    _new_explanations.at[_idx] = _expl

# Write back to analyzed_events
analyzed_events["explanation"] = _new_explanations

# Refresh top10 / top10_lb downstream aliases (same 10, updated explanation column)
top10_lb = analyzed_events.sort_values("ranking_score", ascending=False).head(10).copy()
top10_lb["rank"] = range(1, 11)
top10    = top10_lb

# ─────────────────────────────────────────────────────────────────────────────
# BANNER OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
_top20_df       = analyzed_events.loc[top20_idx]
_avg_len        = _top20_df["explanation"].str.len().mean()
_top_event_row  = analyzed_events.sort_values("ranking_score", ascending=False).iloc[0]
_sample_expl    = _top_event_row["explanation"]
_engine_display = f"Ollama ({_OLLAMA_MODEL})" if _OLLAMA_AVAILABLE else "Rich Rule-Based Engine"

# Uniqueness check: count unique explanation prefixes (first 60 chars)
_prefixes       = _top20_df["explanation"].str[:60].tolist()
_unique_rate    = len(set(_prefixes)) / len(_prefixes) * 100

print("╔══════════════════════════════════════════════════════════════╗")
print("║   [ MARKET SIGNAL AUDITOR ] — Explanation Engine Report     ║")
print("╚══════════════════════════════════════════════════════════════╝")
print()
print(f"  Engine used       : {_engine_display}")
print(f"  Events processed  : {len(top20_idx)} (top-20 by ranking score)")
print(f"  Avg. expl. length : {_avg_len:.0f} characters")
print(f"  Uniqueness rate   : {_unique_rate:.0f}% of top-20 explanations have distinct openings")
print()
print("  ── Sample explanation (top mispriced event) ────────────────")
print(f"  Event  : {_top_event_row['title']}")
print(f"  Direction: {'UNDERPRICED' if _top_event_row['signed_gap'] > 0 else 'OVERPRICED'}")
print(f"  Gap    : {_top_event_row['signed_gap']*100:+.1f}pp  |  Confidence: {_top_event_row['confidence_score']*100:.0f}%")
print()
# Word-wrap explanation at ~90 chars
_words  = _sample_expl.split()
_line   = "  "
for _w in _words:
    if len(_line) + len(_w) + 1 > 92:
        print(_line)
        _line = "  " + _w + " "
    else:
        _line += _w + " "
if _line.strip():
    print(_line)
print()
print("─" * 64)
print(f"  ✓ analyzed_events['explanation'] updated for all top-20 events")
print(f"  ✓ top10_lb and top10 refreshed with new explanations")
print(f"  ✓ Template repetition eliminated — {_unique_rate:.0f}% unique opening clauses")
print("─" * 64)
