
"""
Dashboard Variable Audit
Verifies all required variables for the Streamlit dashboard are present,
correctly typed, and documents their structure for the deployment script.

Upstream: market_signal_auditor → evaluation_artifacts (chained, no parallel conflict).
FEAT_LABELS sourced from evaluation_artifacts only.
"""
import pandas as pd
import numpy as np

print("=" * 65)
print("  DASHBOARD VARIABLE AUDIT")
print("=" * 65)

# ─── [1] analyzed_events ─────────────────────────────────────
print("\n[1] analyzed_events (DataFrame)")
print(f"    Shape        : {analyzed_events.shape}")
key_cols = [
    'event_id', 'title', 'category', 'market_probability',
    'model_probability', 'signed_gap', 'absolute_gap',
    'ranking_score', 'confidence_score', 'signal_direction',
    'mispricing_flag', 'explanation', 'time_horizon',
    'days_to_resolution', 'resolution_date'
]
_missing_key_cols = [c for c in key_cols if c not in analyzed_events.columns]
print(f"    Key cols OK  : {len(key_cols) - len(_missing_key_cols)}/{len(key_cols)}")
if _missing_key_cols:
    print(f"    ⚠ MISSING    : {_missing_key_cols}")
else:
    print("    ✓ All key columns present")
for col in ['market_probability', 'model_probability', 'ranking_score', 'confidence_score']:
    print(f"      {col}: {analyzed_events[col].dtype}")

# ─── [2] top10 ───────────────────────────────────────────────
print("\n[2] top10 (DataFrame)")
print(f"    Shape: {top10.shape}  |  Has 'rank': {'rank' in top10.columns}")

# ─── [3] backtest_results ────────────────────────────────────
print("\n[3] backtest_results (dict)")
print(f"    Keys: {list(backtest_results.keys())}")
sim_m = backtest_results.get('metrics_simulated', {})
n_deciles = len(sim_m.get('calibration_deciles', []))
print(f"    brier_model  : {sim_m.get('brier_model')}  |  brier_market: {sim_m.get('brier_market')}")
print(f"    improvement  : {sim_m.get('brier_improvement_pct')}%  |  accuracy@60%: {sim_m.get('accuracy_thresh_60'):.4f}")
print(f"    calibration_deciles: {n_deciles} deciles")

# ─── [4] calibration_summary ─────────────────────────────────
print("\n[4] calibration_summary (dict)")
print(f"    Keys ({len(calibration_summary)}): {list(calibration_summary.keys())}")
print(f"    primary_model : {calibration_summary['primary_model_selected']}  |  alpha_blend: {calibration_summary['alpha_blend']}")
for mk in ['logistic_regression', 'gradient_boosting', 'blended_model', 'market_baseline']:
    m = calibration_summary.get(mk, {})
    if isinstance(m, dict) and 'brier_score' in m:
        print(f"    {mk:<30} brier={m['brier_score']:.4f}  ece={m.get('ece', 0):.4f}")

# ─── [5] model_insights ──────────────────────────────────────
print("\n[5] model_insights (dict)")
print(f"    Keys ({len(model_insights)}): {list(model_insights.keys())}")
fd = model_insights.get('feature_drivers', {})
tc = fd.get('top_combined_features', [])
print(f"    feature_drivers.top_combined_features: {len(tc)} items")
print(f"    Top 3: {[f['feature'] for f in tc[:3]]}")
ag = model_insights.get('aggregate_stats', {})
print(f"    aggregate_stats: mean_gap={ag.get('mean_absolute_gap'):.4f}  max_gap={ag.get('max_absolute_gap'):.4f}  mean_conf={ag.get('mean_confidence'):.4f}")
print(f"    mispricing_distribution: {model_insights.get('mispricing_distribution')}")
print(f"    signal_direction_distribution: {model_insights.get('signal_direction_distribution')}")

# ─── [6] top_features ────────────────────────────────────────
print(f"\n[6] top_features (DataFrame)")
print(f"    Shape: {top_features.shape}  |  Columns: {list(top_features.columns)}")
print(top_features[['feature', 'hr_label', 'importance_score']].head())

# ─── [7] FEAT_LABELS (from evaluation_artifacts) ─────────────
print(f"\n[7] FEAT_LABELS (dict) — {len(FEAT_LABELS)} entries")
print(f"    sentiment_polarity → '{FEAT_LABELS.get('sentiment_polarity')}'")
print(f"    combined_signal_score → '{FEAT_LABELS.get('combined_signal_score')}'")

# ─── [8] enriched_events ─────────────────────────────────────
print(f"\n[8] enriched_events (DataFrame)  shape: {enriched_events.shape}")
print(f"    Columns[:8]: {list(enriched_events.columns[:8])}")

# ─── [9] validation_summary ──────────────────────────────────
print(f"\n[9] validation_summary (str) — {len(validation_summary)} chars")

# ─── [10] Charts ─────────────────────────────────────────────
print("\n[10] Charts (matplotlib Figures)")
_figs = {
    'fig_ingestion'    : fig_ingestion,
    'fig_eval'         : fig_eval,
    'fig_gap_dist'     : fig_gap_dist,
    'fig_conf_scatter' : fig_conf_scatter,
    'fig_top_signals'  : fig_top_signals,
    'fig_category'     : fig_category,
}
for _n, _f in _figs.items():
    print(f"    {_n}: {type(_f).__name__}")

# ─── FINAL AUDIT CHECKS ──────────────────────────────────────
print("\n" + "=" * 65)
print("  AUDIT SUMMARY")
print("=" * 65)

_checks = {
    "analyzed_events (30×44 DataFrame)"        : analyzed_events.shape == (30, 44),
    "top10 (10 rows, has 'rank' col)"          : top10.shape[0] == 10 and 'rank' in top10.columns,
    "backtest_results (6 keys, 10 deciles)"    : len(backtest_results) == 6 and n_deciles == 10,
    "calibration_summary (10 keys)"            : len(calibration_summary) == 10,
    "model_insights (7 keys)"                  : len(model_insights) == 7,
    "feature_drivers in model_insights"        : 'feature_drivers' in model_insights,
    "aggregate_stats in model_insights"        : 'aggregate_stats' in model_insights,
    "top_features (12×6 DataFrame)"            : top_features.shape == (12, 6),
    "FEAT_LABELS (27 keys)"                    : len(FEAT_LABELS) == 27,
    "enriched_events (30×36 DataFrame)"        : enriched_events.shape == (30, 36),
    "validation_summary (non-empty str)"       : isinstance(validation_summary, str) and len(validation_summary) > 100,
    "6 charts available"                        : all(f is not None for f in _figs.values()),
    "All key cols present in analyzed_events"  : len(_missing_key_cols) == 0,
}

_all_pass = True
for _chk, _res in _checks.items():
    _sym = "✓" if _res else "✗"
    if not _res:
        _all_pass = False
    print(f"  {_sym}  {_chk}")

print()
if _all_pass:
    print("  ✅ ALL CHECKS PASSED — variables ready for Streamlit dashboard.")
else:
    print("  ⚠  SOME CHECKS FAILED — review above for details.")
print("=" * 65)

# ─── VARIABLE MAP (for deployment script reference) ──────────
dashboard_variable_map = {
    "analyzed_events"     : {"type": "DataFrame", "shape": str(analyzed_events.shape),  "block": "market_signal_auditor",  "description": "30 events — model probs, gaps, confidence, explanations, signal_direction"},
    "top10"               : {"type": "DataFrame", "shape": str(top10.shape),             "block": "market_signal_auditor",  "description": "Top 10 ranked mispricing opportunities (with 'rank' col)"},
    "backtest_results"    : {"type": "dict",      "shape": "6 keys",                     "block": "market_signal_auditor",  "description": "Simulated backtest (n=200): Brier, accuracy@60/70/80%, calibration deciles"},
    "calibration_summary" : {"type": "dict",      "shape": "10 keys",                    "block": "market_signal_auditor",  "description": "LR/GBM/blended/market Brier+ECE, calibration curves, primary model"},
    "model_insights"      : {"type": "dict",      "shape": "7 keys",                     "block": "market_signal_auditor",  "description": "feature_drivers, aggregate_stats, mispricing_distribution, top opportunities"},
    "top_features"        : {"type": "DataFrame", "shape": str(top_features.shape),      "block": "evaluation_artifacts",   "description": "Top 12 features with hr_label, importance_score, rank_lr/gbm"},
    "FEAT_LABELS"         : {"type": "dict",      "shape": "27 keys",                    "block": "evaluation_artifacts",   "description": "Feature name → human-readable label mapping (use evaluation_artifacts)"},
    "enriched_events"     : {"type": "DataFrame", "shape": str(enriched_events.shape),   "block": "market_signal_auditor",  "description": "30×36 feature-engineered events (pre-model)"},
    "validation_summary"  : {"type": "str",       "shape": f"{len(validation_summary)} chars", "block": "market_signal_auditor", "description": "Formatted backtest validation report"},
    "fig_ingestion"       : {"type": "Figure",    "block": "market_signal_auditor",      "description": "Category distribution / data ingestion chart"},
    "fig_eval"            : {"type": "Figure",    "block": "evaluation_artifacts",       "description": "2×2 eval: calibration, feature importance, mispricing, confidence"},
    "fig_gap_dist"        : {"type": "Figure",    "block": "market_signal_auditor",      "description": "Gap distribution histogram (underpriced vs overpriced)"},
    "fig_conf_scatter"    : {"type": "Figure",    "block": "market_signal_auditor",      "description": "Confidence vs absolute gap scatter"},
    "fig_top_signals"     : {"type": "Figure",    "block": "market_signal_auditor",      "description": "Top 10 mispricing opportunities horizontal bar chart"},
    "fig_category"        : {"type": "Figure",    "block": "market_signal_auditor",      "description": "Signal direction count (under/over) by category"},
}

print(f"\n  dashboard_variable_map: {len(dashboard_variable_map)} entries")
print("  zerve.variable() import reference:")
for _var, _meta in dashboard_variable_map.items():
    _blk = _meta['block']
    print(f"    {_var:<22} = variable('{_blk}', '{_var}')")
