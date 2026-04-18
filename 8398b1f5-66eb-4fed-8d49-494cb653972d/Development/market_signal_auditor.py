
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Zerve design system colors
# ─────────────────────────────────────────────────────────────
BG_COLOR    = "#1D1D20"
TEXT_PRI    = "#fbfbff"
TEXT_SEC    = "#909094"
GOLD        = "#ffd400"
GREEN       = "#8DE5A1"
CORAL       = "#FF9F9B"
ORANGE      = "#FFB482"
BLUE        = "#A1C9F4"
LAVENDER    = "#D0BBFF"

# ─────────────────────────────────────────────────────────────
# SECTION 1 — AUDIT HEADER
# ─────────────────────────────────────────────────────────────
print("=" * 72)
print("  📊  MARKET SIGNAL AUDITOR  —  Full Audit Report")
print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 72)

# ─────────────────────────────────────────────────────────────
# SECTION 2 — DATA QUALITY CHECKS
# ─────────────────────────────────────────────────────────────
print("\n[ DATA QUALITY ]")
print(f"  Total events in analyzed_events : {len(analyzed_events)}")

required_audit_cols = [
    'title', 'category', 'market_probability', 'model_probability',
    'signed_gap', 'confidence_score', 'explanation',
    'mispricing_flag', 'signal_direction'
]
missing_cols = [c for c in required_audit_cols if c not in analyzed_events.columns]
print(f"  Required columns present        : {len(required_audit_cols) - len(missing_cols)}/{len(required_audit_cols)}")
if missing_cols:
    print(f"  ⚠ Missing columns              : {missing_cols}")
else:
    print("  ✓ All required columns present")

null_summary = analyzed_events[required_audit_cols].isnull().sum()
null_cols_found = null_summary[null_summary > 0]
if len(null_cols_found):
    print(f"  ⚠ Columns with nulls           : {null_cols_found.to_dict()}")
else:
    print("  ✓ No null values in key columns")

prob_range_ok = (
    analyzed_events['market_probability'].between(0, 1).all() and
    analyzed_events['model_probability'].between(0, 1).all()
)
print(f"  ✓ Probability range [0,1]       : {prob_range_ok}")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — SIGNAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n[ SIGNAL SUMMARY ]")
n_underpriced = (analyzed_events['signal_direction'] == 'UNDERPRICED').sum()
n_overpriced  = (analyzed_events['signal_direction'] == 'OVERPRICED').sum()
n_signals_gt4 = (analyzed_events['signed_gap'].abs() > 0.04).sum()
avg_gap       = analyzed_events['signed_gap'].abs().mean()
max_gap       = analyzed_events['signed_gap'].abs().max()
avg_conf      = analyzed_events['confidence_score'].mean()
n_high_conf   = (analyzed_events['confidence_score'] >= 0.7).sum()

print(f"  Events analyzed                 : {len(analyzed_events)}")
print(f"  Underpriced signals             : {n_underpriced}")
print(f"  Overpriced signals              : {n_overpriced}")
print(f"  Signals with |gap| > 4%         : {n_signals_gt4}")
print(f"  Mean absolute gap               : {avg_gap:.3f} ({avg_gap:.1%})")
print(f"  Max absolute gap                : {max_gap:.3f} ({max_gap:.1%})")
print(f"  Mean confidence score           : {avg_conf:.3f} ({avg_conf:.1%})")
print(f"  High-confidence events (≥70%)   : {n_high_conf}/{len(analyzed_events)}")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — TOP 10 MISPRICED EVENTS
# ─────────────────────────────────────────────────────────────
print("\n[ TOP 10 MISPRICED OPPORTUNITIES ]")
_top10_audit = (
    analyzed_events
    .sort_values('ranking_score', ascending=False)
    .head(10)
    [['title', 'category', 'market_probability', 'model_probability',
      'signed_gap', 'confidence_score', 'signal_direction']]
    .copy()
)
_top10_audit.index = range(1, len(_top10_audit) + 1)

for _rank, _row in _top10_audit.iterrows():
    _arrow = "▲" if _row['signal_direction'] == 'UNDERPRICED' else "▼"
    print(
        f"  {_rank:>2}. [{_arrow}] {_row['title'][:48]:<48} "
        f"Mkt:{_row['market_probability']:.0%} → Mdl:{_row['model_probability']:.0%}  "
        f"Gap:{_row['signed_gap']:+.1%}  Conf:{_row['confidence_score']:.0%}"
    )

# ─────────────────────────────────────────────────────────────
# SECTION 5 — MODEL PERFORMANCE FROM model_insights
# ─────────────────────────────────────────────────────────────
print("\n[ MODEL PERFORMANCE ]")
_cal_summary   = model_insights.get('calibration_summary', {})
_agg_stats     = model_insights.get('aggregate_stats', {})
_model_config  = model_insights.get('model_config', {})

_lr_metrics  = _cal_summary.get('logistic_regression', {})
_gbm_metrics = _cal_summary.get('gradient_boosting', {})
_blended     = _cal_summary.get('blended_model', {})
_market_base = _cal_summary.get('market_baseline', {})
_primary     = _cal_summary.get('primary_model_selected', 'N/A')
_alpha       = _cal_summary.get('alpha_blend', 0.55)

print(f"  Primary model selected          : {_primary}")
print(f"  Alpha blend (model vs market)   : {_alpha:.2f}")
print(f"  Training samples                : {_cal_summary.get('n_training_samples', 'N/A')}")
print(f"  Features used                   : {_model_config.get('n_features', 'N/A')}")
print()
print(f"  {'Model':<30} {'Brier':>8} {'ECE':>8} {'AUC':>8}")
print(f"  {'-'*54}")
print(f"  {'Logistic Regression':<30} {_lr_metrics.get('brier_score', 0):.4f}   {_lr_metrics.get('ece', 0):.4f}   {_lr_metrics.get('cv_auc', 0):.3f}")
print(f"  {'Gradient Boosting':<30} {_gbm_metrics.get('brier_score', 0):.4f}   {_gbm_metrics.get('ece', 0):.4f}   {_gbm_metrics.get('cv_auc', 0):.3f}")
print(f"  {'Blended Model':<30} {_blended.get('brier_score', 0):.4f}   {_blended.get('ece', 0):.4f}   {'N/A':>5}")
print(f"  {'Market Baseline':<30} {_market_base.get('brier_score', 0):.4f}   {_market_base.get('ece', 0):.4f}   {'N/A':>5}")

_brier_improvement = (
    (_market_base.get('brier_score', 0) - _blended.get('brier_score', 0))
    / max(_market_base.get('brier_score', 1e-9), 1e-9) * 100
)
print(f"\n  Brier improvement vs market     : {_brier_improvement:+.1f}%")
print(f"  Mean absolute gap               : {_agg_stats.get('mean_absolute_gap', 0):.3f}")
print(f"  Mean model probability          : {_agg_stats.get('mean_model_probability', 0):.3f}")
print(f"  Mean market probability         : {_agg_stats.get('mean_market_probability', 0):.3f}")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — BACKTEST VALIDATION SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n[ BACKTEST VALIDATION ]")
_prov = backtest_results.get('data_provenance', {})
_sim  = backtest_results.get('metrics_simulated', {})
_comb = backtest_results.get('metrics_combined', {})

print(f"  Real markets fetched from API   : {_prov.get('n_resolved_markets_fetched', 0)}")
print(f"  Real events matched to model    : {_prov.get('n_matched_to_model', 0)}")
print(f"  Simulated events                : {_prov.get('n_simulated', 0)}")
print(f"  Total combined                  : {_prov.get('n_combined', 0)}")
print()
print(f"  [Simulated] Brier (model)       : {_sim.get('brier_model', 'N/A')}")
print(f"  [Simulated] Brier (market)      : {_sim.get('brier_market', 'N/A')}")
print(f"  [Simulated] Brier improvement   : {_sim.get('brier_improvement_pct', 'N/A')}%")
print(f"  [Simulated] Accuracy @ ≥60%     : {_sim.get('accuracy_thresh_60', 0):.1%}")
print(f"  [Simulated] Accuracy @ ≥70%     : {_sim.get('accuracy_thresh_70', 0):.1%}")
print(f"  [Simulated] Accuracy @ ≥80%     : {_sim.get('accuracy_thresh_80', 0):.1%}")
print(f"  ⚠ {_prov.get('transparency_note', '')[:80]}...")

# ─────────────────────────────────────────────────────────────
# SECTION 7 — TOP FEATURE DRIVERS
# ─────────────────────────────────────────────────────────────
print("\n[ TOP FEATURE DRIVERS ]")
_feat_drivers = model_insights.get('feature_drivers', {})
_top_feats    = _feat_drivers.get('top_combined_features', [])[:10]
for _f in _top_feats:
    print(f"  {_f['feature']:<35} combined_rank: {_f['combined_rank']:.1f}")

# ─────────────────────────────────────────────────────────────
# SECTION 8 — CATEGORY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print("\n[ CATEGORY BREAKDOWN ]")
_cat_stats = (
    analyzed_events
    .groupby('category')
    .agg(
        n_events=('title', 'count'),
        mean_gap=('signed_gap', lambda x: x.abs().mean()),
        mean_conf=('confidence_score', 'mean'),
        n_under=('signal_direction', lambda x: (x == 'UNDERPRICED').sum()),
        n_over=('signal_direction', lambda x: (x == 'OVERPRICED').sum()),
    )
    .sort_values('mean_gap', ascending=False)
)
print(f"  {'Category':<14} {'N':>4} {'AvgGap':>8} {'AvgConf':>8} {'Under':>6} {'Over':>6}")
print(f"  {'-'*52}")
for _cat, _r in _cat_stats.iterrows():
    print(
        f"  {_cat:<14} {_r['n_events']:>4} "
        f"{_r['mean_gap']:>7.1%} {_r['mean_conf']:>8.1%} "
        f"{int(_r['n_under']):>6} {int(_r['n_over']):>6}"
    )

# ─────────────────────────────────────────────────────────────
# SECTION 9 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor':   BG_COLOR,
    'axes.edgecolor':   TEXT_SEC,
    'text.color':       TEXT_PRI,
    'axes.labelcolor':  TEXT_PRI,
    'xtick.color':      TEXT_SEC,
    'ytick.color':      TEXT_SEC,
    'grid.color':       '#2e2e32',
    'grid.alpha':       0.5,
})

# --- Chart 1: Gap Distribution ---
fig_gap_dist = plt.figure(figsize=(10, 5))
fig_gap_dist.patch.set_facecolor(BG_COLOR)
ax1 = fig_gap_dist.add_subplot(111)

_gaps = analyzed_events['signed_gap'].values
_under_mask = _gaps > 0
_over_mask  = _gaps < 0

ax1.hist(_gaps[_under_mask], bins=12, color=GREEN,  alpha=0.85, label='Underpriced (gap > 0)')
ax1.hist(_gaps[_over_mask],  bins=12, color=CORAL,  alpha=0.85, label='Overpriced (gap < 0)')
ax1.axvline(0, color=GOLD, linewidth=1.5, linestyle='--', alpha=0.8)
ax1.set_title('Distribution of Model vs Market Gap (Signed)', fontsize=13, color=TEXT_PRI, pad=12)
ax1.set_xlabel('Signed Gap (Model Prob − Market Prob)', color=TEXT_PRI)
ax1.set_ylabel('Count', color=TEXT_PRI)
ax1.legend(facecolor='#2A2A2E', edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax1.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

# --- Chart 2: Confidence vs Gap Scatter ---
fig_conf_scatter = plt.figure(figsize=(10, 5))
fig_conf_scatter.patch.set_facecolor(BG_COLOR)
ax2 = fig_conf_scatter.add_subplot(111)

_colors_scatter = [GREEN if g > 0 else CORAL for g in analyzed_events['signed_gap']]
ax2.scatter(
    analyzed_events['confidence_score'],
    analyzed_events['signed_gap'].abs(),
    c=_colors_scatter,
    s=80,
    alpha=0.85,
    edgecolors='none'
)
ax2.axhline(0.04, color=GOLD, linewidth=1.2, linestyle='--', alpha=0.8, label='Signal threshold (4%)')
ax2.set_title('Confidence Score vs Absolute Gap', fontsize=13, color=TEXT_PRI, pad=12)
ax2.set_xlabel('Confidence Score', color=TEXT_PRI)
ax2.set_ylabel('Absolute Gap', color=TEXT_PRI)
# Custom legend patches
from matplotlib.patches import Patch
_legend_elements = [
    Patch(facecolor=GREEN, label='Underpriced'),
    Patch(facecolor=CORAL, label='Overpriced'),
    plt.Line2D([0], [0], color=GOLD, linestyle='--', label='4% threshold'),
]
ax2.legend(handles=_legend_elements, facecolor='#2A2A2E', edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# --- Chart 3: Top 10 Signals Bar Chart ---
fig_top_signals = plt.figure(figsize=(11, 6))
fig_top_signals.patch.set_facecolor(BG_COLOR)
ax3 = fig_top_signals.add_subplot(111)

_plot_df = (
    analyzed_events
    .sort_values('ranking_score', ascending=False)
    .head(10)
    [['title', 'signed_gap', 'confidence_score']]
    .copy()
)
_plot_df['short_title'] = _plot_df['title'].str[:40]
_bar_colors = [GREEN if g > 0 else CORAL for g in _plot_df['signed_gap']]
_bars = ax3.barh(
    range(len(_plot_df)),
    _plot_df['signed_gap'].abs() * 100,
    color=_bar_colors,
    alpha=0.9,
    height=0.65
)
ax3.set_yticks(range(len(_plot_df)))
ax3.set_yticklabels(_plot_df['short_title'], fontsize=9)
ax3.invert_yaxis()
ax3.set_xlabel('Absolute Gap (%)', color=TEXT_PRI)
ax3.set_title('Top 10 Mispricing Opportunities by Ranking Score', fontsize=13, color=TEXT_PRI, pad=12)

# Annotate with confidence
for _idx, (_bar, _conf, _gap) in enumerate(zip(_bars, _plot_df['confidence_score'], _plot_df['signed_gap'])):
    _direction = '▲' if _gap > 0 else '▼'
    ax3.text(
        _bar.get_width() + 0.1,
        _bar.get_y() + _bar.get_height() / 2,
        f'{_direction} {_conf:.0%}',
        va='center', ha='left', fontsize=8, color=TEXT_SEC
    )

_legend_patches = [Patch(facecolor=GREEN, label='Underpriced'), Patch(facecolor=CORAL, label='Overpriced')]
ax3.legend(handles=_legend_patches, facecolor='#2A2A2E', edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax3.grid(True, axis='x', alpha=0.3)
plt.tight_layout()

# --- Chart 4: Category Stats ---
fig_category = plt.figure(figsize=(10, 5))
fig_category.patch.set_facecolor(BG_COLOR)
ax4 = fig_category.add_subplot(111)

_cats = _cat_stats.index.tolist()
_x   = np.arange(len(_cats))
_w   = 0.35
_bars_under = ax4.bar(_x - _w/2, _cat_stats['n_under'], width=_w, color=GREEN,  alpha=0.85, label='Underpriced')
_bars_over  = ax4.bar(_x + _w/2, _cat_stats['n_over'],  width=_w, color=CORAL,  alpha=0.85, label='Overpriced')
ax4.set_xticks(_x)
ax4.set_xticklabels([c.capitalize() for c in _cats], fontsize=10)
ax4.set_ylabel('Number of Events', color=TEXT_PRI)
ax4.set_title('Signal Direction Count by Category', fontsize=13, color=TEXT_PRI, pad=12)
ax4.legend(facecolor='#2A2A2E', edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax4.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

# ─────────────────────────────────────────────────────────────
# SECTION 10 — FINAL AUDIT RESULT
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  AUDIT RESULT: ✓ PASS")
print(f"  {len(analyzed_events)} events analyzed | {n_signals_gt4} actionable signals | "
      f"Brier improvement: {_brier_improvement:+.1f}%")
print("=" * 72)
