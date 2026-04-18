
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Zerve Design System ────────────────────────────────────────────────────────
BG       = "#1D1D20"
TEXT_PRI = "#fbfbff"
TEXT_SEC = "#909094"
BLUE     = "#A1C9F4"
ORANGE   = "#FFB482"
GREEN    = "#8DE5A1"
CORAL    = "#FF9F9B"
LAVENDER = "#D0BBFF"
GOLD     = "#ffd400"
SUCCESS  = "#17b26a"
WARNING  = "#f04438"
PURPLE   = "#9467BD"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT_PRI, "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_PRI, "ytick.color": TEXT_PRI,
    "axes.edgecolor": TEXT_SEC, "grid.color": "#333336",
    "font.family": "sans-serif",
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

# ── Human-readable feature name mapping ──────────────────────────────────────
FEAT_LABELS = {
    "combined_signal_score":    "Composite Signal",
    "sentiment_polarity":       "Sentiment",
    "liquidity_ratio":          "Market Liquidity",
    "prob_vs_base_rate":        "Prob vs Base Rate",
    "market_probability":       "Market Probability",
    "probability_entropy":      "Decision Entropy",
    "category_base_rate":       "Category Base Rate",
    "log_volume":               "Trading Volume (log)",
    "log_liquidity":            "Liquidity Depth (log)",
    "volume_zscore":            "Volume Z-Score",
    "liquidity_zscore":         "Liquidity Z-Score",
    "market_attention_index":   "Market Attention",
    "kw_net_signal":            "Net Keyword Signal",
    "kw_bullish_count":         "Bullish Keywords",
    "kw_bearish_count":         "Bearish Keywords",
    "kw_total_signals":         "Total Keywords",
    "kw_uncertainty_count":     "Uncertainty Keywords",
    "calibration_score":        "Calibration Quality",
    "time_horizon_encoded":     "Time Horizon",
    "resolution_urgency":       "Resolution Urgency",
    "sentiment_subjectivity":   "Text Subjectivity",
    "category_encoded":         "Category (ordinal)",
    "cat_politics":             "Is: Politics",
    "cat_sports":               "Is: Sports",
    "cat_crypto":               "Is: Crypto",
    "cat_economics":            "Is: Economics",
    "cat_science":              "Is: Science",
}

# ── Signal family mapping ─────────────────────────────────────────────────────
_SENTIMENT_FEATS = {"sentiment_polarity", "sentiment_subjectivity"}
_MARKET_FEATS    = {
    "market_probability", "prob_vs_base_rate", "calibration_score",
    "category_base_rate", "combined_signal_score",
    "liquidity_ratio", "log_volume", "log_liquidity",
    "volume_zscore", "liquidity_zscore", "market_attention_index",
    "kw_bullish_count", "kw_bearish_count", "kw_net_signal",
    "kw_uncertainty_count", "kw_total_signals", "probability_entropy"
}
_TEMPORAL_FEATS  = {"time_horizon_encoded", "resolution_urgency"}
_CATEGORY_FEATS  = {"category_encoded", "cat_politics", "cat_sports",
                    "cat_crypto", "cat_economics", "cat_science"}

def _family_color(feat):
    if feat in _SENTIMENT_FEATS: return BLUE
    if feat in _TEMPORAL_FEATS:  return ORANGE
    if feat in _CATEGORY_FEATS:  return PURPLE
    return GREEN

# ─────────────────────────────────────────────────────────────────────────────
# BUILD 2x2 FIGURE GRID
# ─────────────────────────────────────────────────────────────────────────────
fig_eval, ((ax_cal, ax_feat), (ax_misprice, ax_conf)) = plt.subplots(
    2, 2, figsize=(16, 12), dpi=110)
fig_eval.patch.set_facecolor(BG)
fig_eval.suptitle("[ MARKET SIGNAL AUDITOR ] — Model Evaluation",
                  fontsize=15, fontweight="bold", color=GOLD, y=1.01)

# ══ [0,0] CALIBRATION CURVES ═══════════════════════════════════════════════════
ax_cal.set_facecolor(BG)
ax_cal.plot([0, 1], [0, 1], "--", color=TEXT_SEC, lw=1.5, label="Perfect Calibration", zorder=1)

_m_pred  = model_insights["calibration_summary"]["calibration_curve_model"]["mean_predicted"]
_m_frac  = model_insights["calibration_summary"]["calibration_curve_model"]["fraction_positives"]
_lr_pred = model_insights["calibration_summary"]["calibration_curve_lr"]["mean_predicted"]
_lr_frac = model_insights["calibration_summary"]["calibration_curve_lr"]["fraction_positives"]
_gb_pred = model_insights["calibration_summary"]["calibration_curve_gbm"]["mean_predicted"]
_gb_frac = model_insights["calibration_summary"]["calibration_curve_gbm"]["fraction_positives"]

ax_cal.plot(_m_pred,  _m_frac,  "o-",  color=BLUE,   lw=2.2, ms=7, label=f"Blended  ECE={ece_model:.3f}", zorder=4)
ax_cal.plot(_lr_pred, _lr_frac, "s--", color=ORANGE,  lw=1.6, ms=5, label=f"LR       ECE={ece_lr:.3f}", zorder=3)
ax_cal.plot(_gb_pred, _gb_frac, "^--", color=GREEN,   lw=1.6, ms=5, label=f"GBM      ECE={ece_gbm:.3f}", zorder=3)
ax_cal.fill_between(_m_pred, _m_pred, _m_frac, color=BLUE, alpha=0.07)

ax_cal.set_xlim(-0.02, 1.02); ax_cal.set_ylim(-0.02, 1.15)
ax_cal.set_xlabel("Mean Predicted Probability"); ax_cal.set_ylabel("Fraction of Positives")
ax_cal.set_title("Calibration Curves", fontweight="bold", color=TEXT_PRI)
ax_cal.legend(loc="lower right", fontsize=8.5, facecolor="#2a2a2d",
              edgecolor=TEXT_SEC, labelcolor=TEXT_PRI, framealpha=0.88)
ax_cal.grid(True, alpha=0.15, lw=0.7)

# ══ [0,1] FEATURE IMPORTANCE — TOP 12, human-readable labels ══════════════════
top_features = combined_ranks.sort_values("combined_rank").head(12).copy()
top_features["hr_label"] = top_features["feature"].map(lambda f: FEAT_LABELS.get(f, f.replace("_", " ").title()))
_max_rank = top_features["combined_rank"].max()
top_features["importance_score"] = (_max_rank + 1) - top_features["combined_rank"]
top_features = top_features.sort_values("importance_score", ascending=True)

_bar_colors = [_family_color(f) for f in top_features["feature"]]

ax_feat.set_facecolor(BG)
_bars_feat = ax_feat.barh(top_features["hr_label"], top_features["importance_score"],
                          color=_bar_colors, edgecolor="none", height=0.65)
for _b, _v in zip(_bars_feat, top_features["importance_score"]):
    ax_feat.text(_b.get_width() + 0.05, _b.get_y() + _b.get_height() / 2,
                 f"{_v:.1f}", va="center", ha="left", fontsize=8, color=TEXT_PRI)

ax_feat.set_xlabel("Relative Importance (higher = more predictive)")
ax_feat.set_title("Top 12 Feature Drivers\n(colour = signal family)", fontweight="bold", color=TEXT_PRI)
ax_feat.set_xlim(0, top_features["importance_score"].max() * 1.2)
ax_feat.grid(axis="x", alpha=0.12, lw=0.7)
ax_feat.spines[["top", "right"]].set_visible(False)

_legend_handles = [
    mpatches.Patch(color=BLUE,   label="Sentiment"),
    mpatches.Patch(color=GREEN,  label="Market / Signal"),
    mpatches.Patch(color=ORANGE, label="Temporal"),
    mpatches.Patch(color=PURPLE, label="Category"),
]
ax_feat.legend(handles=_legend_handles, loc="lower right", fontsize=8,
               facecolor="#2a2a2d", edgecolor=TEXT_SEC, labelcolor=TEXT_PRI, framealpha=0.88)

# ══ [1,0] MISPRICING DISTRIBUTION ═════════════════════════════════════════════
ax_misprice.set_facecolor(BG)
gaps = analyzed_events["signed_gap"].values * 100
underprice_gaps = [g for g in gaps if g > 0]
overprice_gaps  = [g for g in gaps if g < 0]
fair_gaps       = [g for g in gaps if g == 0]
bins = np.linspace(gaps.min() - 0.5, gaps.max() + 0.5, 22)

ax_misprice.hist(underprice_gaps, bins=bins, color=GREEN, alpha=0.82,
                 label=f"Underpriced (n={len(underprice_gaps)})", edgecolor=BG, lw=0.4)
ax_misprice.hist(overprice_gaps,  bins=bins, color=CORAL, alpha=0.82,
                 label=f"Overpriced  (n={len(overprice_gaps)})",  edgecolor=BG, lw=0.4)
if fair_gaps:
    ax_misprice.hist(fair_gaps, bins=bins, color=GOLD, alpha=0.90,
                     label=f"Fair-value (n={len(fair_gaps)})", edgecolor=BG, lw=0.4)

ax_misprice.axvline(0, color=TEXT_PRI, lw=2.0, ls="-", alpha=0.6, zorder=5)
mean_gap = float(np.mean(gaps))
ax_misprice.axvline(mean_gap, color=GOLD, lw=1.8, ls=":", alpha=0.88, zorder=6)

_y_top = ax_misprice.get_ylim()[1] if ax_misprice.get_ylim()[1] > 0 else 5
ax_misprice.text(0.4, _y_top * 0.96, "Market\nprice", color=TEXT_SEC, fontsize=8, va="top")
ax_misprice.text(mean_gap + 0.25, _y_top * 0.76, f"Mean\n{mean_gap:+.2f}pp", color=GOLD, fontsize=8, va="top")

n_under = len(underprice_gaps)
n_over  = len(overprice_gaps)
n_fair  = len(fair_gaps)
stats_txt = (f"Mean |gap|: {float(np.mean(np.abs(gaps))):.2f}pp\n"
             f"Std dev:   {float(np.std(gaps)):.2f}pp")
ax_misprice.text(0.98, 0.97, stats_txt, transform=ax_misprice.transAxes, fontsize=8.5,
                 color=TEXT_PRI, va="top", ha="right", family="monospace",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a2d", edgecolor=TEXT_SEC, alpha=0.88))

ax_misprice.set_xlabel("Mispricing Gap (Model − Market, percentage points)")
ax_misprice.set_ylabel("Number of Events")
ax_misprice.set_title("Mispricing Distribution — 50 Events", fontweight="bold", color=TEXT_PRI)
ax_misprice.legend(fontsize=9, facecolor="#2a2a2d", edgecolor=TEXT_SEC, labelcolor=TEXT_PRI, framealpha=0.88)
ax_misprice.grid(axis="y", alpha=0.12, lw=0.7)
ax_misprice.spines[["top", "right"]].set_visible(False)

# ══ [1,1] CONFIDENCE DISTRIBUTION ════════════════════════════════════════════
ax_conf.set_facecolor(BG)
conf_vals = analyzed_events["confidence_score"].values * 100

ax_conf.hist(conf_vals, bins=20, color=BLUE, alpha=0.82, edgecolor=BG, lw=0.4)

# Threshold lines: High / Med / Low
_HIGH_T = 70; _MED_T = 45
ax_conf.axvline(_HIGH_T, color=GREEN, lw=1.8, ls="--", alpha=0.9, label=f"High ≥ {_HIGH_T}%")
ax_conf.axvline(_MED_T,  color=ORANGE, lw=1.8, ls="--", alpha=0.9, label=f"Med ≥ {_MED_T}%")

_y2_top = ax_conf.get_ylim()[1] if ax_conf.get_ylim()[1] > 0 else 10
ax_conf.text(_HIGH_T + 1, _y2_top * 0.92, "High", color=GREEN,  fontsize=9, fontweight="bold")
ax_conf.text(_MED_T  + 1, _y2_top * 0.92, "Med",  color=ORANGE, fontsize=9, fontweight="bold")
ax_conf.text(1, _y2_top * 0.92, "Low", color=CORAL,  fontsize=9, fontweight="bold")

_n_high = (conf_vals >= _HIGH_T).sum()
_n_med  = ((conf_vals >= _MED_T) & (conf_vals < _HIGH_T)).sum()
_n_low  = (conf_vals < _MED_T).sum()
_conf_txt = f"High: {_n_high}  |  Med: {_n_med}  |  Low: {_n_low}"
ax_conf.text(0.50, 0.95, _conf_txt, transform=ax_conf.transAxes, fontsize=9,
             color=TEXT_PRI, va="top", ha="center", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2a2d", edgecolor=TEXT_SEC, alpha=0.88))

ax_conf.set_xlabel("Confidence Score (%)")
ax_conf.set_ylabel("Number of Events")
ax_conf.set_title("Confidence Distribution", fontweight="bold", color=TEXT_PRI)
ax_conf.legend(fontsize=9, facecolor="#2a2a2d", edgecolor=TEXT_SEC, labelcolor=TEXT_PRI, framealpha=0.88)
ax_conf.grid(axis="y", alpha=0.12, lw=0.7)
ax_conf.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()

# ════════════════════════════════════════════════════════
# EVALUATION SUMMARY — ONE CLEAN LINE
# ════════════════════════════════════════════════════════
_pct_high_conf = (_n_high / len(conf_vals)) * 100
print("╔══════════════════════════════════════════════════════════╗")
print("║   [ MARKET SIGNAL AUDITOR ] — Model Evaluation          ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
print(
    f"  Brier Score: {brier_model:.4f}  |  ECE: {ece_model:.4f}  |  "
    f"AUC: {max(lr_auc, gbm_auc):.3f}  |  High-Confidence Events: {_pct_high_conf:.0f}%"
)
print()
print(f"  ✓ Calibration curves plotted — Blended vs Perfect diagonal")
print(f"  ✓ Feature importance — top 12 drivers, colour-coded by signal family")
print(f"  ✓ Mispricing distribution — {n_under} underpriced / {n_over} overpriced / {n_fair} fair")
print(f"  ✓ Confidence tiers — {_n_high} High / {_n_med} Medium / {_n_low} Low confidence events")
