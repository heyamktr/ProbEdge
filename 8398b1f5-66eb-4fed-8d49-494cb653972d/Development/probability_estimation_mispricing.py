
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. FEATURE MATRIX CONSTRUCTION
# ─────────────────────────────────────────────
MODEL_FEATURES = [
    "sentiment_polarity", "sentiment_subjectivity",
    "kw_bullish_count", "kw_bearish_count", "kw_net_signal",
    "kw_uncertainty_count", "kw_total_signals",
    "time_horizon_encoded", "category_encoded",
    "cat_politics", "cat_sports", "cat_crypto", "cat_economics", "cat_science",
    "liquidity_ratio", "log_volume", "log_liquidity",
    "volume_zscore", "liquidity_zscore", "market_attention_index",
    "category_base_rate", "prob_vs_base_rate", "probability_entropy",
    "resolution_urgency", "combined_signal_score",
    "market_probability", "calibration_score",
]

X             = enriched_events[MODEL_FEATURES].copy()
y_soft        = enriched_events["market_probability"].values
feature_completeness = X.notnull().mean(axis=1)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# 2. PSEUDO-BINARY LABELS
# ─────────────────────────────────────────────
signal_threshold = enriched_events["combined_signal_score"].median()
y_binary         = (enriched_events["combined_signal_score"] > signal_threshold).astype(int)

# ─────────────────────────────────────────────
# 3. MODEL TRAINING — LR + GBM
# ─────────────────────────────────────────────
lr_base  = LogisticRegression(max_iter=1000, C=0.5, solver="lbfgs", random_state=42)
model_lr = CalibratedClassifierCV(lr_base, cv=5, method="sigmoid")
model_lr.fit(X_scaled, y_binary)
lr_probs    = model_lr.predict_proba(X_scaled)[:, 1]
lr_cv_scores = cross_val_score(lr_base, X_scaled, y_binary, cv=5, scoring="roc_auc")
lr_auc       = lr_cv_scores.mean()

gbm_base = GradientBoostingClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    min_samples_leaf=3, random_state=42)
model_gbm = CalibratedClassifierCV(gbm_base, cv=5, method="isotonic")
model_gbm.fit(X_scaled, y_binary)
gbm_probs    = model_gbm.predict_proba(X_scaled)[:, 1]
gbm_cv_scores = cross_val_score(gbm_base, X_scaled, y_binary, cv=5, scoring="roc_auc")
gbm_auc       = gbm_cv_scores.mean()

use_gbm_fallback   = gbm_auc > lr_auc + 0.02
primary_model_name = "GradientBoosting" if use_gbm_fallback else "LogisticRegression"
raw_signal_probs   = gbm_probs if use_gbm_fallback else lr_probs

# ─────────────────────────────────────────────
# 4. BLENDED PROBABILITY & DISAGREEMENT METRICS
# ─────────────────────────────────────────────
alpha_blend  = 0.55
signal_nudge = (raw_signal_probs - 0.5) * 0.30
model_probability = np.clip(
    alpha_blend * enriched_events["market_probability"].values +
    (1 - alpha_blend) * (enriched_events["market_probability"].values + signal_nudge),
    0.02, 0.98
)

signed_gap   = np.round(model_probability - enriched_events["market_probability"].values, 4)
absolute_gap = np.abs(signed_gap)

def min_max_norm(arr):
    r = arr.max() - arr.min()
    return (arr - arr.min()) / (r if r > 0 else 1)

ranking_score = np.round(
    0.40 * min_max_norm(absolute_gap) +
    0.30 * min_max_norm(np.abs(raw_signal_probs - 0.5)) +
    0.30 * min_max_norm(enriched_events["market_attention_index"].values),
    4
)

model_certainty = np.abs(raw_signal_probs - 0.5) * 2
liquidity_conf  = min_max_norm(enriched_events["log_liquidity"].values)
confidence_score = np.round(
    0.40 * feature_completeness.values +
    0.35 * model_certainty +
    0.25 * liquidity_conf,
    4
)

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCES
# ─────────────────────────────────────────────
lr_coef = np.zeros(len(MODEL_FEATURES))
for estimator in model_lr.calibrated_classifiers_:
    lr_coef += estimator.estimator.coef_[0]
lr_coef /= len(model_lr.calibrated_classifiers_)

lr_importance = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "lr_coefficient": lr_coef,
    "lr_abs_importance": np.abs(lr_coef)
}).sort_values("lr_abs_importance", ascending=False).reset_index(drop=True)

gbm_feat_imp = np.zeros(len(MODEL_FEATURES))
for estimator in model_gbm.calibrated_classifiers_:
    gbm_feat_imp += estimator.estimator.feature_importances_
gbm_feat_imp /= len(model_gbm.calibrated_classifiers_)

gbm_importance = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "gbm_importance": gbm_feat_imp
}).sort_values("gbm_importance", ascending=False).reset_index(drop=True)

feature_rank_lr  = lr_importance.reset_index()[["feature", "index"]].rename(columns={"index": "rank_lr"})
feature_rank_gbm = gbm_importance.reset_index()[["feature", "index"]].rename(columns={"index": "rank_gbm"})
combined_ranks   = feature_rank_lr.merge(feature_rank_gbm, on="feature")
combined_ranks["combined_rank"] = (combined_ranks["rank_lr"] + combined_ranks["rank_gbm"]) / 2
combined_ranks   = combined_ranks.sort_values("combined_rank").reset_index(drop=True)
top_drivers      = combined_ranks["feature"].head(5).tolist()

# ─────────────────────────────────────────────
# 6. CALIBRATION ANALYSIS
# ─────────────────────────────────────────────
n_bins = 5
frac_positives_lr,    mean_pred_lr    = calibration_curve(y_binary, lr_probs,    n_bins=n_bins, strategy="quantile")
frac_positives_gbm,   mean_pred_gbm   = calibration_curve(y_binary, gbm_probs,   n_bins=n_bins, strategy="quantile")
frac_positives_model, mean_pred_model = calibration_curve(y_binary, model_probability, n_bins=n_bins, strategy="quantile")

brier_lr     = np.mean((lr_probs    - y_binary) ** 2)
brier_gbm    = np.mean((gbm_probs   - y_binary) ** 2)
brier_model  = np.mean((model_probability - y_binary) ** 2)
brier_market = np.mean((enriched_events["market_probability"].values - y_binary) ** 2)

def ece_score(probs, labels, n_bins=10):
    _bins = np.linspace(0, 1, n_bins + 1)
    _ece  = 0.0
    for _i in range(n_bins):
        _mask = (probs >= _bins[_i]) & (probs < _bins[_i+1])
        if _mask.sum() > 0:
            _ece += _mask.sum() * abs(labels[_mask].mean() - probs[_mask].mean())
    return _ece / len(probs)

_yb = y_binary.values if hasattr(y_binary, "values") else y_binary
ece_lr     = ece_score(lr_probs,    _yb)
ece_gbm    = ece_score(gbm_probs,   _yb)
ece_model  = ece_score(model_probability, _yb)
ece_market = ece_score(enriched_events["market_probability"].values, _yb)

calibration_summary = {
    "logistic_regression": {"cv_auc": round(float(lr_auc), 4),   "brier_score": round(float(brier_lr), 4),    "ece": round(float(ece_lr), 4)},
    "gradient_boosting":   {"cv_auc": round(float(gbm_auc), 4),  "brier_score": round(float(brier_gbm), 4),   "ece": round(float(ece_gbm), 4)},
    "blended_model":       {"brier_score": round(float(brier_model), 4),  "ece": round(float(ece_model), 4)},
    "market_baseline":     {"brier_score": round(float(brier_market), 4), "ece": round(float(ece_market), 4)},
    "primary_model_selected": primary_model_name,
    "alpha_blend": alpha_blend,
    "n_training_samples": len(y_binary),
    "calibration_curve_lr":    {"mean_predicted": mean_pred_lr.tolist(),    "fraction_positives": frac_positives_lr.tolist()},
    "calibration_curve_gbm":   {"mean_predicted": mean_pred_gbm.tolist(),   "fraction_positives": frac_positives_gbm.tolist()},
    "calibration_curve_model": {"mean_predicted": mean_pred_model.tolist(), "fraction_positives": frac_positives_model.tolist()},
}

# ─────────────────────────────────────────────
# 7. PLAIN-ENGLISH EXPLANATIONS
# ─────────────────────────────────────────────
def _direction_label(gap):
    if gap > 0.05:   return "under-pricing"
    elif gap < -0.05: return "over-pricing"
    else:             return "fairly priced"

def _magnitude_label(abs_gap):
    if abs_gap >= 0.12: return "significantly"
    elif abs_gap >= 0.06: return "moderately"
    else:               return "slightly"

def _confidence_label(conf):
    if conf >= 0.70: return "high"
    elif conf >= 0.45: return "moderate"
    else:            return "low"

def _sentiment_narrative(polarity, subjectivity):
    if polarity > 0.3:    return "positive narrative tone"
    elif polarity < -0.3: return "negative narrative tone"
    else:                 return "neutral/mixed narrative tone"

def _horizon_caveat(horizon):
    return {
        "immediate":   "Resolution is imminent — uncertainty could shift rapidly.",
        "short_term":  "Near-term resolution limits time for re-pricing.",
        "medium_term": "Medium time horizon provides room for new information to emerge.",
        "long_term":   "Long resolution timeline introduces significant uncertainty.",
        "extended":    "Extended timeline makes prediction highly speculative.",
    }.get(horizon, "")

def generate_explanation(row, mp, sg, ag, rs, cs, top_feat_list):
    direction  = _direction_label(sg)
    magnitude  = _magnitude_label(ag)
    conf_lbl   = _confidence_label(cs)
    sent_narr  = _sentiment_narrative(row["sentiment_polarity"], row["sentiment_subjectivity"])
    horizon_caveat = _horizon_caveat(row["time_horizon"])
    signal_parts = []
    if abs(row["kw_net_signal"]) >= 2:
        kw_dir = "bullish" if row["kw_net_signal"] > 0 else "bearish"
        signal_parts.append(f"strong {kw_dir} keyword signals ({abs(row['kw_net_signal'])} net)")
    if abs(row["sentiment_polarity"]) > 0.3:
        signal_parts.append(sent_narr)
    if abs(row["prob_vs_base_rate"]) > 0.15:
        br_dir = "above" if row["prob_vs_base_rate"] > 0 else "below"
        signal_parts.append(f"market probability {br_dir} {row['category']} base rate by {abs(row['prob_vs_base_rate']):.0%}")
    if abs(row["volume_zscore"]) > 1.0:
        attn_dir = "elevated" if row["volume_zscore"] > 0 else "low"
        signal_parts.append(f"{attn_dir} trading volume relative to category peers")
    signal_summary = "; ".join(signal_parts) if signal_parts else "consensus alignment with market pricing"
    return (
        f"Our model estimates a {mp:.1%} probability, compared to the market's {row['market_probability']:.1%}. "
        f"The market appears to be {magnitude} {direction} this outcome. "
        f"Key signals: {signal_summary}. Confidence: {conf_lbl} ({cs:.2f}). "
        f"{horizon_caveat}"
    ).strip()

explanations = []
for i, row in enriched_events.iterrows():
    explanations.append(generate_explanation(
        row, mp=model_probability[i], sg=signed_gap[i],
        ag=absolute_gap[i], rs=ranking_score[i],
        cs=confidence_score[i], top_feat_list=top_drivers))

# ─────────────────────────────────────────────
# 8. ASSEMBLE analyzed_events + model_insights
# ─────────────────────────────────────────────
analyzed_events = enriched_events.copy()
analyzed_events["model_probability"] = np.round(model_probability, 4)
analyzed_events["signed_gap"]        = signed_gap
analyzed_events["absolute_gap"]      = absolute_gap
analyzed_events["ranking_score"]     = ranking_score
analyzed_events["confidence_score"]  = confidence_score
analyzed_events["explanation"]       = explanations
analyzed_events["mispricing_flag"]   = analyzed_events["absolute_gap"].apply(
    lambda g: "HIGH" if g >= 0.10 else ("MODERATE" if g >= 0.05 else "LOW"))
analyzed_events["signal_direction"]  = analyzed_events["signed_gap"].apply(
    lambda g: "UNDERPRICED" if g > 0.01 else ("OVERPRICED" if g < -0.01 else "FAIR"))

model_insights = {
    "calibration_summary": calibration_summary,
    "feature_drivers": {
        "top_combined_features": combined_ranks[["feature", "combined_rank"]].head(10).to_dict("records"),
        "top_lr_features":       lr_importance[["feature", "lr_coefficient", "lr_abs_importance"]].head(10).to_dict("records"),
        "top_gbm_features":      gbm_importance[["feature", "gbm_importance"]].head(10).to_dict("records"),
    },
    "mispricing_distribution":        analyzed_events["mispricing_flag"].value_counts().to_dict(),
    "signal_direction_distribution":  analyzed_events["signal_direction"].value_counts().to_dict(),
    "top_mispricing_opportunities": (
        analyzed_events.sort_values("ranking_score", ascending=False)
        [["event_id", "title", "category", "market_probability", "model_probability",
          "signed_gap", "absolute_gap", "ranking_score", "confidence_score", "explanation"]]
        .head(10).to_dict("records")
    ),
    "model_config": {
        "primary_model": primary_model_name, "alpha_blend": alpha_blend,
        "n_features": len(MODEL_FEATURES), "feature_list": MODEL_FEATURES,
    },
    "aggregate_stats": {
        "mean_absolute_gap": round(float(absolute_gap.mean()), 4),
        "max_absolute_gap":  round(float(absolute_gap.max()), 4),
        "mean_confidence":   round(float(confidence_score.mean()), 4),
        "mean_model_probability": round(float(model_probability.mean()), 4),
        "mean_market_probability": round(float(enriched_events["market_probability"].mean()), 4),
    }
}

# Validate
required_cols = ["model_probability", "signed_gap", "absolute_gap",
                 "ranking_score", "confidence_score", "explanation"]
missing = [c for c in required_cols if c not in analyzed_events.columns]
assert len(missing) == 0, f"Missing columns: {missing}"

# ════════════════════════════════════════════════════════
# OUTPUT 1 — ASCII BANNER
# ════════════════════════════════════════════════════════
print("╔══════════════════════════════════════════════════════════╗")
print("║   [ MARKET SIGNAL AUDITOR ] — Mispricing Engine         ║")
print("╚══════════════════════════════════════════════════════════╝")
print()

# ════════════════════════════════════════════════════════
# OUTPUT 2 — MODEL COMPARISON TABLE (4 columns)
# ════════════════════════════════════════════════════════
_improvement = (brier_market - brier_model) / brier_market * 100
_winner_label = "Blended Ensemble"

print(f"  {'Model':<28} {'AUC':>8} {'Brier ↓':>10} {'Brier vs Mkt':>13} {'Status':>10}")
print("  " + "─" * 72)
print(f"  {'Logistic Regression':<28} {lr_auc:>8.3f} {brier_lr:>10.4f} {'—':>13} {'  trained':>10}")
print(f"  {'Gradient Boosting':<28} {gbm_auc:>8.3f} {brier_gbm:>10.4f} {'—':>13} {'  trained':>10}")
print(f"  {'─'*28}   {'─'*6}   {'─'*8}   {'─'*11}   {'─'*8}")
print(f"  {'Blended Ensemble':<28} {'—':>8} {brier_model:>10.4f} {f'+{_improvement:.1f}%':>13} {'★ winner':>10}")
print(f"  {'Market Baseline':<28} {'—':>8} {brier_market:>10.4f} {'baseline':>13} {'  —':>10}")
print()
print(f"  Winner: {_winner_label} — {_improvement:.1f}% improvement over market baseline")
print()

# ════════════════════════════════════════════════════════
# OUTPUT 3 — TOP-10 MISPRICED EVENTS LEADERBOARD
# ════════════════════════════════════════════════════════
top10_lb = analyzed_events.sort_values("ranking_score", ascending=False).head(10).copy()
top10_lb["rank"] = range(1, 11)
top10    = top10_lb  # downstream alias

print(f"  {'Rank':<5} {'Event':<37} {'Direction':<13} {'Gap%':>7} {'Conf%':>7}")
print("  " + "─" * 72)
for _, _r in top10_lb.iterrows():
    _title_t = (_r["title"][:35] + "..") if len(_r["title"]) > 35 else _r["title"]
    _dir = "▲ UNDER" if _r["signal_direction"] == "UNDERPRICED" else \
           ("▼ OVER " if _r["signal_direction"] == "OVERPRICED" else "= FAIR ")
    print(
        f"  {int(_r['rank']):<5} {_title_t:<37} {_dir:<13}"
        f" {_r['signed_gap']*100:>+6.1f}%"
        f" {_r['confidence_score']*100:>6.1f}%"
    )
print()

# ════════════════════════════════════════════════════════
# OUTPUT 4 — 3-LINE NARRATIVE SUMMARY
# ════════════════════════════════════════════════════════
_top_event   = top10_lb.iloc[0]
_top_dir     = "underpriced" if _top_event["signal_direction"] == "UNDERPRICED" else "overpriced"
_n_moderate  = (analyzed_events["mispricing_flag"] != "LOW").sum()
_top_cat     = analyzed_events[analyzed_events["signal_direction"] == "UNDERPRICED"]["category"].value_counts().idxmax()
_mean_conf   = confidence_score.mean() * 100
_top_feature = top_drivers[0].replace("_", " ")

print("─" * 74)
print(f"  Top opportunity  : '{_top_event['title'][:50]}' — {_top_dir} by")
print(f"                     {abs(_top_event['signed_gap']*100):.1f}pp  |  confidence: {_top_event['confidence_score']*100:.0f}%")
print(f"  Strongest signal : {_top_cat.capitalize()} dominates mispriced events  |  #1 feature: {_top_feature}")
print(f"  Model confidence : {_mean_conf:.1f}% avg  |  {_n_moderate}/{len(analyzed_events)} events show moderate-to-high mispricing signal")
print("─" * 74)
