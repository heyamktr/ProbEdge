
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — FETCH RESOLVED POLYMARKET MARKETS (up to 150)
# ─────────────────────────────────────────────────────────────
CLOB_BASE = "https://clob.polymarket.com"
_resolved_markets = []
_next_cursor = None
_page_limit = 6  # 6 pages × ~25 results ≈ 150

print("Fetching resolved markets from Polymarket CLOB API...")
for _page in range(_page_limit):
    _params = {"closed": "true", "limit": 25}
    if _next_cursor:
        _params["next_cursor"] = _next_cursor
    try:
        _resp = requests.get(f"{CLOB_BASE}/markets", params=_params, timeout=10)
        if _resp.status_code != 200:
            print(f"  [WARNING] Page {_page+1}: HTTP {_resp.status_code} — stopping pagination.")
            break
        _data = _resp.json()
        _markets = _data.get("data", _data) if isinstance(_data, dict) else _data
        if isinstance(_markets, list):
            _resolved_markets.extend(_markets)
        elif isinstance(_markets, dict):
            # single-market response edge case
            _resolved_markets.append(_markets)
        _next_cursor = _data.get("next_cursor") if isinstance(_data, dict) else None
        print(f"  Page {_page+1}: fetched {len(_markets) if isinstance(_markets, list) else 1} markets (total so far: {len(_resolved_markets)})")
        if not _next_cursor:
            break
    except Exception as _e:
        print(f"  [WARNING] Page {_page+1}: request failed ({_e}) — stopping pagination.")
        break

print(f"Total resolved markets fetched: {len(_resolved_markets)}")

# ─────────────────────────────────────────────────────────────
# SECTION 2 — EXTRACT FIELDS FROM RESOLVED MARKETS
# ─────────────────────────────────────────────────────────────
_real_records = []
for _m in _resolved_markets:
    # Handle both CLOB market format and gamma format
    _question = (_m.get("question") or _m.get("title") or _m.get("description") or "").strip()
    _cond_id  = _m.get("condition_id") or _m.get("market_slug") or _m.get("id") or ""

    # Final price — last traded / best ask or token price
    _tokens = _m.get("tokens", [])
    _final_prob = None
    if _tokens:
        # YES token price is the market probability
        for _tok in _tokens:
            if isinstance(_tok, dict) and str(_tok.get("outcome", "")).upper() == "YES":
                _final_prob = _tok.get("price")
                break
        if _final_prob is None and len(_tokens) > 0:
            _final_prob = _tokens[0].get("price")
    if _final_prob is None:
        _final_prob = _m.get("last_trade_price") or _m.get("outcomePrices", [None])[0]

    # Resolved outcome
    _outcome_raw = (
        _m.get("resolved_yes") or
        _m.get("winner") or
        _m.get("resolution") or
        _m.get("outcome")
    )
    if isinstance(_outcome_raw, bool):
        _resolved_yes = int(_outcome_raw)
    elif str(_outcome_raw).lower() in ("yes", "1", "true", "win"):
        _resolved_yes = 1
    elif str(_outcome_raw).lower() in ("no", "0", "false", "loss"):
        _resolved_yes = 0
    else:
        _resolved_yes = None  # unknown — skip

    if _resolved_yes is None or _final_prob is None:
        continue

    try:
        _fp = float(_final_prob)
    except (ValueError, TypeError):
        continue

    _real_records.append({
        "source":         "real",
        "condition_id":   str(_cond_id),
        "question":       _question,
        "final_prob":     np.clip(_fp, 0.01, 0.99),
        "resolved_yes":   int(_resolved_yes),
    })

_df_real = pd.DataFrame(_real_records)
n_real_fetched = len(_df_real)
print(f"Parsed real resolved events with valid outcome+probability: {n_real_fetched}")

# ─────────────────────────────────────────────────────────────
# SECTION 3 — MATCH TO MODEL PREDICTIONS (event_id / title)
# ─────────────────────────────────────────────────────────────
# Use analysed_events from upstream — compare by question text similarity
_model_df = analyzed_events[["event_id", "title", "model_probability", "market_probability"]].copy()
_model_df["title_lower"] = _model_df["title"].str.lower().str.strip()

_matched_real = []
if n_real_fetched > 0:
    _df_real["question_lower"] = _df_real["question"].str.lower().str.strip()
    for _, _rr in _df_real.iterrows():
        # Fuzzy match: look for any word overlap >= 3 words in common
        _rwords = set(_rr["question_lower"].split())
        _best_idx, _best_score = None, 0
        for _mi, _mr in _model_df.iterrows():
            _mwords = set(_mr["title_lower"].split())
            _overlap = len(_rwords & _mwords)
            if _overlap > _best_score and _overlap >= 3:
                _best_score = _overlap
                _best_idx   = _mi
        _rec = {
            "source":            "real",
            "question":          _rr["question"],
            "final_prob":        _rr["final_prob"],
            "resolved_yes":      _rr["resolved_yes"],
            "model_pred":        _model_df.loc[_best_idx, "model_probability"] if _best_idx is not None else np.nan,
            "market_pred":       _model_df.loc[_best_idx, "market_probability"] if _best_idx is not None else np.nan,
            "matched_to_model":  _best_idx is not None,
        }
        _matched_real.append(_rec)

_df_real_matched = pd.DataFrame(_matched_real) if _matched_real else pd.DataFrame(
    columns=["source", "question", "final_prob", "resolved_yes", "model_pred", "market_pred", "matched_to_model"]
)
n_real_matched = int(_df_real_matched["matched_to_model"].sum()) if len(_df_real_matched) else 0
print(f"Matched to upstream model predictions: {n_real_matched} / {n_real_fetched}")

# ─────────────────────────────────────────────────────────────
# SECTION 4 — PRINCIPLED SIMULATION (200 synthetic events)
# ─────────────────────────────────────────────────────────────
# Strategy: use model's own probability estimates as the true Bernoulli bias,
# generating synthetic resolved outcomes consistent with model beliefs.
np.random.seed(42)
N_SIM = 200

# Draw model probabilities: sample with replacement from the trained model_probability distribution
_sim_model_probs = np.random.choice(model_probability, size=N_SIM, replace=True)
_sim_model_probs = np.clip(
    _sim_model_probs + np.random.normal(0, 0.03, N_SIM),  # tiny jitter for realism
    0.02, 0.98
)
# Outcomes sampled from the model's own Bernoulli(p) — model-consistent ground truth
_sim_outcomes = np.random.binomial(1, _sim_model_probs)

# Also generate corresponding "market" probabilities (slight independent noise)
_market_probs_from_analysis = enriched_events["market_probability"].values
_sim_market_probs = np.random.choice(_market_probs_from_analysis, size=N_SIM, replace=True)
_sim_market_probs = np.clip(
    _sim_market_probs + np.random.normal(0, 0.05, N_SIM),
    0.02, 0.98
)

_df_sim = pd.DataFrame({
    "source":           "simulated",
    "question":         [f"sim_event_{_j:04d}" for _j in range(N_SIM)],
    "final_prob":       _sim_market_probs,   # simulated market price at close
    "resolved_yes":     _sim_outcomes,
    "model_pred":       _sim_model_probs,
    "market_pred":      _sim_market_probs,
    "matched_to_model": True,
})
print(f"Generated {N_SIM} simulated model-consistent events")

# ─────────────────────────────────────────────────────────────
# SECTION 5 — COMBINE & COMPUTE METRICS
# ─────────────────────────────────────────────────────────────
_df_all = pd.concat([_df_real_matched, _df_sim], ignore_index=True)

# Fill NaN model_pred with final_prob (market price as proxy) for real events without match
_df_all["model_pred"]  = _df_all["model_pred"].fillna(_df_all["final_prob"])
_df_all["market_pred"] = _df_all["market_pred"].fillna(_df_all["final_prob"])

def _brier(probs, outcomes):
    return float(np.mean((np.array(probs) - np.array(outcomes)) ** 2))

def _accuracy_at_threshold(probs, outcomes, threshold):
    _mask = np.array(probs) >= threshold
    if _mask.sum() == 0:
        return None
    _preds = (_mask).astype(int)  # predict YES for all above threshold
    return float((_preds[_mask] == np.array(outcomes)[_mask]).mean())

def _calibration_by_decile(probs, outcomes, n_deciles=10):
    _probs  = np.array(probs)
    _outs   = np.array(outcomes)
    _edges  = np.percentile(_probs, np.linspace(0, 100, n_deciles + 1))
    _rows   = []
    for _d in range(n_deciles):
        _lo, _hi = _edges[_d], _edges[_d + 1]
        _mask = (_probs >= _lo) & (_probs <= _hi)
        if _mask.sum() == 0:
            continue
        _rows.append({
            "decile":        _d + 1,
            "prob_lo":       round(_lo, 3),
            "prob_hi":       round(_hi, 3),
            "n":             int(_mask.sum()),
            "mean_pred":     round(float(_probs[_mask].mean()), 4),
            "actual_rate":   round(float(_outs[_mask].mean()), 4),
            "calibration_error": round(abs(float(_probs[_mask].mean()) - float(_outs[_mask].mean())), 4),
        })
    return _rows

def _compute_group_metrics(df_group, label):
    _probs   = df_group["model_pred"].values
    _outs    = df_group["resolved_yes"].values
    _mprobs  = df_group["market_pred"].values
    _n       = len(df_group)
    if _n == 0:
        return {"label": label, "n": 0, "note": "No events in this group."}
    _brier_model  = _brier(_probs, _outs)
    _brier_market = _brier(_mprobs, _outs)
    return {
        "label":            label,
        "n":                _n,
        "brier_model":      round(_brier_model, 5),
        "brier_market":     round(_brier_market, 5),
        "brier_improvement_pct": round((_brier_market - _brier_model) / max(_brier_market, 1e-9) * 100, 2),
        "accuracy_thresh_60": _accuracy_at_threshold(_probs, _outs, 0.60),
        "accuracy_thresh_70": _accuracy_at_threshold(_probs, _outs, 0.70),
        "accuracy_thresh_80": _accuracy_at_threshold(_probs, _outs, 0.80),
        "calibration_deciles": _calibration_by_decile(_probs, _outs, 10),
        "mean_model_prob":  round(float(_probs.mean()), 4),
        "mean_actual_rate": round(float(_outs.mean()), 4),
        "outcome_prevalence": round(float(_outs.mean()), 4),
    }

# Metrics split by source and combined
_metrics_real      = _compute_group_metrics(_df_real_matched, "real_resolved_events")
_metrics_sim       = _compute_group_metrics(_df_sim, "simulated_events")
_metrics_combined  = _compute_group_metrics(_df_all, "combined_real_plus_simulated")

# Metrics on real-matched-only subset
_df_real_with_match = _df_real_matched[_df_real_matched["matched_to_model"] == True] if len(_df_real_matched) else _df_real_matched
_metrics_real_matched = _compute_group_metrics(_df_real_with_match, "real_model_matched_only")

# ─────────────────────────────────────────────────────────────
# SECTION 6 — ASSEMBLE backtest_results DICT
# ─────────────────────────────────────────────────────────────
backtest_results = {
    "data_provenance": {
        "n_resolved_markets_fetched":   n_real_fetched,
        "n_real_with_valid_outcome":    n_real_fetched,
        "n_matched_to_model":           n_real_matched,
        "n_simulated":                  N_SIM,
        "n_combined":                   len(_df_all),
        "simulation_method":            "model-consistent Bernoulli sampling from model_probability distribution",
        "transparency_note":            (
            "Simulated outcomes are generated from the model's own probability estimates as ground-truth bias. "
            "This tests internal consistency, not external generalization. "
            "Real-event metrics are labeled separately for honest evaluation."
        ),
    },
    "metrics_real_events":          _metrics_real,
    "metrics_real_model_matched":   _metrics_real_matched,
    "metrics_simulated":            _metrics_sim,
    "metrics_combined":             _metrics_combined,
    "raw_real_events":              _df_real_matched.to_dict("records") if len(_df_real_matched) > 0 else [],
}

# ─────────────────────────────────────────────────────────────
# SECTION 7 — BUILD validation_summary STRING
# ─────────────────────────────────────────────────────────────
def _fmt_acc(v):
    return f"{v*100:.1f}%" if v is not None else "N/A (no events above threshold)"

def _fmt_decile_table(deciles):
    if not deciles:
        return "  (no data)\n"
    _lines = [f"  {'Decile':>7} {'Pred Lo':>8} {'Pred Hi':>8} {'N':>5} {'Mean Pred':>10} {'Actual Rate':>12} {'Cal. Err':>9}"]
    _lines.append("  " + "─" * 65)
    for _d in deciles:
        _lines.append(
            f"  {_d['decile']:>7} {_d['prob_lo']:>8.3f} {_d['prob_hi']:>8.3f} "
            f"{_d['n']:>5} {_d['mean_pred']:>10.4f} {_d['actual_rate']:>12.4f} {_d['calibration_error']:>9.4f}"
        )
    return "\n".join(_lines) + "\n"

_mr = _metrics_real
_ms = _metrics_sim
_mc = _metrics_combined

validation_summary = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║          [ POLYMARKET BACKTEST VALIDATION REPORT ]                       ║
╚══════════════════════════════════════════════════════════════════════════╝

DATA PROVENANCE
───────────────
  Resolved markets fetched from API  : {n_real_fetched}
  Real events with valid outcome+prob: {_mr['n']}
  Matched to model predictions       : {n_real_matched}
  Simulated events (model-consistent): {N_SIM}
  Total combined                     : {len(_df_all)}

  ⚠ TRANSPARENCY: Metrics are labeled by source below. Simulated events
    use the model's own probability distribution as ground truth bias,
    testing internal consistency — NOT external predictive accuracy.
    Real-event metrics (if available) reflect genuine out-of-sample quality.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[REAL RESOLVED EVENTS]  N = {_mr['n']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brier Score (model)   : {_mr.get('brier_model',  'N/A')}
  Brier Score (market)  : {_mr.get('brier_market', 'N/A')}
  Brier improvement     : {str(_mr.get('brier_improvement_pct', 'N/A')) + '%' if _mr['n'] > 0 else 'N/A'}
  Accuracy  @ ≥60%      : {_fmt_acc(_mr.get('accuracy_thresh_60'))}
  Accuracy  @ ≥70%      : {_fmt_acc(_mr.get('accuracy_thresh_70'))}
  Accuracy  @ ≥80%      : {_fmt_acc(_mr.get('accuracy_thresh_80'))}
  Mean model prob       : {_mr.get('mean_model_prob',  'N/A')}
  Mean actual rate      : {_mr.get('mean_actual_rate', 'N/A')}

  Calibration by Decile (model predictions vs actual resolution rate):
{_fmt_decile_table(_mr.get('calibration_deciles', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SIMULATED EVENTS — model-consistent ground truth]  N = {_ms['n']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brier Score (model)   : {_ms.get('brier_model',  'N/A')}
  Brier Score (market)  : {_ms.get('brier_market', 'N/A')}
  Brier improvement     : {str(_ms.get('brier_improvement_pct', 'N/A')) + '%'}
  Accuracy  @ ≥60%      : {_fmt_acc(_ms.get('accuracy_thresh_60'))}
  Accuracy  @ ≥70%      : {_fmt_acc(_ms.get('accuracy_thresh_70'))}
  Accuracy  @ ≥80%      : {_fmt_acc(_ms.get('accuracy_thresh_80'))}
  Mean model prob       : {_ms.get('mean_model_prob',  'N/A')}
  Mean actual rate      : {_ms.get('mean_actual_rate', 'N/A')}

  Calibration by Decile (simulated):
{_fmt_decile_table(_ms.get('calibration_deciles', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[COMBINED — Real + Simulated]  N = {_mc['n']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brier Score (model)   : {_mc.get('brier_model',  'N/A')}
  Brier Score (market)  : {_mc.get('brier_market', 'N/A')}
  Brier improvement     : {str(_mc.get('brier_improvement_pct', 'N/A')) + '%'}
  Accuracy  @ ≥60%      : {_fmt_acc(_mc.get('accuracy_thresh_60'))}
  Accuracy  @ ≥70%      : {_fmt_acc(_mc.get('accuracy_thresh_70'))}
  Accuracy  @ ≥80%      : {_fmt_acc(_mc.get('accuracy_thresh_80'))}

  Calibration by Decile (combined):
{_fmt_decile_table(_mc.get('calibration_deciles', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE: A lower Brier Score is better (0 = perfect, 1 = worst).
      Brier improvement = (market_brier - model_brier) / market_brier × 100%.
      Calibration error near 0 indicates well-calibrated probabilities.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()

print(validation_summary)

# Sanity checks
assert isinstance(backtest_results, dict), "backtest_results must be a dict"
assert "metrics_real_events" in backtest_results
assert "metrics_simulated" in backtest_results
assert "metrics_combined" in backtest_results
assert isinstance(validation_summary, str) and len(validation_summary) > 100
print("\n✓ Validation: backtest_results and validation_summary created successfully.")
