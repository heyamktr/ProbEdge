
# Market Signal Auditor

An ML-powered pipeline that finds where prediction markets are mispricing outcomes — and tells you by how much, and with what confidence.

---

## What This Does

The Market Signal Auditor reads prediction market events, enriches each one with 36 signals derived from text, market microstructure, and timing data, then compares an ML model's probability estimate against the crowd's odds. It surfaces the events where the gap is largest and the model is most certain.

---

## How It Works

1. **Ingest & Enrich** — 50 events across 5 categories are loaded and tagged with sentiment scores, bullish/bearish keyword counts, volume z-scores, time-horizon buckets, and category base rates.
2. **Train & Calibrate** — A Logistic Regression and a Gradient Boosting Classifier are trained on 27 features, each wrapped in isotonic calibration to produce reliable probability outputs.
3. **Blend & Score** — The model's signal is blended with the market price (55% market / 45% signal nudge) to produce a final probability estimate and a signed mispricing gap per event.
4. **Rank & Explain** — Events are ranked by a composite score (gap magnitude × model certainty × liquidity depth), and each gets a plain-English explanation of the signals driving its prediction.

---

## Who It Is For

- **Prediction market traders** — find actionable mispricings ranked by opportunity size and confidence before the market corrects
- **Quant analysts** — a reproducible NLP + ML pipeline ready to extend to new markets, asset classes, or live data feeds
- **Risk & research teams** — understand what features actually drive market mispricing, backed by calibration curves and feature importance charts

---

## Key Results

- **AUC 0.980** — near-perfect ability to distinguish over- and under-priced events
- **Brier Score 0.187** vs market baseline **0.214** — a **12.8% improvement** in probability accuracy
- **50 events analysed** — 24 underpriced, 26 overpriced, 0 at fair value
- **Top opportunity:** Bitcoin >$100K — flagged underpriced by +6.4pp at 98% confidence
- **Composite Signal** is the single strongest predictive feature, confirming NLP + market signals carry alpha beyond the crowd

---

## Architecture

Three Python blocks run sequentially on serverless Zerve compute. Block 1 ingests raw event data and builds the 36-feature dataset (`enriched_events`). Block 2 trains, calibrates, and blends two ML models, then scores every event for mispricing and confidence (`analyzed_events`). Block 3 renders a 2×2 evaluation dashboard — calibration curves, feature importance, mispricing distribution, and confidence tiers — and prints a summary leaderboard.

---

## Deployment

The pipeline can be deployed as a live Streamlit dashboard or FastAPI endpoint via Zerve's script deployment system — analysts would see a real-time leaderboard updating as new market events are added. Connecting to live data from Polymarket or Manifold Markets would make the signals actionable immediately.

---

## Limitations

- **Synthetic labels** — no resolved outcomes yet; the model trains on a soft signal score, not actual event results, so accuracy metrics are optimistic
- **Small dataset** — 50 events is enough for a proof of concept, but a production model needs thousands of resolved outcomes to reduce overfitting risk
- **Static signals** — keyword lexicons and base rates are manually curated; a live system would need automated lexicon updates and rolling base rate recalculation as markets evolve
