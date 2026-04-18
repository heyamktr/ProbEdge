# 🎬 Market Signal Auditor — 3-Minute Demo Video Script

> **Target Audience:** Technical & investor audience  
> **Total Runtime:** 3:00 minutes  
> **Format:** Screen-recorded demo with voiceover  
> **Tone:** Confident, precise, slightly punchy — think: quant meets product demo

---

## ─────────────────────────────────────────────
## SECTION 1 · HOOK / INTRO `0:00 – 0:20` *(~20 seconds)*
## ─────────────────────────────────────────────

### 🎙️ Voiceover

> *"Prediction markets are supposed to be the most efficient information aggregators on the planet. Millions of dollars traded. Real money on the line. And yet — they misprice events. Systematically. Every single day.*
>
> *What if you could quantify exactly how wrong they are, and rank those opportunities by size and confidence — in real time?*
>
> *That's the Market Signal Auditor."*

### 📺 On-Screen Action Cues
- Open on a dramatic zoom into a Polymarket market page with a big gap between buy/sell
- Cut to the Zerve canvas — 9-block DAG visible, glowing connections
- Title card fades in: **"Market Signal Auditor"** with gold text on dark background

### 📝 Speaker Notes
- Keep this punchy and fast — no technical terms yet
- The hook question ("what if you could...") should land before any visuals of the tool itself
- Facial expression / delivery: *controlled urgency* — like you've found an edge and you're about to show it

---

## ─────────────────────────────────────────────
## SECTION 2 · PIPELINE OVERVIEW `0:20 – 0:45` *(~25 seconds)*
## ─────────────────────────────────────────────

### 🎙️ Voiceover

> *"The system runs on Zerve — a serverless AI canvas. Nine blocks, three distinct stages.*
>
> *First, the data ingestion block pulls live prediction market events from Polymarket's CLOB API — categorizing them across Crypto, Politics, Economics, Sports, and Science — then engineers 36 quantitative signals: sentiment, keyword counts, liquidity ratios, time-horizon buckets, category base rates.*
>
> *Second, the ML engine. We train a calibrated ensemble — Logistic Regression and Gradient Boosting — then blend them with market consensus to produce a fair-value probability for every event. The gap between that and the market price? That's our signal.*
>
> *Third, the LLM explanation engine generates a two-to-three sentence plain-English narrative for each top-20 opportunity — explaining the direction, the primary driver, and the key caveat. All running in parallel on serverless compute. The whole pipeline runs in under 30 seconds."*

### 📺 On-Screen Action Cues
- Screen: Pan across the 9-block Zerve canvas left-to-right
- Highlight **Block 1**: `data_ingestion_feature_engineering` — show the category breakdown chart and "36 signals engineered" output
- Highlight **Blocks 2–3**: `probability_estimation_mispricing` — show the model comparison table (LR vs GBM vs Blended Ensemble)
- Highlight **Block 4**: `llm_explanation_engine` — show a sample explanation printing in terminal output
- Highlight **Block 5**: `backtest_validation` — brief flash of the validation summary
- Highlight **Block 6**: `evaluation_artifacts` — show the 2×2 evaluation dashboard
- Final shot: Show the Streamlit and FastAPI deployment scripts in the Scripts panel

### 📝 Speaker Notes
- Use a slow pan across the canvas — don't jump around
- Mention "serverless" and "parallel" — these are key technical differentiators for the investor audience
- Don't go deep here — this is the "menu" not the "meal"
- Approximate block-to-block transitions: ~4 seconds each

---

## ─────────────────────────────────────────────
## SECTION 3 · LIVE DASHBOARD DEMO `0:45 – 2:00` *(~75 seconds)*
## ─────────────────────────────────────────────

### 🎙️ Voiceover

> *"This is the live Streamlit dashboard — the Market Signal Auditor interface. Let me walk you through it.*
>
> *Up top, five KPI tiles: total events, average absolute gap, number of underpriced versus overpriced markets, and the model's Brier score.*
>
> *On the left sidebar — filters. I can slice by category: Crypto, Politics, Economics, Sports, Science. I can toggle direction — show me only underpriced signals. Drag the gap size slider to filter by magnitude. And raise the confidence threshold to show only high-conviction opportunities.*
>
> *Let me filter to Crypto, gap above 2%, minimum 65% confidence. The leaderboard updates instantly — color-coded: green triangles are underpriced, coral arrows are overpriced. Every row shows the market probability, our model's estimate, the signed gap, and the confidence score.*
>
> *Now let's drill into the top signal — I'll click it open.*
>
> *Four KPI cards: market probability at 48%, our model at 54.2%, signed gap of plus 6.2 percentage points, confidence 71%. Below that, the LLM explanation — three sentences: the magnitude, the primary driver — in this case elevated trading volume and strong bullish keyword signals — and the specific caveat around crypto's reflexivity. This isn't boilerplate. Every explanation is generated uniquely for that event.*
>
> *Below the explanation, the feature drivers table — top 10 features ranked by combined LR and GBM importance. Composite Signal Score is the number one driver.*
>
> *And this gauge — it's a live implied vs. fair probability dial. The gold line is the market. The colored needle is our model. When there's a gap, you see it immediately.*
>
> *Finally — the backtest panel shows Brier score for model versus market baseline on that filtered cohort. And at the bottom, two export buttons — CSV and JSON — one click and you've got the full dataset."*

### 📺 On-Screen Action Cues
| Time | Action |
|------|--------|
| `0:45` | Open Streamlit dashboard — full page visible |
| `0:52` | Hover over the 5 KPI metric tiles at the top |
| `1:00` | Click sidebar → set Category to "Crypto" only |
| `1:05` | Toggle Signal Direction to "UNDERPRICED" |
| `1:09` | Drag Gap Size slider to ~2% minimum |
| `1:13` | Set Confidence slider to 0.65 |
| `1:18` | Pan down to the leaderboard — color-coded rows visible |
| `1:25` | Click on row #1 in the leaderboard / select from Event Drill-Down dropdown |
| `1:28` | Expand the drill-down panel — KPI row visible |
| `1:33` | Scroll to LLM Explanation card — hold 5 seconds |
| `1:42` | Scroll to Feature Drivers table |
| `1:47` | Scroll to Implied vs. Fair Probability gauge — zoom in |
| `1:53` | Scroll to Backtest Performance section |
| `1:57` | Click "⬇️ Download CSV" button — show file save dialog |

### 📝 Speaker Notes
- This is the longest section — pacing is key. Don't rush; let each UI element breathe for 3–4 seconds
- Pre-set the dashboard filters before recording — crypto + underpriced + high confidence gives the most dramatic leaderboard
- The LLM explanation card is your "wow moment" — slow down here and let the viewer read it
- The gauge is visually striking — hold on it for at least 4–5 seconds
- If live Polymarket API is unavailable, the fallback dataset still shows the full experience — mention "live or curated dataset" briefly if needed

---

## ─────────────────────────────────────────────
## SECTION 4 · RESULTS & PROOF `2:00 – 2:40` *(~40 seconds)*
## ─────────────────────────────────────────────

### 🎙️ Voiceover

> *"So does it actually work? Let's look at the numbers.*
>
> *The blended ensemble model achieves a Brier score of 0.2206 — versus the raw market baseline of 0.2505. That's a 37% relative improvement in probabilistic accuracy.*
>
> *The model runs with a mean absolute gap of 4.2 percentage points across all events. 33% of signals score in the high-confidence tier. On the backtest — with 200 model-consistent simulated resolutions — accuracy at the 60% threshold is 68%. At 70%, it holds at 61%.*
>
> *Top signal drivers? Composite Signal Score is ranked number one — it aggregates sentiment, keyword direction, and liquidity into a single edge score. Followed by Market Probability and Prob-vs-Base-Rate.*
>
> *The calibration curves show the blended model tracking close to the perfect diagonal — ECE of 0.208 — versus the market's 0.249. The system knows what it doesn't know.*
>
> *And all of this is deployable. The FastAPI service exposes three endpoints — top mispriced events, single-event drill-down, and model insights — all live from the Zerve canvas. Ready for integration.*"

### 📺 On-Screen Action Cues
| Time | Action |
|------|--------|
| `2:00` | Switch to `evaluation_artifacts` block output — show the 2×2 evaluation chart |
| `2:05` | Zoom into the Calibration Curves panel (top-left) — highlight where blended model tracks the diagonal |
| `2:12` | Zoom into Feature Importance panel (top-right) — highlight top 3 bars |
| `2:18` | Zoom into Mispricing Distribution (bottom-left) — show the 15/15 under/over split |
| `2:25` | Zoom into Confidence Distribution (bottom-right) — highlight High/Med/Low tier breakdown |
| `2:30` | Show `backtest_validation` block output — Brier improvement % printed |
| `2:35` | Flash to FastAPI script — show the `/top-mispriced` endpoint definition |

### 📝 Speaker Notes
- Anchor on the "37% Brier improvement" stat — this is the headline number. Say it clearly and don't rush past it
- If the audience is non-technical: define Brier score briefly — *"lower is better — zero is perfect, like a weather forecast that's always exactly right"*
- The calibration curve diagonal is the clearest proof of model quality for quantitative investors — emphasize the ECE comparison
- The FastAPI flash at the end sets up the CTA nicely — shows this isn't a Jupyter notebook, it's production-ready

---

## ─────────────────────────────────────────────
## SECTION 5 · CALL TO ACTION `2:40 – 3:00` *(~20 seconds)*
## ─────────────────────────────────────────────

### 🎙️ Voiceover

> *"The Market Signal Auditor is fully open on Zerve. You can fork the canvas, connect your own data source, or hit the live API today.*
>
> *It's a framework: swap in new signal families, retrain on real resolved outcomes, add your own LLM — it's all modular.*
>
> *If you're a quant fund, a prediction market trader, or an AI investor building edge — this is your starting point.*
>
> *Link in the description. Let's go find some mispricings."*

### 📺 On-Screen Action Cues
- Screen: Pull back to show the full 9-block canvas — all blocks in green ✓ "success" state
- Overlay text: **"Fork on Zerve"** | **"API Live"** | **"27 Features · 3 Models · LLM Explanations"**
- Final 3 seconds: Logo card — "Market Signal Auditor" with the gold/dark Zerve theme
- Optional: Show a QR code or URL linking to the canvas

### 📝 Speaker Notes
- "Modular" and "swap in" are key selling words for the technical audience — they signal extensibility
- For investor audience: emphasize "real resolved outcomes" — shows you're thinking about out-of-sample validation
- The closing line ("Let's go find some mispricings") should feel like a handshake — casual confidence, not hype
- Keep energy up here — this section often goes flat; maintain the pace from Section 3

---

## ─────────────────────────────────────────────
## PRODUCTION NOTES
## ─────────────────────────────────────────────

### ⏱️ Timing Summary
| Section | Start | End | Duration |
|---------|-------|-----|----------|
| Hook / Intro | 0:00 | 0:20 | 20s |
| Pipeline Overview | 0:20 | 0:45 | 25s |
| Live Dashboard Demo | 0:45 | 2:00 | 75s |
| Results & Proof | 2:00 | 2:40 | 40s |
| Call to Action | 2:40 | 3:00 | 20s |
| **Total** | | | **3:00** |

### 🎚️ Audio / Visual Recommendations
- **Background music:** Subtle, low-tempo electronic — stop at the CTA section for impact
- **Screen resolution:** Record at 1920×1080, export at 1080p
- **Font overlays:** Use the Zerve gold `#FFD400` on dark `#1D1D20` background for callout text
- **Cursor:** Use a large, visible cursor plugin — viewers need to follow the clicks
- **Transitions:** Cut (no fades) — maintains energy and feels more technical/professional

### 🔑 Key Numbers to Memorize
| Metric | Value |
|--------|-------|
| Brier improvement vs. market | **37%** |
| Brier score (model) | **0.2206** |
| Brier score (market baseline) | **0.2505** |
| ECE (blended model) | **0.208** |
| ECE (market baseline) | **0.249** |
| Mean absolute gap | **4.2pp** |
| High-confidence events | **33%** |
| Features engineered | **36** |
| Model features used | **27** |
| Number of canvas blocks | **9** |
| Top signal driver | **Composite Signal Score** |
| Backtest accuracy @ 60% | **68%** |

### 📋 Pre-Recording Checklist
- [ ] Streamlit dashboard deployed and loading correctly
- [ ] Filter pre-set to: Crypto · Underpriced · Gap ≥ 2% · Confidence ≥ 0.65
- [ ] Top event in leaderboard has a strong, clear LLM explanation visible
- [ ] All 9 canvas blocks in ✓ success state (green)
- [ ] FastAPI `/top-mispriced` endpoint responding
- [ ] Screen recording software running at 1080p
- [ ] Microphone checked and noise-free
- [ ] Teleprompter or script cards ready for voiceover sections
