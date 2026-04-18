# ProbEdge - Market Signal Auditor

ProbEdge is a Zerve project export for the "Market Signal Auditor", a machine learning pipeline that looks for gaps between prediction market prices and model-estimated probabilities, then ranks the strongest opportunities by gap size, confidence, and signal quality.

## What This Repository Contains

This repository is not a conventional Python package or a finished web app. It is primarily:

- A Zerve canvas export under `8398b1f5-66eb-4fed-8d49-494cb653972d`
- Python block scripts that are meant to run sequentially and share variables across blocks
- Evaluation, backtest, explanation, audit, and deployment-template generation steps
- Project-level documentation files at the repository root

The strongest way to think about this repo is "pipeline export plus supporting docs", not "ready-to-run productized service".

## Pipeline Overview

The project is organized around a multi-step prediction-market analysis workflow:

1. `data_ingestion_feature_engineering.py`
   Pulls active Polymarket markets when available and falls back to a curated sample dataset when live requests fail. It engineers 36 signals across sentiment, market structure, temporal, category, and calibration feature families.
2. `probability_estimation_mispricing.py`
   Trains calibrated Logistic Regression and Gradient Boosting models on 27 model features, blends the model signal with market probability, scores mispricing gaps, and ranks opportunities.
3. `llm_explanation_engine.py`
   Rewrites the top 20 explanations using Ollama if available, otherwise a richer rule-based fallback engine.
4. `evaluation_artifacts.py`
   Produces the evaluation visuals, including calibration curves, top feature drivers, mispricing distribution, and confidence tiers.
5. `backtest_validation.py`
   Separates real-event and simulated-event validation, and includes an explicit transparency note about what simulated results do and do not prove.
6. `market_signal_auditor.py`
   Pulls the main artifacts together into a readable audit report.
7. `dashboard_variable_audit.py`
   Verifies the downstream variables expected for dashboard-style consumption.
8. `deployment_files_generator.py`
   Generates example deployment artifacts such as Docker, Streamlit, FastAPI, CI/CD, and README templates.

## Repository Layout

- `project_documentation.md`
  Existing high-level project writeup from an earlier iteration of the project.
- `demo_video_script.md`
  A 3-minute demo script aligned to the current repository contents.
- `8398b1f5-66eb-4fed-8d49-494cb653972d/canvas.yaml`
  Zerve canvas metadata.
- `8398b1f5-66eb-4fed-8d49-494cb653972d/Development/layer.yaml`
  Block and edge definitions for the Development layer.
- `8398b1f5-66eb-4fed-8d49-494cb653972d/Development/*.py`
  The exported Zerve blocks that make up the pipeline.

## How To Run It

The scripts in this repo are designed for Zerve-style sequential execution, where upstream block variables remain available to downstream blocks. They are not currently packaged as standalone CLI programs or importable modules.

Best option:

- Run the project inside Zerve using the exported canvas structure.

If you want to run it locally, use a shared Python session or notebook and execute the blocks in order. The minimum practical order is:

1. `data_ingestion_feature_engineering.py`
2. `probability_estimation_mispricing.py`
3. `evaluation_artifacts.py`
4. `llm_explanation_engine.py`
5. `backtest_validation.py`
6. `market_signal_auditor.py`
7. `dashboard_variable_audit.py`

`deployment_files_generator.py` can be run separately because it generates template artifacts rather than consuming the full upstream state.

Main Python dependencies implied by the code:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `requests`
- Optional: `ollama`

## Important Notes And Caveats

- Live ingestion uses the Polymarket CLOB API.
- When live API access fails, the code falls back to a curated 30-event sample dataset.
- Some older generated docs in this repo still describe a 50-event sample run. Treat those numbers as historical sample outputs, not guaranteed current outputs for the checked-in code.
- The deployment generator creates example Streamlit and FastAPI deployment files, but those applications are not checked into this repository as working app source.
- The validation step mixes real resolved events with model-consistent simulated events. The simulated metrics are useful for internal consistency checks, but they are not the same thing as true out-of-sample predictive validation.

## Documented Sample Highlights

The bundled `project_documentation.md` records an earlier sample run with results such as:

- AUC around `0.980`
- Brier score `0.187` versus market baseline `0.214`
- A top flagged event around the Bitcoin `>$100K` thesis

Those sample figures are still helpful for understanding the intended output, but the current code path can produce different counts and metrics depending on whether it runs on live data or the fallback dataset.

## Why This Project Is Interesting

- It combines text-derived signals, market microstructure, and category priors in a single ranking workflow.
- It treats explanation as a first-class output rather than an afterthought.
- It is honest about uncertainty by exposing calibration, confidence, and validation caveats.
- It already includes a bridge to future deployment through generated service and dashboard templates.

## Suggested Next Steps

- Convert the shared-state block scripts into importable modules.
- Add a pinned `requirements.txt` for local reproducibility.
- Store a versioned sample output artifact so README metrics match the current code path.
- Replace simulated validation with larger resolved-market backtests.
- Turn the generated deployment templates into a checked-in app if you want a full demoable product surface.
