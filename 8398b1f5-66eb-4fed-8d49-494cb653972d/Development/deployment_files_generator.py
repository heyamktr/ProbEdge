
# ============================================================
# Production Deployment Files Generator
# Generates all 6 deployment files and prints them clearly
# ============================================================

SEP = "=" * 70

# ── 1. Dockerfile ────────────────────────────────────────────────────────────
dockerfile_content = """\
# Dockerfile — Streamlit Application
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency manifest first (leverages Docker layer cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health-check so orchestrators know when the app is ready
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
"""

# ── 2. docker-compose.yml ────────────────────────────────────────────────────
docker_compose_content = """\
version: "3.9"

networks:
  app_network:
    driver: bridge

services:

  streamlit_app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: streamlit_app
    ports:
      - "8501:8501"
    environment:
      - POLYMARKET_API_URL=${POLYMARKET_API_URL}
      - OLLAMA_MODEL=${OLLAMA_MODEL}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - APP_ENV=${APP_ENV:-production}
    networks:
      - app_network
    restart: unless-stopped
    depends_on:
      - fastapi_app

  fastapi_app:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: fastapi_app
    ports:
      - "8000:8000"
    environment:
      - POLYMARKET_API_URL=${POLYMARKET_API_URL}
      - OLLAMA_MODEL=${OLLAMA_MODEL}
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - APP_ENV=${APP_ENV:-production}
    networks:
      - app_network
    restart: unless-stopped
    command: ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# ── 3. requirements.txt ──────────────────────────────────────────────────────
requirements_content = """\
# Web & API frameworks
streamlit==1.32.0
fastapi==0.110.0
uvicorn==0.29.0

# Data & ML
pandas==2.2.1
scikit-learn==1.4.1
numpy==1.26.4

# LLM / AI
ollama==0.1.7

# HTTP & async
httpx==0.27.0

# Visualisation
plotly==5.20.0

# NLP
textblob==0.18.0

# Config & validation
python-dotenv==1.0.1
pydantic==2.6.4
"""

# ── 4. .github/workflows/deploy.yml ─────────────────────────────────────────
github_actions_content = """\
name: CI/CD Pipeline

on:
  push:
    branches:
      - main

env:
  DOCKER_IMAGE: ${{ secrets.DOCKERHUB_USERNAME }}/polymarket-mispricing

jobs:

  # ── Job 1: Run tests ───────────────────────────────────────────────────────
  test:
    name: Run Test Suite
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run pytest
        run: python -m pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  # ── Job 2: Build Docker image ──────────────────────────────────────────────
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: test
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build Docker image
        run: |
          docker build -t ${{ env.DOCKER_IMAGE }}:${{ github.sha }} .
          docker tag ${{ env.DOCKER_IMAGE }}:${{ github.sha }} ${{ env.DOCKER_IMAGE }}:latest

  # ── Job 3: Push to Docker Hub ──────────────────────────────────────────────
  deploy:
    name: Push to Docker Hub & Deploy
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Push Docker image
        run: |
          docker push ${{ env.DOCKER_IMAGE }}:${{ github.sha }}
          docker push ${{ env.DOCKER_IMAGE }}:latest

      - name: Deployment success notification
        run: echo "✅ Image ${{ env.DOCKER_IMAGE }}:${{ github.sha }} pushed successfully."
"""

# ── 5. .env.example ──────────────────────────────────────────────────────────
env_example_content = """\
# ── Polymarket API ────────────────────────────────────────────────────────────
# Base URL for the Polymarket prediction market API
POLYMARKET_API_URL=https://gamma-api.polymarket.com

# ── LLM / Ollama ─────────────────────────────────────────────────────────────
# Ollama model to use for natural-language explanations
OLLAMA_MODEL=llama3

# ── Logging & Environment ─────────────────────────────────────────────────────
# Log verbosity: DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_LEVEL=INFO

# Deployment environment: development | staging | production
APP_ENV=development
"""

# ── 6. README.md ─────────────────────────────────────────────────────────────
readme_content = """\
# 📊 Polymarket Mispricing Detector

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Deployment](https://img.shields.io/badge/deployment-Docker-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions)

> An end-to-end machine learning system that ingests live prediction-market
> events from Polymarket, engineers rich behavioural and sentiment features,
> trains a blended LR + GBM probability model, and surfaces statistically
> significant mispricings — all served through an interactive Streamlit
> dashboard backed by a FastAPI inference service.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository and configure secrets
git clone https://github.com/your-org/polymarket-mispricing.git
cd polymarket-mispricing && cp .env.example .env

# 2. Build and start all services
docker-compose up --build -d

# 3. Open the dashboard in your browser
open http://localhost:8501
```

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                            |
|--------|-----------------------|----------------------------------------|
| GET    | `/health`             | Service liveness check                 |
| GET    | `/events`             | Fetch and score all active markets     |
| GET    | `/events/{id}`        | Get detailed mispricing report         |
| POST   | `/predict`            | Score a custom event payload           |
| GET    | `/model/importance`   | Return feature importance rankings     |
| GET    | `/model/calibration`  | Return calibration diagnostics         |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User / Client                               │
└─────────────────────┬───────────────────────┬───────────────────────┘
                      │  :8501                │  :8000
          ┌───────────▼──────────┐  ┌─────────▼──────────┐
          │   Streamlit App      │  │   FastAPI Service   │
          │  (Dashboard / UI)    │  │  (REST Inference)   │
          └───────────┬──────────┘  └─────────┬──────────┘
                      │                       │
          ┌───────────▼───────────────────────▼───────────┐
          │          Shared App Network (Docker)           │
          └───────────────────────┬───────────────────────┘
                                  │
          ┌───────────────────────▼───────────────────────┐
          │            ML Pipeline (Zerve Canvas)          │
          │                                                │
          │  data_ingestion  →  feature_engineering        │
          │        ↓                                       │
          │  probability_estimation  →  mispricing_score   │
          │        ↓                                       │
          │  evaluation_artifacts  →  model_insights       │
          └────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────▼───────────────────────┐
          │        Polymarket Gamma API  (live data)       │
          └────────────────────────────────────────────────┘
```

---

## 🤝 Contributing

1. Fork the repository and create a feature branch (`git checkout -b feat/my-feature`)
2. Run the test suite (`pytest tests/ -v`) and ensure all tests pass
3. Follow PEP 8 style; use `black` and `ruff` for formatting and linting
4. Open a pull request with a clear description of your changes — CI will run automatically
5. A maintainer will review and merge your PR within 48 hours

---

*Built with ❤️ on [Zerve](https://zerve.ai) · MIT License*
"""

# ── Assemble deployment_files dict ──────────────────────────────────────────
deployment_files = {
    "Dockerfile":                       dockerfile_content,
    "docker-compose.yml":               docker_compose_content,
    "requirements.txt":                 requirements_content,
    ".github/workflows/deploy.yml":     github_actions_content,
    ".env.example":                     env_example_content,
    "README.md":                        readme_content,
}

# ── Pretty-print every file ──────────────────────────────────────────────────
for filename, content in deployment_files.items():
    print(f"\n{SEP}")
    print(f"  FILE: {filename}")
    print(SEP)
    print(content)

# ── Summary ──────────────────────────────────────────────────────────────────
print(SEP)
print(f"\n✅  All {len(deployment_files)} deployment files generated successfully!\n")
for fname in deployment_files:
    print(f"  • {fname}")
print(f"\n'deployment_files' dict populated with {len(deployment_files)} entries.\n")
