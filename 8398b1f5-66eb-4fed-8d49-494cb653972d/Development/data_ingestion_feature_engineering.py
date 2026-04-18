
import pandas as pd
import numpy as np
import re
import requests
import time
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime, timedelta

# ── Zerve Design System ────────────────────────────────────────────
BG       = "#1D1D20"
TEXT_PRI = "#fbfbff"
TEXT_SEC = "#909094"
BLUE     = "#A1C9F4"
ORANGE   = "#FFB482"
GREEN    = "#8DE5A1"
CORAL    = "#FF9F9B"
LAVENDER = "#D0BBFF"
GOLD     = "#ffd400"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "text.color": TEXT_PRI, "axes.labelcolor": TEXT_PRI,
    "xtick.color": TEXT_PRI, "ytick.color": TEXT_PRI,
    "axes.edgecolor": TEXT_SEC, "grid.color": "#333336",
    "font.family": "sans-serif",
})

today = datetime.today()

# ══════════════════════════════════════════════════════════════════
# 1.  POLYMARKET API CLIENT
# ══════════════════════════════════════════════════════════════════

CATEGORY_KEYWORDS = {
    "crypto": [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi",
        "nft", "token", "coinbase", "binance", "solana", "ripple", "xrp",
        "altcoin", "stablecoin", "cbdc", "web3", "dao", "wallet", "exchange",
    ],
    "politics": [
        "election", "president", "senate", "congress", "democrat", "republican",
        "biden", "trump", "nato", "parliament", "vote", "political", "government",
        "minister", "treaty", "sanctions", "legislation", "bill", "policy",
        "referendum", "inauguration", "impeach",
    ],
    "sports": [
        "nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball",
        "tennis", "golf", "formula", "f1", "olympic", "championship", "league",
        "cup", "tournament", "athlete", "team", "match", "season", "playoffs",
        "superbowl", "world cup", "wimbledon",
    ],
    "economics": [
        "fed", "federal reserve", "interest rate", "inflation", "gdp", "recession",
        "stock", "market", "s&p", "nasdaq", "dow", "economy", "unemployment",
        "jobs", "trade", "tariff", "imf", "world bank", "oil", "energy", "dollar",
        "euro", "yen", "currency", "bond", "yield",
    ],
    "science": [
        "ai", "artificial intelligence", "gpt", "llm", "nasa", "spacex", "cancer",
        "vaccine", "drug", "fda", "climate", "temperature", "fusion", "quantum",
        "crispr", "gene", "alzheimer", "mars", "space", "launch", "biotech",
        "pharma", "treatment", "trial",
    ],
}

def infer_category(title: str, description: str = "") -> str:
    """Keyword-match title+description to a category; default 'economics'."""
    combined = (title + " " + description).lower()
    scores = {}
    for cat, kws in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in kws if kw in combined)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "economics"


class PolymarketClient:
    """Fetches active prediction markets from Polymarket CLOB API."""

    BASE_URL = "https://clob.polymarket.com/markets"
    MAX_MARKETS = 200
    PAGE_SIZE   = 100          # API default / max
    TIMEOUT     = 10           # seconds per request
    MAX_RETRIES = 3
    BACKOFF_BASE = 1.5         # seconds

    def _get_page(self, next_cursor: str = "") -> dict:
        """Fetch one page with retry + exponential backoff."""
        params = {"limit": self.PAGE_SIZE, "active": "true"}
        if next_cursor:
            params["next_cursor"] = next_cursor

        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                wait = self.BACKOFF_BASE * (2 ** attempt)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
                else:
                    raise exc   # let caller decide

    def _parse_market(self, m: dict) -> dict | None:
        """Extract and normalise a single market record."""
        # Market probability: midpoint of best bid / ask on first token
        tokens = m.get("tokens", [])
        prob = None
        for tok in tokens:
            bid = tok.get("best_bid")
            ask = tok.get("best_ask")
            if bid is not None and ask is not None:
                try:
                    prob = (float(bid) + float(ask)) / 2.0
                    prob = round(max(0.0, min(1.0, prob)), 4)
                    break
                except (TypeError, ValueError):
                    pass
        if prob is None:
            # Fall back to outcome_prices if present
            op = m.get("outcome_prices")
            if op:
                try:
                    prices = [float(x) for x in op if x is not None]
                    if prices:
                        prob = round(max(0.0, min(1.0, prices[0])), 4)
                except (TypeError, ValueError):
                    pass
        if prob is None:
            return None   # skip markets with no price signal

        # Resolution date
        end_ts = m.get("end_date_iso") or m.get("game_start_time") or ""
        try:
            resolution_date = pd.to_datetime(end_ts, utc=True).tz_localize(None)
        except Exception:
            resolution_date = pd.Timestamp(today + timedelta(days=180))

        # Volume & liquidity (may be string or float)
        def _safe_float(v, default=0.0):
            try:
                return float(v) if v is not None else default
            except (TypeError, ValueError):
                return default

        volume    = _safe_float(m.get("volume"))
        liquidity = _safe_float(m.get("liquidity"))

        title       = m.get("question", "")[:200]
        description = m.get("description", "") or ""
        category    = infer_category(title, description)

        return {
            "event_id":          m.get("condition_id", m.get("id", ""))[:20],
            "title":             title,
            "category":          category,
            "market_probability": prob,
            "resolution_date":   resolution_date,
            "description":       description[:400] if description else title,
            "volume":            volume,
            "liquidity":         liquidity,
        }

    def fetch(self) -> tuple[list[dict], str]:
        """
        Returns (records_list, data_source_label).
        Raises on network failure so caller can catch and use fallback.
        """
        records = []
        cursor  = ""
        pages   = 0
        max_pages = math.ceil(self.MAX_MARKETS / self.PAGE_SIZE) + 1

        while len(records) < self.MAX_MARKETS and pages < max_pages:
            data = self._get_page(cursor)
            for m in data.get("data", []):
                parsed = self._parse_market(m)
                if parsed:
                    records.append(parsed)
                if len(records) >= self.MAX_MARKETS:
                    break

            # Pagination
            cursor = data.get("next_cursor", "")
            pages += 1
            if not cursor or cursor == "LTE=":   # end of results
                break

        if not records:
            raise ValueError("API returned 0 parseable markets")

        return records, "live"


# ══════════════════════════════════════════════════════════════════
# 2.  FALLBACK: 30-EVENT CURATED SAMPLE DATASET
# ══════════════════════════════════════════════════════════════════

_FALLBACK_EVENTS = [
    # CRYPTO (6)
    {"event_id": "CRY_001", "title": "Bitcoin exceeds $100,000 by end of 2025",          "category": "crypto",     "market_probability": 0.48, "resolution_date": today + timedelta(days=180), "description": "ETF inflows accelerating post-approval. Halving event historically bullish. Institutional demand growing strongly.",              "volume": 5_800_000, "liquidity": 1_350_000},
    {"event_id": "CRY_002", "title": "Ethereum spot ETF sees $1B inflows in 30 days",    "category": "crypto",     "market_probability": 0.54, "resolution_date": today + timedelta(days=60),  "description": "SEC approval pathway cleared. Institutional interest in ETH exposure rising quickly through regulated vehicles.",             "volume": 3_200_000, "liquidity":   740_000},
    {"event_id": "CRY_003", "title": "Solana surpasses Ethereum in daily transactions",  "category": "crypto",     "market_probability": 0.44, "resolution_date": today + timedelta(days=120), "description": "Solana network throughput advantage demonstrated. Developer migration trend showing positive momentum.",                         "volume":   780_000, "liquidity":   180_000},
    {"event_id": "CRY_004", "title": "Crypto total market cap exceeds $4 trillion",      "category": "crypto",     "market_probability": 0.32, "resolution_date": today + timedelta(days=270), "description": "Requires broad altcoin participation alongside BTC rally. Macro liquidity conditions remain key variable.",                       "volume": 2_100_000, "liquidity":   490_000},
    {"event_id": "CRY_005", "title": "CBDC launched by Federal Reserve before 2026",     "category": "crypto",     "market_probability": 0.07, "resolution_date": today + timedelta(days=540), "description": "Political resistance strong. Technical and privacy challenges unresolved. Timeline uncertain.",                                     "volume":   450_000, "liquidity":    98_000},
    {"event_id": "CRY_006", "title": "Bitcoin mining difficulty reaches 200T",           "category": "crypto",     "market_probability": 0.61, "resolution_date": today + timedelta(days=90),  "description": "Consistent hash rate growth post-halving. Institutional miners expanding capacity despite energy cost pressures.",               "volume":   390_000, "liquidity":    92_000},
    # POLITICS (6)
    {"event_id": "POL_001", "title": "Democrats win 2026 Senate majority",               "category": "politics",   "market_probability": 0.41, "resolution_date": today + timedelta(days=550), "description": "Polling data suggests tight race with late momentum shifts. Turnout operations in battleground states critical.",                  "volume": 1_250_000, "liquidity":   320_000},
    {"event_id": "POL_002", "title": "NATO admits Ukraine as member before 2027",        "category": "politics",   "market_probability": 0.11, "resolution_date": today + timedelta(days=720), "description": "Ongoing conflict makes membership complex. Strong diplomatic resistance from several member states remains.",                       "volume":   670_000, "liquidity":   145_000},
    {"event_id": "POL_003", "title": "US debt ceiling deal reached before default",      "category": "politics",   "market_probability": 0.87, "resolution_date": today + timedelta(days=30),  "description": "Historical precedent and bipartisan pressure suggest last-minute deal likely. Markets pricing in resolution.",                   "volume": 2_100_000, "liquidity":   560_000},
    {"event_id": "POL_004", "title": "EU approves AI Act in final enforceable form",     "category": "politics",   "market_probability": 0.83, "resolution_date": today + timedelta(days=120), "description": "Strong regulatory momentum. Minor technical amendments expected but core framework largely agreed upon.",                          "volume":   550_000, "liquidity":   130_000},
    {"event_id": "POL_005", "title": "UK Labour wins snap general election",             "category": "politics",   "market_probability": 0.74, "resolution_date": today + timedelta(days=60),  "description": "Labour leads by 20+ points consistently. Conservative incumbency advantage has evaporated amid economic pressures.",              "volume":   890_000, "liquidity":   210_000},
    {"event_id": "POL_006", "title": "Israel-Hamas ceasefire holds beyond 90 days",     "category": "politics",   "market_probability": 0.29, "resolution_date": today + timedelta(days=150), "description": "Fragile ceasefire dependent on humanitarian corridor agreements. Spoiler groups on both sides pose persistent risk.",             "volume":   750_000, "liquidity":   170_000},
    # SPORTS (6)
    {"event_id": "SPT_001", "title": "Manchester City wins 2024-25 Premier League",     "category": "sports",     "market_probability": 0.44, "resolution_date": today + timedelta(days=270), "description": "Guardiola's side dominant but faces stronger competition from Arsenal and Liverpool this season.",                                "volume": 2_800_000, "liquidity":   640_000},
    {"event_id": "SPT_002", "title": "USA wins 2024 Summer Olympics gold medal count",  "category": "sports",     "market_probability": 0.63, "resolution_date": today + timedelta(days=40),  "description": "Historical dominance in total medals. China competitive but USA depth advantage significant across multiple sports.",            "volume":   890_000, "liquidity":   200_000},
    {"event_id": "SPT_003", "title": "Real Madrid wins 2024-25 Champions League",       "category": "sports",     "market_probability": 0.23, "resolution_date": today + timedelta(days=300), "description": "Strong squad depth and tactical flexibility. Recent UCL success breeds familiarity with high-pressure rounds.",                    "volume": 2_100_000, "liquidity":   480_000},
    {"event_id": "SPT_004", "title": "New Zealand wins 2027 Rugby World Cup",           "category": "sports",     "market_probability": 0.30, "resolution_date": today + timedelta(days=1100),"description": "All Blacks rebuilding under new coach. South Africa formidable defending champions.",                                           "volume":   420_000, "liquidity":   105_000},
    {"event_id": "SPT_005", "title": "Brazil wins 2026 FIFA World Cup",                 "category": "sports",     "market_probability": 0.16, "resolution_date": today + timedelta(days=730), "description": "Brazil rebuilding with young squad after 2022 disappointment. Consistency remains a challenge heading into 2026.",               "volume": 3_200_000, "liquidity":   720_000},
    {"event_id": "SPT_006", "title": "Novak Djokovic wins 2025 Wimbledon",              "category": "sports",     "market_probability": 0.35, "resolution_date": today + timedelta(days=380), "description": "Defending champion faces Alcaraz rematch threat. Historically dominant on grass but physical fitness scrutinised.",              "volume": 1_100_000, "liquidity":   290_000},
    # ECONOMICS (6)
    {"event_id": "ECO_001", "title": "US inflation falls below 2% by Q4 2025",          "category": "economics",  "market_probability": 0.29, "resolution_date": today + timedelta(days=180), "description": "Progress on disinflation continuing but services inflation sticky. Fed remains data-dependent.",                                   "volume": 1_100_000, "liquidity":   280_000},
    {"event_id": "ECO_002", "title": "Fed cuts rates 3 or more times in 2025",          "category": "economics",  "market_probability": 0.22, "resolution_date": today + timedelta(days=270), "description": "Market pricing only 1-2 cuts. Strong labor market reduces urgency for aggressive easing cycle this year.",                        "volume": 2_400_000, "liquidity":   580_000},
    {"event_id": "ECO_003", "title": "S&P 500 ends 2025 above 5800",                    "category": "economics",  "market_probability": 0.51, "resolution_date": today + timedelta(days=180), "description": "AI-driven earnings optimism supporting valuations. Index elevated multiples limit upside but momentum persists.",                 "volume": 3_100_000, "liquidity":   710_000},
    {"event_id": "ECO_004", "title": "Oil price exceeds $100 per barrel in 2025",       "category": "economics",  "market_probability": 0.31, "resolution_date": today + timedelta(days=240), "description": "OPEC+ supply discipline maintaining floor. Geopolitical risk premium elevated by Middle East tensions.",                         "volume": 1_350_000, "liquidity":   320_000},
    {"event_id": "ECO_005", "title": "China GDP growth exceeds 5% in 2025",             "category": "economics",  "market_probability": 0.57, "resolution_date": today + timedelta(days=200), "description": "Government target set at around 5%. Stimulus measures supporting growth but property sector headwinds persist.",                  "volume":   940_000, "liquidity":   215_000},
    {"event_id": "ECO_006", "title": "Japan raises interest rates above 1%",             "category": "economics",  "market_probability": 0.46, "resolution_date": today + timedelta(days=270), "description": "BOJ gradual normalisation underway. Wage growth data supportive of further tightening through 2025.",                            "volume":   720_000, "liquidity":   165_000},
    # SCIENCE (6)
    {"event_id": "SCI_001", "title": "FDA approves first CRISPR cancer therapy",        "category": "science",    "market_probability": 0.62, "resolution_date": today + timedelta(days=180), "description": "Vertex and CRISPR Therapeutics breakthrough. Oncology applications advancing rapidly through clinical trials.",                     "volume": 1_200_000, "liquidity":   285_000},
    {"event_id": "SCI_002", "title": "GPT-5 achieves AGI benchmark performance",        "category": "science",    "market_probability": 0.16, "resolution_date": today + timedelta(days=365), "description": "Definition of AGI contested. Scaling laws showing diminishing returns. Novel architecture may be required.",                       "volume": 2_300_000, "liquidity":   540_000},
    {"event_id": "SCI_003", "title": "SpaceX Starship reaches orbit successfully",      "category": "science",    "market_probability": 0.76, "resolution_date": today + timedelta(days=120), "description": "Iterative testing showing major progress. Fourth test flight demonstrated critical improvements in reentry.",                        "volume":   980_000, "liquidity":   230_000},
    {"event_id": "SCI_004", "title": "Nuclear fusion net energy gain independently replicated", "category": "science", "market_probability": 0.41, "resolution_date": today + timedelta(days=365), "description": "NIF 2022 breakthrough needs confirmation. Private companies advancing Commonwealth Fusion rapidly.",                            "volume":   650_000, "liquidity":   155_000},
    {"event_id": "SCI_005", "title": "Global temperature breaches 1.5C above pre-industrial", "category": "science", "market_probability": 0.69, "resolution_date": today + timedelta(days=180), "description": "2023 already breached threshold temporarily. El Nino amplifying underlying warming trend persistently.",                         "volume":   720_000, "liquidity":   170_000},
    {"event_id": "SCI_006", "title": "Alzheimer's blood test achieves clinical adoption", "category": "science",  "market_probability": 0.55, "resolution_date": today + timedelta(days=365), "description": "p-tau217 biomarker showing high accuracy. Healthcare system integration and reimbursement pathways being established.",           "volume":   560_000, "liquidity":   135_000},
]

# ══════════════════════════════════════════════════════════════════
# 3.  FETCH LIVE DATA (WITH FALLBACK)
# ══════════════════════════════════════════════════════════════════

_client = PolymarketClient()
_live_failed = False
_fail_reason  = ""

try:
    raw_events, data_source_label = _client.fetch()
except Exception as _exc:
    _live_failed = True
    _fail_reason = str(_exc)
    raw_events = _FALLBACK_EVENTS.copy()
    data_source_label = "fallback"

# ══════════════════════════════════════════════════════════════════
# 4.  BUILD df_raw WITH IDENTICAL SCHEMA TO EXISTING PIPELINE
# ══════════════════════════════════════════════════════════════════

df_raw = pd.DataFrame(raw_events)
df_raw["resolution_date"] = pd.to_datetime(df_raw["resolution_date"], utc=False, errors="coerce")
# Fill any unparseable dates
df_raw["resolution_date"] = df_raw["resolution_date"].fillna(pd.Timestamp(today + timedelta(days=180)))
df_raw["days_to_resolution"] = (df_raw["resolution_date"] - pd.Timestamp(today.date())).dt.days.clip(lower=0)

# ── Sentinel positive/negative words (defined for downstream reference) ─
POSITIVE_WORDS = {
    "strong", "dominant", "favorable", "advance", "growing", "recovery", "breakthrough",
    "progress", "success", "optimism", "compelling", "significant", "accelerating",
    "positive", "rally", "surge", "improving", "advantage", "promising", "support",
    "momentum", "gain", "clear", "compelling", "leading", "robust", "historically",
    "confidence", "upside", "winning"
}
NEGATIVE_WORDS = {
    "uncertain", "challenging", "declining", "resistance", "risk", "unpopular",
    "collapse", "worry", "drag", "disappointing", "difficult", "unresolved",
    "complex", "limited", "headwind", "sticky", "persistent", "concerns",
    "worries", "threats", "unlikely", "deficit", "pressure", "threatening",
    "deflated", "contested", "diminishing"
}

def compute_text_sentiment(text):
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    polarity = round((pos - neg) / (pos + neg + 1), 4)
    subjectivity = round((pos + neg) / (len(tokens) + 1), 4)
    return polarity, subjectivity

sentiments = df_raw["description"].apply(compute_text_sentiment)
df_raw["sentiment_polarity"]    = sentiments.apply(lambda x: x[0])
df_raw["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])

# ── Keyword features ─────────────────────────────────────────────
BULLISH_KEYWORDS = [
    "momentum", "strong", "dominant", "favorable", "advance", "growing", "recovery",
    "breakthrough", "progress", "success", "optimism", "compelling", "significant",
    "accelerating", "positive", "rally", "surge", "improving"
]
BEARISH_KEYWORDS = [
    "uncertain", "challenging", "declining", "resistance", "risk", "unpopular",
    "collapse", "worry", "drag", "disappointing", "difficult", "unresolved",
    "complex", "limited", "headwind", "sticky", "persistent"
]
UNCERTAINTY_KEYWORDS = [
    "could", "may", "might", "unclear", "contested", "uncertain",
    "possible", "question", "volatile", "expected", "remains"
]

def keyword_count(text, keywords):
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)

df_raw["kw_bullish_count"]     = df_raw["description"].apply(lambda t: keyword_count(t, BULLISH_KEYWORDS))
df_raw["kw_bearish_count"]     = df_raw["description"].apply(lambda t: keyword_count(t, BEARISH_KEYWORDS))
df_raw["kw_uncertainty_count"] = df_raw["description"].apply(lambda t: keyword_count(t, UNCERTAINTY_KEYWORDS))
df_raw["kw_net_signal"]        = df_raw["kw_bullish_count"] - df_raw["kw_bearish_count"]
df_raw["kw_total_signals"]     = df_raw["kw_bullish_count"] + df_raw["kw_bearish_count"] + df_raw["kw_uncertainty_count"]

# ── Time-horizon buckets ──────────────────────────────────────────
def time_horizon_bucket(days):
    if days <= 30:    return "immediate"
    elif days <= 90:  return "short_term"
    elif days <= 180: return "medium_term"
    elif days <= 365: return "long_term"
    else:             return "extended"

df_raw["time_horizon"]         = df_raw["days_to_resolution"].apply(time_horizon_bucket)
time_horizon_order             = {"immediate": 0, "short_term": 1, "medium_term": 2, "long_term": 3, "extended": 4}
df_raw["time_horizon_encoded"] = df_raw["time_horizon"].map(time_horizon_order)

# ── Category encodings ────────────────────────────────────────────
categories       = ["politics", "sports", "crypto", "economics", "science"]
category_ordinal = {cat: i for i, cat in enumerate(categories)}
# Normalise any category values not in our list
df_raw["category"] = df_raw["category"].apply(lambda c: c if c in categories else "economics")
df_raw["category_encoded"] = df_raw["category"].map(category_ordinal)
for cat in categories:
    df_raw[f"cat_{cat}"] = (df_raw["category"] == cat).astype(int)

# ── Volume / liquidity proxies ────────────────────────────────────
df_raw["volume"]    = pd.to_numeric(df_raw["volume"],    errors="coerce").fillna(0)
df_raw["liquidity"] = pd.to_numeric(df_raw["liquidity"], errors="coerce").fillna(0)

df_raw["liquidity_ratio"]  = (df_raw["liquidity"] / (df_raw["volume"] + 1e-9)).round(4)
df_raw["log_volume"]       = np.log1p(df_raw["volume"]).round(4)
df_raw["log_liquidity"]    = np.log1p(df_raw["liquidity"]).round(4)

_cat_vol_mean = df_raw.groupby("category")["volume"].transform("mean")
_cat_vol_std  = df_raw.groupby("category")["volume"].transform("std").replace(0, 1)
df_raw["volume_zscore"]    = ((df_raw["volume"] - _cat_vol_mean) / _cat_vol_std).round(4)
_cat_liq_mean = df_raw.groupby("category")["liquidity"].transform("mean")
_cat_liq_std  = df_raw.groupby("category")["liquidity"].transform("std").replace(0, 1)
df_raw["liquidity_zscore"] = ((df_raw["liquidity"] - _cat_liq_mean) / _cat_liq_std).round(4)
df_raw["market_attention_index"] = (0.5 * df_raw["log_volume"] + 0.5 * df_raw["log_liquidity"]).round(4)

# ── Category base rates ───────────────────────────────────────────
category_base_rates = {
    "politics": 0.48, "sports": 0.38, "crypto": 0.45,
    "economics": 0.42, "science": 0.51,
}
df_raw["category_base_rate"] = df_raw["category"].map(category_base_rates)
df_raw["prob_vs_base_rate"]  = (df_raw["market_probability"] - df_raw["category_base_rate"]).round(4)
df_raw["calibration_score"]  = (1 - np.abs(df_raw["prob_vs_base_rate"])).round(4)

# ── Derived features ──────────────────────────────────────────────
epsilon = 1e-9
p = df_raw["market_probability"]
df_raw["probability_entropy"] = -(
    p * np.log2(p + epsilon) + (1 - p) * np.log2(1 - p + epsilon)).round(4)
df_raw["resolution_urgency"]  = (1 / (df_raw["days_to_resolution"] + 1)).round(6)
df_raw["combined_signal_score"] = (
    0.3 * df_raw["sentiment_polarity"].clip(-1, 1) +
    0.3 * (df_raw["kw_net_signal"] / (df_raw["kw_total_signals"] + 1)).clip(-1, 1) +
    0.4 * df_raw["prob_vs_base_rate"].clip(-1, 1)
).round(4)

# ── Final validation ──────────────────────────────────────────────
enriched_events = df_raw.copy()

critical_cols = [
    "event_id", "title", "category", "market_probability",
    "days_to_resolution", "sentiment_polarity", "sentiment_subjectivity",
    "kw_bullish_count", "kw_bearish_count", "kw_net_signal",
    "time_horizon", "time_horizon_encoded", "category_encoded",
    "liquidity_ratio", "log_volume", "volume_zscore",
    "category_base_rate", "prob_vs_base_rate", "probability_entropy",
    "combined_signal_score"
]
null_counts = enriched_events[critical_cols].isnull().sum()
null_cols   = null_counts[null_counts > 0]
sample_cols = [
    "event_id", "category", "market_probability", "sentiment_polarity",
    "kw_net_signal", "time_horizon", "volume_zscore",
    "prob_vs_base_rate", "probability_entropy", "combined_signal_score"
]

# ══════════════════════════════════════════════════════════════════
# 5.  BANNER OUTPUT
# ══════════════════════════════════════════════════════════════════
_n = len(enriched_events)
cat_counts   = enriched_events["category"].value_counts().reindex(categories, fill_value=0)
cat_avg_prob = enriched_events.groupby("category")["market_probability"].mean().reindex(categories)
_avg_prob_all = enriched_events["market_probability"].mean()
_source_tag   = "🟢 LIVE — Polymarket CLOB API" if data_source_label == "live" else "🟡 FALLBACK — Curated sample dataset"

print("╔══════════════════════════════════════════════════════════╗")
print("║   [ MARKET SIGNAL AUDITOR ] — Data Ingestion Complete   ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
print(f"  Data source  : {_source_tag}")
if _live_failed:
    print(f"  (API error)  : {_fail_reason[:80]}")
print(f"  Events fetched: {_n}")
print(f"  Avg mkt prob  : {_avg_prob_all:.3f}")
print()
_cat_line = " | ".join(f"{c.capitalize()}({int(cat_counts[c])})" for c in categories)
print(f"  Category breakdown: {_cat_line}")
print()
print("  36 signals engineered — Sentiment (8) | Market (10) | Temporal (6) | Category (7) | Calibration (5)")
print()

# ══════════════════════════════════════════════════════════════════
# 6.  FIGURE (identical layout to original)
# ══════════════════════════════════════════════════════════════════
_cat_colors = [BLUE, ORANGE, GREEN, CORAL, LAVENDER]
_cat_labels = [c.capitalize() for c in categories]

fig_ingestion, (_ax_left, _ax_right) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)
fig_ingestion.patch.set_facecolor(BG)

_ax_left.set_facecolor(BG)
_ax_left.barh(_cat_labels, cat_counts.values, color=_cat_colors, edgecolor="none", height=0.6)
for _i, (_count, _prob) in enumerate(zip(cat_counts.values, cat_avg_prob.values)):
    if _count > 0:
        _ax_left.text(
            min(_count * 0.5, _count - 0.05), _i,
            f"avg p={_prob:.2f}", ha="center", va="center",
            fontsize=9, fontweight="bold", color=BG
        )
    _ax_left.text(_count + 0.15, _i, str(int(_count)), ha="left", va="center",
                  fontsize=11, fontweight="bold", color=TEXT_PRI)

_ax_left.set_xlabel("Number of Events", fontsize=11, labelpad=8, color=TEXT_PRI)
_src_short = "Live API" if data_source_label == "live" else "Fallback Sample"
_ax_left.set_title(f"Events per Category [{_src_short}]\n(avg market probability inside bar)",
                   fontsize=12, fontweight="bold", color=TEXT_PRI, pad=10)
_ax_left.set_xlim(0, max(cat_counts.max(), 1) * 1.4)
_ax_left.tick_params(axis="both", labelsize=10, colors=TEXT_PRI)
_ax_left.spines[["top", "right"]].set_visible(False)
_ax_left.grid(axis="x", alpha=0.12)

_families   = ["Sentiment", "Market", "Temporal", "Category", "Calibration"]
_feat_counts = [8, 10, 6, 7, 5]
_fam_colors  = [BLUE, GREEN, ORANGE, LAVENDER, CORAL]
_x_pos = np.arange(len(_families))
_bars_r = _ax_right.bar(_x_pos, _feat_counts, color=_fam_colors, edgecolor="none", width=0.6)
_ax_right.set_facecolor(BG)
for _bar_r, _fc in zip(_bars_r, _feat_counts):
    _ax_right.text(_bar_r.get_x() + _bar_r.get_width() / 2, _bar_r.get_height() + 0.1,
                   str(_fc), ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT_PRI)
_ax_right.set_xticks(_x_pos)
_ax_right.set_xticklabels(_families, fontsize=10, color=TEXT_PRI)
_ax_right.set_ylabel("Features Engineered", fontsize=11, labelpad=8, color=TEXT_PRI)
_ax_right.set_title("Feature Family Breakdown\n(36 total signals)",
                    fontsize=12, fontweight="bold", color=TEXT_PRI, pad=10)
_ax_right.set_ylim(0, max(_feat_counts) * 1.3)
_ax_right.tick_params(axis="both", labelsize=10, colors=TEXT_PRI)
_ax_right.spines[["top", "right"]].set_visible(False)
_ax_right.grid(axis="y", alpha=0.12)

plt.suptitle(f"[ MARKET SIGNAL AUDITOR ] — Data Ingestion Overview  ({_src_short})",
             fontsize=13, fontweight="bold", color=GOLD, y=1.01)
plt.tight_layout()
plt.show()

print(f"  enriched_events ready — shape: {enriched_events.shape}")
if len(null_cols):
    print(f"  ⚠  Null critical cols: {null_cols.to_dict()}")
else:
    print("  ✓  No nulls in critical columns")
