from datetime import datetime, timezone
from typing import Any
import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

SOURCE_URL = os.getenv(
    "K3_HISTORY_URL",
    "https://draw.ar-lottery01.com/K3/K3_1M/GetHistoryIssuePage.json?pageNo=1&pageSize=100"
)
MARKET_OFFSET_SECONDS = int(os.getenv("MARKET_OFFSET_SECONDS", "0")) % 60
REQUEST_TIMEOUT = float(os.getenv("K3_REQUEST_TIMEOUT", "8"))

app = FastAPI(
    title="K3 1M JSON Prediction API",
    description="JSON-only API: analyze latest 10 K3 sums and predict issueNumber + 1.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def seconds_remaining(now=None):
    now = now or datetime.now(timezone.utc)
    seconds_in_minute = now.second
    elapsed = (seconds_in_minute - MARKET_OFFSET_SECONDS) % 60
    return 60 - elapsed if elapsed else 60

async def fetch_history():
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/138.0.0.0 Mobile Safari/537.36",
        "Referer": "https://draw.ar-lottery01.com/",
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(SOURCE_URL, headers=headers)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="K3 source API timed out.")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not connect to K3 source API: {type(exc).__name__}")

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"K3 source API returned HTTP {response.status_code}. The source may block server requests."
        )
    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="K3 source API did not return valid JSON.")

    rows = payload.get("data", {}).get("list") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail="K3 response format changed: expected data.list.")

    valid = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        issue = str(row.get("issueNumber", "")).strip()
        try:
            total = int(row.get("sum"))
        except (TypeError, ValueError):
            continue
        if not issue or not (3 <= total <= 18):
            continue
        valid.append({"issueNumber": issue, "sum": total, "side": "SMALL" if total <= 10 else "BIG"})
    if len(valid) < 10:
        raise HTTPException(status_code=502, detail=f"Only {len(valid)} valid K3 rows; need at least 10.")

    try:
        valid.sort(key=lambda row: int(row["issueNumber"]), reverse=True)
    except ValueError:
        pass
    return valid[:100]

def next_issue(issue):
    try:
        return str(int(issue) + 1).zfill(len(issue))
    except (TypeError, ValueError):
        raise HTTPException(status_code=502, detail=f"Cannot increment non-numeric issueNumber: {issue!r}")

def analyze(rows):
    # rows are newest first
    recent = rows[:10]
    sides = [row["side"] for row in recent]
    scores = {"BIG": 0.0, "SMALL": 0.0}
    evidence = []

    # Recency weighted scoring
    for age, side in enumerate(sides):
        scores[side] += max(1, 6 - age) * 0.7

    # Recent-five trend
    if sides[:5].count("BIG") >= 4:
        scores["BIG"] += 2.0
        evidence.append("recent_5_big_trend")
    elif sides[:5].count("SMALL") >= 4:
        scores["SMALL"] += 2.0
        evidence.append("recent_5_small_trend")

    # Streak at newest end
    newest = sides[0]
    streak = 0
    for side in sides:
        if side != newest:
            break
        streak += 1
    if streak >= 2:
        scores[newest] += min(streak, 4) * 0.6
        evidence.append(f"current_{newest.lower()}_streak_{streak}")

    # Alternation in most recent six, read chronologically
    last6 = list(reversed(sides[:6]))
    switches = sum(last6[i] != last6[i-1] for i in range(1, len(last6)))
    if switches >= 4:
        other = "SMALL" if newest == "BIG" else "BIG"
        scores[other] += 1.2
        evidence.append("zigzag_pattern")

    big_count, small_count = sides.count("BIG"), sides.count("SMALL")
    if big_count >= 7:
        scores["SMALL"] += 0.8
        evidence.append("big_dominance_last10")
    elif small_count >= 7:
        scores["BIG"] += 0.8
        evidence.append("small_dominance_last10")

    if scores["BIG"] > scores["SMALL"]:
        prediction = "BIG"
    elif scores["SMALL"] > scores["BIG"]:
        prediction = "SMALL"
    else:
        weighted = {"BIG": 0, "SMALL": 0}
        for age, side in enumerate(sides):
            weighted[side] += max(1, 6 - age)
        if weighted["BIG"] != weighted["SMALL"]:
            prediction = "BIG" if weighted["BIG"] > weighted["SMALL"] else "SMALL"
        else:
            prediction = "SMALL" if newest == "BIG" else "BIG"
        evidence.append("tie_breaker_used")

    total_score = scores["BIG"] + scores["SMALL"]
    gap = abs(scores["BIG"] - scores["SMALL"])
    strength = round(min(95.0, 50.0 + gap / max(total_score, 1.0) * 45.0), 1)
    return {
        "prediction": prediction,
        "strength": strength,
        "strength_note": "Heuristic score, not a true probability or guarantee.",
        "scores": {key: round(value, 2) for key, value in scores.items()},
        "analyzed_count": len(recent),
        "recent_10": recent,
        "recent_5": recent[:5],
        "big_count_last10": big_count,
        "small_count_last10": small_count,
        "current_streak": {"side": newest, "length": streak},
        "evidence": evidence,
    }

@app.get("/")
async def root():
    # Root is JSON only; no HTML dashboard.
    return {
        "service": "K3 1M JSON Prediction API",
        "status": "ok",
        "endpoints": {
            "health": "/api/health",
            "prediction": "/api/predict",
            "docs": "/api/docs"
        }
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "k3-1m-json-prediction-api"}

@app.get("/predict")
@app.get("/api/predict")
async def predict():
    rows = await fetch_history()
    current_issue = rows[0]["issueNumber"]
    result = analyze(rows)
    now = datetime.now(timezone.utc)
    return {
        "market": "K3_1M",
        "source_status": "ok",
        "server_time_utc": now.isoformat(),
        "timer": {
            "period_seconds": 60,
            "seconds_remaining": seconds_remaining(now),
            "market_offset_seconds": MARKET_OFFSET_SECONDS,
            "sync_note": "Wall-clock minute estimate; set MARKET_OFFSET_SECONDS to match the market boundary."
        },
        "current_issue": current_issue,
        "prediction_issue": next_issue(current_issue),
        **result
    }
