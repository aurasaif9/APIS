from datetime import datetime, timezone
from typing import Any
import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

SOURCE_URL = os.getenv(
    "K3_HISTORY_URL",
    "https://draw.ar-lottery01.com/K3/K3_1M/GetHistoryIssuePage.json?pageNo=1&pageSize=100"
)
MARKET_OFFSET_SECONDS = int(os.getenv("MARKET_OFFSET_SECONDS", "0")) % 60
REQUEST_TIMEOUT = float(os.getenv("K3_REQUEST_TIMEOUT", "8"))

app = FastAPI(
    title="K3 1M JSON Prediction API",
    description="JSON-only API: 2-level pattern prediction.",
    version="5.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CACHE_FILE = os.getenv("K3_CACHE_FILE", "/tmp/k3_pred_cache.json")

def _load_cache():
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(c):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(c, f)
    except Exception:
        pass

def seconds_remaining(now=None):
    now = now or datetime.now(timezone.utc)
    elapsed = (now.second - MARKET_OFFSET_SECONDS) % 60
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
        raise HTTPException(status_code=502, detail=f"Could not connect: {type(exc).__name__}")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Source HTTP {response.status_code}")

    try:
        payload = response.json()
    except ValueError:
        raise HTTPException(status_code=502, detail="Invalid JSON from source.")

    rows = payload.get("data", {}).get("list") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise HTTPException(status_code=502, detail="Unexpected response format.")

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
        valid.append({
            "issueNumber": issue,
            "sum": total,
            "side": "SMALL" if total <= 10 else "BIG"
        })
    if len(valid) < 10:
        raise HTTPException(status_code=502, detail=f"Only {len(valid)} valid rows; need 10+.")

    try:
        valid.sort(key=lambda r: int(r["issueNumber"]), reverse=True)
    except ValueError:
        pass
    return valid[:100]

def next_issue(issue):
    try:
        return str(int(issue) + 1).zfill(len(issue))
    except (TypeError, ValueError):
        raise HTTPException(status_code=502, detail=f"Cannot increment: {issue!r}")

# ==========================================================
# LEVEL 1 — Core scoring engine
# ==========================================================
def level1_score(sides, sums):
    scores = {"BIG": 0.0, "SMALL": 0.0}
    n = len(sides)
    newest = sides[0]

    for age, side in enumerate(sides):
        scores[side] += max(1, 10 - age) * 0.9

    streak = 0
    for s in sides:
        if s != newest:
            break
        streak += 1
    if streak >= 2:
        if streak <= 3:
            scores[newest] += streak * 0.6
        else:
            other = "SMALL" if newest == "BIG" else "BIG"
            scores[other] += streak * 1.0

    l5 = sides[:5]
    if l5.count("BIG") >= 4:
        scores["BIG"] += 2.5
    elif l5.count("SMALL") >= 4:
        scores["SMALL"] += 2.5

    l3 = sides[:3]
    if l3.count("BIG") >= 3:
        scores["BIG"] += 2.0
    elif l3.count("SMALL") >= 3:
        scores["SMALL"] += 2.0

    l6 = list(reversed(sides[:6]))
    sw = sum(l6[i] != l6[i-1] for i in range(1, len(l6)))
    if sw >= 4:
        other = "SMALL" if newest == "BIG" else "BIG"
        scores[other] += 2.2

    bc, sc = sides.count("BIG"), sides.count("SMALL")
    if bc >= 7:
        scores["SMALL"] += 2.0
    elif sc >= 7:
        scores["BIG"] += 2.0

    odd = [sides[i] for i in range(0, n, 2)]
    even = [sides[i] for i in range(1, n, 2)]
    if odd.count("BIG") >= 4 and even.count("BIG") >= 4:
        scores["BIG"] += 1.0
    elif odd.count("SMALL") >= 4 and even.count("SMALL") >= 4:
        scores["SMALL"] += 1.0

    doubles = sum(1 for i in range(0, n-1, 2) if sides[i] == sides[i+1])
    if doubles >= 3 and n >= 2 and sides[0] == sides[1]:
        other = "SMALL" if sides[0] == "BIG" else "BIG"
        scores[other] += 0.8

    triples = sum(1 for i in range(0, n-2, 3) if sides[i] == sides[i+1] == sides[i+2])
    if triples >= 1:
        other = "SMALL" if sides[0] == "BIG" else "BIG"
        scores[other] += 1.2

    if len(sums) >= 3:
        if sums[0] > sums[1] > sums[2]:
            scores["SMALL"] += 0.8
        elif sums[0] < sums[1] < sums[2]:
            scores["BIG"] += 0.8

    if n >= 6:
        new3_big = sides[:3].count("BIG")
        old3_big = sides[3:6].count("BIG")
        if new3_big > old3_big + 1:
            scores["BIG"] += 1.0
        elif new3_big < old3_big - 1:
            scores["SMALL"] += 1.0

    if len(sums) >= 4:
        ex_small = sum(1 for s in sums[:4] if s <= 5)
        ex_big = sum(1 for s in sums[:4] if s >= 16)
        if ex_small >= 2:
            scores["BIG"] += 1.0
        elif ex_big >= 2:
            scores["SMALL"] += 1.0

    if bc >= 8:
        scores["SMALL"] += 1.5
    elif sc >= 8:
        scores["BIG"] += 1.5

    l8 = list(reversed(sides[:8]))
    sw8 = sum(l8[i] != l8[i-1] for i in range(1, len(l8)))
    if sw8 >= 6:
        other = "SMALL" if newest == "BIG" else "BIG"
        scores[other] += 1.5

    if scores["BIG"] > scores["SMALL"]:
        pred = "BIG"
    elif scores["SMALL"] > scores["BIG"]:
        pred = "SMALL"
    else:
        pred = "SMALL" if newest == "BIG" else "BIG"

    total = scores["BIG"] + scores["SMALL"]
    gap = abs(scores["BIG"] - scores["SMALL"])
    strength = round(min(95.0, 50.0 + (gap / max(total, 1.0)) * 45.0), 1)

    return {"prediction": pred, "strength": strength}

# ==========================================================
# LEVEL 2 — Override rules
# ==========================================================
def level2_override(sides, sums, l1):
    l1_pred = l1["prediction"]
    l1_str = l1["strength"]
    n = len(sides)
    newest = sides[0]
    override = None

    if n >= 4:
        l4 = sides[:4]
        if l4[0] == l4[1] and l4[2] != l4[1] and l4[3] == l4[2]:
            override = l4[0]

    if not override and n >= 3 and sides[0] == sides[1] == sides[2]:
        override = "SMALL" if sides[0] == "BIG" else "BIG"

    if not override:
        l6 = list(reversed(sides[:6]))
        sw = sum(l6[i] != l6[i-1] for i in range(1, len(l6)))
        if sw >= 5:
            override = "SMALL" if newest == "BIG" else "BIG"

    if not override:
        pstr = "".join("B" if s == "BIG" else "S" for s in sides)
        for length in (2, 3):
            if len(pstr) >= length * 2:
                sub = pstr[:length]
                prev = pstr.find(sub, length)
                if prev != -1 and prev + length < len(pstr):
                    nxt = pstr[prev + length]
                    override = "BIG" if nxt == "B" else "SMALL"
                    break

    if not override and l1_str < 60:
        l3 = sides[:3]
        if l3.count("BIG") >= 2:
            override = "BIG"
        elif l3.count("SMALL") >= 2:
            override = "SMALL"

    if not override:
        streak = 0
        for s in sides:
            if s != newest:
                break
            streak += 1
        if streak >= 4:
            override = "SMALL" if newest == "BIG" else "BIG"

    if not override:
        bc, sc = sides.count("BIG"), sides.count("SMALL")
        l3 = sides[:3]
        if bc >= 7 and l3.count("SMALL") >= 2:
            override = "SMALL"
        elif sc >= 7 and l3.count("BIG") >= 2:
            override = "BIG"

    if not override and len(sums) >= 4:
        ex_small = sum(1 for s in sums[:4] if s <= 5)
        ex_big = sum(1 for s in sums[:4] if s >= 16)
        if ex_small >= 2:
            override = "BIG"
        elif ex_big >= 2:
            override = "SMALL"

    if not override:
        if n >= 5 and len(set(sides[:5])) == 1:
            override = "SMALL" if newest == "BIG" else "BIG"

    if not override:
        l7 = list(reversed(sides[:7]))
        sw7 = sum(l7[i] != l7[i-1] for i in range(1, len(l7)))
        if sw7 == 6:
            override = "SMALL" if newest == "BIG" else "BIG"

    if not override and len(sums) >= 3:
        if sums[0] > sums[1] > sums[2] and sums[0] >= 13:
            override = "SMALL"
        elif sums[0] < sums[1] < sums[2] and sums[0] <= 8:
            override = "BIG"

    if not override and n >= 6:
        p = sides[:6]
        if p[0] == p[2] == p[4] and p[1] == p[3] == p[5] and p[0] != p[1]:
            override = p[0]

    if override and override != l1_pred:
        return {
            "prediction": override,
            "strength": round(min(95.0, l1_str + 10.0), 1),
            "source": "level2_override",
        }
    return {
        "prediction": l1_pred,
        "strength": l1_str,
        "source": "level1",
    }

def analyze(rows):
    recent = rows[:10]
    sides = [r["side"] for r in recent]
    sums = [r["sum"] for r in recent]
    l1 = level1_score(sides, sums)
    l2 = level2_override(sides, sums, l1)
    return {
        "prediction": l2["prediction"],
        "source": l2["source"],
    }

# ==========================================================
# Endpoints
# ==========================================================
@app.get("/")
async def root():
    return {
        "service": "K3 1M JSON Prediction API",
        "status": "ok",
        "version": "5.0.0",
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "k3-1m-json-prediction-api", "version": "5.0.0"}

@app.get("/predict")
@app.get("/api/predict")
async def predict():
    rows = await fetch_history()
    # period = current issue + 1 (prediction target)
    period = next_issue(rows[0]["issueNumber"])
    result = analyze(rows)

    cache = _load_cache()
    prev = cache.get("last_prediction", {})

    previous_period = prev.get("period")
    previous_prediction = prev.get("prediction")
    previous_actual = None
    previous_result = None

    if previous_period:
        for r in rows:
            if r["issueNumber"] == previous_period:
                previous_actual = r["side"]
                break
        if previous_prediction and previous_actual:
            previous_result = "WIN" if previous_prediction == previous_actual else "LOSS"

    cache["last_prediction"] = {
        "period": period,
        "prediction": result["prediction"],
    }
    _save_cache(cache)

    payload = {
        "success": True,
        "period": period,
        "prediction": result["prediction"],
        "previous_period": previous_period,
        "previous_prediction": previous_prediction,
        "previous_actual": previous_actual,
        "previous_result": previous_result,
        "history_count_used": 10,
    }

    # Pretty print with indent=2
    return JSONResponse(content=payload, media_type="application/json; charset=utf-8", headers={
        "X-Pretty": "true"
    })
