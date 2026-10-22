from datetime import datetime, timezone
from typing import Any
import os
import json
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

SOURCE_URL = os.getenv(
    "K3_HISTORY_URL",
    "https://draw.ar-lottery01.com/K3/K3_1M/GetHistoryIssuePage.json?pageNo=1&pageSize=100"
)
MARKET_OFFSET_SECONDS = int(os.getenv("MARKET_OFFSET_SECONDS", "0")) % 60
REQUEST_TIMEOUT = float(os.getenv("K3_REQUEST_TIMEOUT", "8"))

app = FastAPI(
    title="K3 1M JSON Prediction API",
    description="JSON-only API: even/odd + BIG/SMALL 2-level prediction.",
    version="6.0.0",
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
            "side": "SMALL" if total <= 10 else "BIG",
            "parity": "EVEN" if total % 2 == 0 else "ODD",
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
# Big/Small scoring (Level 1)
# ==========================================================
def level1_bigsmall(sides, sums):
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

    odd_pos = [sides[i] for i in range(0, n, 2)]
    even_pos = [sides[i] for i in range(1, n, 2)]
    if odd_pos.count("BIG") >= 4 and even_pos.count("BIG") >= 4:
        scores["BIG"] += 1.0
    elif odd_pos.count("SMALL") >= 4 and even_pos.count("SMALL") >= 4:
        scores["SMALL"] += 1.0

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
# Level 2 big/small override
# ==========================================================
def level2_bigsmall(sides, sums, l1):
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

    if override and override != l1_pred:
        return {"prediction": override, "source": "level2_override"}
    return {"prediction": l1_pred, "source": "level1"}

# ==========================================================
# EVEN / ODD logic (dedicated parity engine)
# ==========================================================
def parity_prediction(sums):
    """
    Analyze last 10 sums' parity (EVEN/ODD) and predict next parity.
    Uses recency weighting, streaks, alternation, dominance reversal.
    """
    parities = ["EVEN" if s % 2 == 0 else "ODD" for s in sums[:10]]
    n = len(parities)
    newest = parities[0]
    scores = {"EVEN": 0.0, "ODD": 0.0}
    evidence = []

    # 1. Recency weighted (newest heaviest)
    for age, p in enumerate(parities):
        scores[p] += max(1, 10 - age) * 1.0

    # 2. Current parity streak
    streak = 0
    for p in parities:
        if p != newest:
            break
        streak += 1
    if streak >= 2:
        if streak <= 2:
            scores[newest] += streak * 0.7
            evidence.append(f"parity_streak_cont_{newest.lower()}_{streak}")
        else:
            other = "ODD" if newest == "EVEN" else "EVEN"
            scores[other] += streak * 1.2
            evidence.append(f"parity_streak_break_{newest.lower()}_{streak}")

    # 3. Last 4 parity trend
    l4 = parities[:4]
    if l4.count("EVEN") >= 3:
        scores["EVEN"] += 2.0
        evidence.append("l4_even")
    elif l4.count("ODD") >= 3:
        scores["ODD"] += 2.0
        evidence.append("l4_odd")

    # 4. Parity alternation in last 6
    l6 = list(reversed(parities[:6]))
    sw = sum(l6[i] != l6[i-1] for i in range(1, len(l6)))
    if sw >= 4:
        other = "ODD" if newest == "EVEN" else "EVEN"
        scores[other] += 2.0
        evidence.append(f"parity_zigzag_{sw}")

    # 5. Parity dominance reversal
    ec = parities.count("EVEN")
    oc = parities.count("ODD")
    if ec >= 7:
        scores["ODD"] += 2.0
        evidence.append("parity_dom_even_reversal")
    elif oc >= 7:
        scores["EVEN"] += 2.0
        evidence.append("parity_dom_odd_reversal")

    # 6. Parity position bias (odd/even index)
    op = [parities[i] for i in range(0, n, 2)]
    ep = [parities[i] for i in range(1, n, 2)]
    if op.count("EVEN") >= 4 and ep.count("EVEN") >= 4:
        scores["EVEN"] += 0.8
    elif op.count("ODD") >= 4 and ep.count("ODD") >= 4:
        scores["ODD"] += 0.8

    # 7. Sum magnitude + parity correlation
    # Recent big sums (13+) tend ODD more; small sums (<=8) tend EVEN more
    if len(sums) >= 3:
        last_sums = sums[:3]
        big_odd = sum(1 for s in last_sums if s >= 13 and s % 2 == 1)
        small_even = sum(1 for s in last_sums if s <= 8 and s % 2 == 0)
        if big_odd >= 2:
            scores["ODD"] += 0.8
            evidence.append("big_odd_correlation")
        elif small_even >= 2:
            scores["EVEN"] += 0.8
            evidence.append("small_even_correlation")

    # 8. Parity pattern repetition
    pstr = "".join("E" if p == "EVEN" else "O" for p in parities)
    for length in (2, 3):
        if len(pstr) >= length * 2:
            sub = pstr[:length]
            prev = pstr.find(sub, length)
            if prev != -1 and prev + length < len(pstr):
                nxt = pstr[prev + length]
                expected = "EVEN" if nxt == "E" else "ODD"
                scores[expected] += 1.5
                evidence.append(f"parity_repeat_{length}")
                break

    # 9. Anti-streak (4+ parity streak)
    if streak >= 4:
        other = "ODD" if newest == "EVEN" else "EVEN"
        scores[other] += 2.5
        evidence.append(f"parity_anti_streak_{streak}")

    # Decide
    if scores["EVEN"] > scores["ODD"]:
        prediction = "EVEN"
    elif scores["ODD"] > scores["EVEN"]:
        prediction = "ODD"
    else:
        prediction = "ODD" if newest == "EVEN" else "EVEN"
        evidence.append("parity_tie_break")

    total = scores["EVEN"] + scores["ODD"]
    gap = abs(scores["EVEN"] - scores["ODD"])
    strength = round(min(95.0, 50.0 + (gap / max(total, 1.0)) * 45.0), 1)

    return {
        "parity_prediction": prediction,
        "parity_strength": strength,
        "parity_scores": {k: round(v, 2) for k, v in scores.items()},
        "parity_evidence": evidence,
        "parity_streak": {"parity": newest, "length": streak},
        "even_count_last10": ec,
        "odd_count_last10": oc,
    }

def analyze(rows):
    recent = rows[:10]
    sides = [r["side"] for r in recent]
    sums = [r["sum"] for r in recent]

    l1 = level1_bigsmall(sides, sums)
    l2 = level2_bigsmall(sides, sums, l1)
    parity = parity_prediction(sums)

    return {
        "prediction": l2["prediction"],
        "source": l2["source"],
        "parity": parity,
    }

# ==========================================================
# Endpoints
# ==========================================================
@app.get("/")
async def root():
    return {
        "service": "K3 1M JSON Prediction API",
        "status": "ok",
        "version": "6.0.0",
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "k3-1m-json-prediction-api", "version": "6.0.0"}

@app.get("/predict")
@app.get("/api/predict")
async def predict():
    rows = await fetch_history()
    period = next_issue(rows[0]["issueNumber"])
    result = analyze(rows)
    parity = result["parity"]

    cache = _load_cache()
    prev = cache.get("last_prediction", {})

    previous_period = prev.get("period")
    previous_prediction = prev.get("prediction")
    previous_parity_prediction = prev.get("parity_prediction")
    previous_actual = None
    previous_actual_parity = None
    previous_result = None
    previous_parity_result = None

    if previous_period:
        for r in rows:
            if r["issueNumber"] == previous_period:
                previous_actual = r["side"]
                previous_actual_parity = r["parity"]
                break
        if previous_prediction and previous_actual:
            previous_result = "WIN" if previous_prediction == previous_actual else "LOSS"
        if previous_parity_prediction and previous_actual_parity:
            previous_parity_result = "WIN" if previous_parity_prediction == previous_actual_parity else "LOSS"

    cache["last_prediction"] = {
        "period": period,
        "prediction": result["prediction"],
        "parity_prediction": parity["parity_prediction"],
    }
    _save_cache(cache)

    payload = {
        "success": True,
        "period": period,
        "prediction": result["prediction"],
        "parity_prediction": parity["parity_prediction"],
        "previous_period": previous_period,
        "previous_prediction": previous_prediction,
        "previous_actual": previous_actual,
        "previous_result": previous_result,
        "previous_parity_prediction": previous_parity_prediction,
        "previous_actual_parity": previous_actual_parity,
        "previous_parity_result": previous_parity_result,
        "history_count_used": 10,
    }

    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(content=pretty, media_type="application/json; charset=utf-8")

@app.get("/predict/detail")
@app.get("/api/predict/detail")
async def predict_detail():
    """Detailed view with all parity analysis + evidence."""
    rows = await fetch_history()
    period = next_issue(rows[0]["issueNumber"])
    result = analyze(rows)
    parity = result["parity"]

    payload = {
        "success": True,
        "period": period,
        "prediction": result["prediction"],
        "source": result["source"],
        "parity_prediction": parity["parity_prediction"],
        "parity_strength": parity["parity_strength"],
        "parity_scores": parity["parity_scores"],
        "parity_streak": parity["parity_streak"],
        "even_count_last10": parity["even_count_last10"],
        "odd_count_last10": parity["odd_count_last10"],
        "parity_evidence": parity["parity_evidence"],
        "history_count_used": 10,
    }
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(content=pretty, media_type="application/json; charset=utf-8")
