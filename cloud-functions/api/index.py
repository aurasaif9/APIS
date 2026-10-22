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
    title="K3 1M EVEN/ODD Prediction API",
    description="JSON-only API: EVEN/ODD parity prediction from last 10 K3 sums.",
    version="7.0.0",
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
# LEVEL 1 — EVEN/ODD scoring engine (15 rules)
# ==========================================================
def level1_parity(sums):
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
            evidence.append(f"streak_cont_{newest.lower()}_{streak}")
        else:
            other = "ODD" if newest == "EVEN" else "EVEN"
            scores[other] += streak * 1.2
            evidence.append(f"streak_break_{newest.lower()}_{streak}")

    # 3. Last 4 parity trend
    l4 = parities[:4]
    if l4.count("EVEN") >= 3:
        scores["EVEN"] += 2.0
        evidence.append("l4_even")
    elif l4.count("ODD") >= 3:
        scores["ODD"] += 2.0
        evidence.append("l4_odd")

    # 4. Last 3 parity trend
    l3 = parities[:3]
    if l3.count("EVEN") >= 3:
        scores["EVEN"] += 1.5
        evidence.append("l3_even")
    elif l3.count("ODD") >= 3:
        scores["ODD"] += 1.5
        evidence.append("l3_odd")

    # 5. Parity alternation in last 6
    l6 = list(reversed(parities[:6]))
    sw = sum(l6[i] != l6[i-1] for i in range(1, len(l6)))
    if sw >= 4:
        other = "ODD" if newest == "EVEN" else "EVEN"
        scores[other] += 2.0
        evidence.append(f"zigzag_{sw}")

    # 6. Parity dominance reversal
    ec = parities.count("EVEN")
    oc = parities.count("ODD")
    if ec >= 7:
        scores["ODD"] += 2.2
        evidence.append("dom_even_reversal")
    elif oc >= 7:
        scores["EVEN"] += 2.2
        evidence.append("dom_odd_reversal")

    # 7. Parity position bias (odd/even index)
    op = [parities[i] for i in range(0, n, 2)]
    ep = [parities[i] for i in range(1, n, 2)]
    if op.count("EVEN") >= 4 and ep.count("EVEN") >= 4:
        scores["EVEN"] += 1.0
        evidence.append("pos_even_bias")
    elif op.count("ODD") >= 4 and ep.count("ODD") >= 4:
        scores["ODD"] += 1.0
        evidence.append("pos_odd_bias")

    # 8. Sum magnitude + parity correlation
    if len(sums) >= 3:
        big_odd = sum(1 for s in sums[:3] if s >= 13 and s % 2 == 1)
        small_even = sum(1 for s in sums[:3] if s <= 8 and s % 2 == 0)
        if big_odd >= 2:
            scores["ODD"] += 1.0
            evidence.append("big_odd_corr")
        elif small_even >= 2:
            scores["EVEN"] += 1.0
            evidence.append("small_even_corr")

    # 9. Parity pattern repetition
    pstr = "".join("E" if p == "EVEN" else "O" for p in parities)
    for length in (2, 3):
        if len(pstr) >= length * 2:
            sub = pstr[:length]
            prev = pstr.find(sub, length)
            if prev != -1 and prev + length < len(pstr):
                nxt = pstr[prev + length]
                expected = "EVEN" if nxt == "E" else "ODD"
                scores[expected] += 1.8
                evidence.append(f"pattern_repeat_{length}")
                break

    # 10. Anti-streak (4+ parity streak → break)
    if streak >= 4:
        other = "ODD" if newest == "EVEN" else "EVEN"
        scores[other] += 2.5
        evidence.append(f"anti_streak_{streak}")

    # 11. Triple parity pattern
    if n >= 3 and parities[0] == parities[1] == parities[2]:
        other = "ODD" if parities[0] == "EVEN" else "EVEN"
        scores[other] += 1.5
        evidence.append("triple_break")

    # 12. Quad parity pattern
    if n >= 4:
        l4p = parities[:4]
        if l4p[0] == l4p[1] and l4p[2] != l4p[1] and l4p[3] == l4p[2]:
            scores[l4p[0]] += 1.5
            evidence.append("quad_pattern")

    # 13. Momentum shift (last 3 vs previous 3)
    if n >= 6:
        new3_even = parities[:3].count("EVEN")
        old3_even = parities[3:6].count("EVEN")
        if new3_even > old3_even + 1:
            scores["EVEN"] += 1.0
            evidence.append("momentum_even")
        elif new3_even < old3_even - 1:
            scores["ODD"] += 1.0
            evidence.append("momentum_odd")

    # 14. 5-parity hard break
    if n >= 5 and len(set(parities[:5])) == 1:
        other = "ODD" if parities[0] == "EVEN" else "EVEN"
        scores[other] += 3.0
        evidence.append("five_streak_hard_break")

    # 15. Perfect parity alternation (6 switches in 7)
    if n >= 7:
        l7 = list(reversed(parities[:7]))
        sw7 = sum(l7[i] != l7[i-1] for i in range(1, len(l7)))
        if sw7 == 6:
            other = "ODD" if newest == "EVEN" else "EVEN"
            scores[other] += 2.0
            evidence.append("perfect_zigzag_6")

    # Decide Level 1
    if scores["EVEN"] > scores["ODD"]:
        pred = "EVEN"
    elif scores["ODD"] > scores["EVEN"]:
        pred = "ODD"
    else:
        pred = "ODD" if newest == "EVEN" else "EVEN"
        evidence.append("tie_break")

    total = scores["EVEN"] + scores["ODD"]
    gap = abs(scores["EVEN"] - scores["ODD"])
    strength = round(min(95.0, 50.0 + (gap / max(total, 1.0)) * 45.0), 1)

    return {
        "prediction": pred,
        "strength": strength,
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "evidence": evidence,
        "streak": {"parity": newest, "length": streak},
    }

# ==========================================================
# LEVEL 2 — Override rules (10 rules)
# ==========================================================
def level2_parity(parities, sums, l1):
    l1_pred = l1["prediction"]
    l1_str = l1["strength"]
    n = len(parities)
    newest = parities[0]
    override = None
    reason = None

    # A) Quad parity pattern BBSS/SSBB style (EEOO / OOEE)
    if n >= 4:
        l4 = parities[:4]
        if l4[0] == l4[1] and l4[2] != l4[1] and l4[3] == l4[2]:
            expected = l4[0]
            if l1_pred != expected:
                override = expected
                reason = "quad_pattern"

    # B) Triple parity break
    if not override and n >= 3 and parities[0] == parities[1] == parities[2]:
        expected = "ODD" if parities[0] == "EVEN" else "EVEN"
        if l1_pred != expected:
            override = expected
            reason = f"triple_break_{expected.lower()}"

    # C) Strong zigzag (5+ switches in 6)
    if not override:
        l6 = list(reversed(parities[:6]))
        sw = sum(l6[i] != l6[i-1] for i in range(1, len(l6)))
        if sw >= 5:
            expected = "ODD" if newest == "EVEN" else "EVEN"
            if l1_pred != expected:
                override = expected
                reason = "strong_zigzag"

    # D) Pattern repetition
    if not override:
        pstr = "".join("E" if p == "EVEN" else "O" for p in parities)
        for length in (2, 3):
            if len(pstr) >= length * 2:
                sub = pstr[:length]
                prev = pstr.find(sub, length)
                if prev != -1 and prev + length < len(pstr):
                    nxt = pstr[prev + length]
                    expected = "EVEN" if nxt == "E" else "ODD"
                    if l1_pred != expected:
                        override = expected
                        reason = f"pattern_repeat_{length}"
                        break

    # E) Low-confidence momentum fallback
    if not override and l1_str < 60:
        l3 = parities[:3]
        if l3.count("EVEN") >= 2:
            if l1_pred != "EVEN":
                override = "EVEN"
                reason = "low_conf_momentum_even"
        elif l3.count("ODD") >= 2:
            if l1_pred != "ODD":
                override = "ODD"
                reason = "low_conf_momentum_odd"

    # F) Anti-streak (4+)
    if not override:
        streak = 0
        for p in parities:
            if p != newest:
                break
            streak += 1
        if streak >= 4:
            expected = "ODD" if newest == "EVEN" else "EVEN"
            if l1_pred != expected:
                override = expected
                reason = f"anti_streak_{streak}"

    # G) Dominance shift
    if not override:
        ec, oc = parities.count("EVEN"), parities.count("ODD")
        l3 = parities[:3]
        if ec >= 7 and l3.count("ODD") >= 2 and l1_pred != "ODD":
            override = "ODD"
            reason = "dom_shift_odd"
        elif oc >= 7 and l3.count("EVEN") >= 2 and l1_pred != "EVEN":
            override = "EVEN"
            reason = "dom_shift_even"

    # H) 5-parity hard break
    if not override:
        if n >= 5 and len(set(parities[:5])) == 1:
            expected = "ODD" if newest == "EVEN" else "EVEN"
            if l1_pred != expected:
                override = expected
                reason = "five_hard_break"

    # I) Perfect alternation (6 switches in 7)
    if not override:
        l7 = list(reversed(parities[:7]))
        sw7 = sum(l7[i] != l7[i-1] for i in range(1, len(l7)))
        if sw7 == 6:
            expected = "ODD" if newest == "EVEN" else "EVEN"
            if l1_pred != expected:
                override = expected
                reason = "perfect_zigzag"

    # J) Sum-parity correlation override
    if not override and len(sums) >= 3:
        last_sums = sums[:3]
        if all(s >= 13 and s % 2 == 1 for s in last_sums[:2]):
            if l1_pred != "EVEN":
                override = "EVEN"
                reason = "big_odd_exhaustion"
        elif all(s <= 8 and s % 2 == 0 for s in last_sums[:2]):
            if l1_pred != "ODD":
                override = "ODD"
                reason = "small_even_exhaustion"

    if override:
        return {
            "prediction": override,
            "strength": round(min(95.0, l1_str + 10.0), 1),
            "source": "level2_override",
            "override_reason": reason,
        }
    return {
        "prediction": l1_pred,
        "strength": l1_str,
        "source": "level1",
        "override_reason": None,
    }

def analyze(rows):
    recent = rows[:10]
    sums = [r["sum"] for r in recent]
    parities = [r["parity"] for r in recent]

    l1 = level1_parity(sums)
    l2 = level2_parity(parities, sums, l1)

    return {
        "prediction": l2["prediction"],
        "strength": l2["strength"],
        "source": l2["source"],
        "override_reason": l2["override_reason"],
        "level1_prediction": l1["prediction"],
        "level1_strength": l1["strength"],
        "level1_scores": l1["scores"],
        "evidence": l1["evidence"],
        "streak": l1["streak"],
        "even_count_last10": parities.count("EVEN"),
        "odd_count_last10": parities.count("ODD"),
    }

# ==========================================================
# Endpoints
# ==========================================================
@app.get("/")
async def root():
    return {
        "service": "K3 1M EVEN/ODD Prediction API",
        "status": "ok",
        "version": "7.0.0",
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "k3-1m-even-odd-api", "version": "7.0.0"}

@app.get("/predict")
@app.get("/api/predict")
async def predict():
    rows = await fetch_history()
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
                previous_actual = r["parity"]
                break
        if previous_prediction and previous_actual:
            previous_result = "WIN" if previous_prediction == previous_actual else "LOSS"

    cache["last_prediction"] = {
        "period": period,
        "prediction": result["prediction"],
    }
    _save_cache(cache)

    # EXACT same format as you asked — only EVEN/ODD
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

    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(content=pretty, media_type="application/json; charset=utf-8")

@app.get("/predict/detail")
@app.get("/api/predict/detail")
async def predict_detail():
    rows = await fetch_history()
    period = next_issue(rows[0]["issueNumber"])
    result = analyze(rows)

    payload = {
        "success": True,
        "period": period,
        "prediction": result["prediction"],
        "strength": result["strength"],
        "source": result["source"],
        "override_reason": result["override_reason"],
        "level1_prediction": result["level1_prediction"],
        "level1_strength": result["level1_strength"],
        "level1_scores": result["level1_scores"],
        "streak": result["streak"],
        "even_count_last10": result["even_count_last10"],
        "odd_count_last10": result["odd_count_last10"],
        "evidence": result["evidence"],
        "history_count_used": 10,
    }
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(content=pretty, media_type="application/json; charset=utf-8")
