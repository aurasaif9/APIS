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
    description="JSON-only API: analyze latest 10 K3 sums and predict issueNumber + 1 with 2-level logic.",
    version="2.0.0",
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

def detect_patterns(sides):
    """Detect all possible patterns in the last 10 results (newest first)."""
    patterns = {}
    n = len(sides)
    
    # 1. Streak detection (current streak at newest end)
    newest = sides[0]
    streak = 0
    for side in sides:
        if side != newest:
            break
        streak += 1
    patterns["current_streak"] = {"side": newest, "length": streak}
    
    # 2. Recent 5 trend
    last5 = sides[:5]
    big5 = last5.count("BIG")
    small5 = last5.count("SMALL")
    patterns["last5"] = {"BIG": big5, "SMALL": small5}
    
    # 3. Recent 3 trend
    last3 = sides[:3]
    big3 = last3.count("BIG")
    small3 = last3.count("SMALL")
    patterns["last3"] = {"BIG": big3, "SMALL": small3}
    
    # 4. Alternation/zigzag in last 6 (chronological)
    last6_chrono = list(reversed(sides[:6]))
    switches = sum(last6_chrono[i] != last6_chrono[i-1] for i in range(1, len(last6_chrono)))
    patterns["zigzag_6"] = switches >= 4
    patterns["switch_count_6"] = switches
    
    # 5. Double pattern (BB, SS pairs)
    pairs = []
    for i in range(0, min(10, n) - 1, 2):
        if sides[i] == sides[i+1]:
            pairs.append(sides[i])
    patterns["double_pairs"] = pairs
    patterns["double_count"] = len(pairs)
    
    # 6. Triple pattern
    triples = 0
    for i in range(0, min(9, n) - 2, 3):
        if sides[i] == sides[i+1] == sides[i+2]:
            triples += 1
    patterns["triple_count"] = triples
    
    # 7. Dominance in last 10
    big10 = sides.count("BIG")
    small10 = sides.count("SMALL")
    patterns["last10"] = {"BIG": big10, "SMALL": small10}
    
    # 8. Position-based analysis (odd/even positions)
    odd_pos = [sides[i] for i in range(0, n, 2)]  # positions 0,2,4,6,8 (newest side)
    even_pos = [sides[i] for i in range(1, n, 2)]  # positions 1,3,5,7,9
    patterns["odd_positions"] = {"BIG": odd_pos.count("BIG"), "SMALL": odd_pos.count("SMALL")}
    patterns["even_positions"] = {"BIG": even_pos.count("BIG"), "SMALL": even_pos.count("SMALL")}
    
    # 9. Sum-based analysis (actual sum values, not just side)
    # We'll pass sums separately if needed
    
    # 10. Consecutive same-side count before last switch
    # Find last switch position
    last_switch_idx = 0
    for i in range(1, n):
        if sides[i] != sides[i-1]:
            last_switch_idx = i
            break
    patterns["since_last_switch"] = last_switch_idx
    
    # 11. Pattern repetition: check if recent pattern appeared before
    # Look for repeating sub-sequences
    pattern_str = "".join("B" if s == "BIG" else "S" for s in sides[:8])
    repeats = 0
    for length in [2, 3, 4]:
        for start in range(0, len(pattern_str) - length * 2 + 1):
            sub = pattern_str[start:start+length]
            # Count occurrences in the string
            count = pattern_str.count(sub)
            if count >= 2:
                repeats += count - 1
    patterns["pattern_repeats"] = repeats
    
    # 12. Consecutive pattern: BSBS, SBSB, BBSB, etc.
    # Check last 2, 3, 4 pattern
    patterns["last2"] = sides[:2]
    patterns["last3_seq"] = sides[:3]
    patterns["last4_seq"] = sides[:4]
    
    return patterns

def level1_prediction(sides, patterns):
    """Level 1: Primary prediction using weighted scoring."""
    scores = {"BIG": 0.0, "SMALL": 0.0}
    evidence = []
    
    # Recency weighted scoring (newest gets highest weight)
    for age, side in enumerate(sides[:10]):
        weight = max(1, 10 - age) * 0.8
        scores[side] += weight
    
    # Current streak bonus/penalty
    streak_side = patterns["current_streak"]["side"]
    streak_len = patterns["current_streak"]["length"]
    
    if streak_len >= 2:
        # Long streaks tend to break, but short streaks tend to continue
        if streak_len <= 3:
            scores[streak_side] += streak_len * 0.5
            evidence.append(f"streak_continue_{streak_side.lower()}_{streak_len}")
        else:
            # 4+ streak: expect break
            other = "SMALL" if streak_side == "BIG" else "BIG"
            scores[other] += streak_len * 0.8
            evidence.append(f"streak_break_{streak_side.lower()}_{streak_len}")
    
    # Recent 5 trend
    if patterns["last5"]["BIG"] >= 4:
        scores["BIG"] += 2.0
        evidence.append("last5_big_trend")
    elif patterns["last5"]["SMALL"] >= 4:
        scores["SMALL"] += 2.0
        evidence.append("last5_small_trend")
    
    # Recent 3 trend
    if patterns["last3"]["BIG"] >= 3:
        scores["BIG"] += 1.5
        evidence.append("last3_big_trend")
    elif patterns["last3"]["SMALL"] >= 3:
        scores["SMALL"] += 1.5
        evidence.append("last3_small_trend")
    
    # Zigzag pattern
    if patterns["zigzag_6"]:
        other = "SMALL" if sides[0] == "BIG" else "BIG"
        scores[other] += 2.0
        evidence.append("zigzag_6")
    
    # Dominance correction (if 7+ same side in 10, expect reversal)
    if patterns["last10"]["BIG"] >= 7:
        scores["SMALL"] += 1.5
        evidence.append("big_dominance_reversal")
    elif patterns["last10"]["SMALL"] >= 7:
        scores["BIG"] += 1.5
        evidence.append("small_dominance_reversal")
    
    # Double pattern analysis
    if patterns["double_count"] >= 3:
        # Many doubles: pattern may continue with doubles
        evidence.append("double_heavy")
    
    # Position bias
    odd_big = patterns["odd_positions"]["BIG"]
    even_big = patterns["even_positions"]["BIG"]
    if odd_big >= 4 and even_big >= 4:
        scores["BIG"] += 0.5
        evidence.append("position_big_bias")
    elif odd_big <= 1 and even_big <= 1:
        scores["SMALL"] += 0.5
        evidence.append("position_small_bias")
    
    # Determine prediction
    if scores["BIG"] > scores["SMALL"]:
        prediction = "BIG"
    elif scores["SMALL"] > scores["BIG"]:
        prediction = "SMALL"
    else:
        # Tie-breaker: use most recent
        prediction = "SMALL" if sides[0] == "BIG" else "BIG"
        evidence.append("tie_breaker")
    
    total = scores["BIG"] + scores["SMALL"]
    gap = abs(scores["BIG"] - scores["SMALL"])
    strength = min(95.0, 50.0 + (gap / max(total, 1.0)) * 45.0)
    
    return {
        "prediction": prediction,
        "strength": round(strength, 1),
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "evidence": evidence
    }

def level2_prediction(sides, patterns, level1):
    """Level 2: Validation/override layer that catches patterns Level 1 might miss."""
    l1_pred = level1["prediction"]
    l1_strength = level1["strength"]
    
    # Level 2 override logic
    override = None
    override_reason = None
    
    # RULE 1: Strong consecutive pair pattern (BBSB, SSBS, etc.)
    last4 = sides[:4]
    if last4[0] == last4[1] and last4[2] != last4[1] and last4[3] == last4[2]:
        # Pattern like BBSS or SSBB -> next tends to be the first type
        expected = last4[0]
        if l1_pred != expected:
            override = expected
            override_reason = f"quad_pattern_{expected.lower()}"
    
    # RULE 2: Three consecutive same then switch pattern
    if patterns["triple_count"] >= 1 and patterns["since_last_switch"] >= 3:
        # After triple, expect opposite of triple
        triple_side = sides[0]
        expected = "SMALL" if triple_side == "BIG" else "BIG"
        if l1_pred != expected:
            override = expected
            override_reason = f"triple_break_{expected.lower()}"
    
    # RULE 3: Strong zigzag (5+ switches in last 6) - very reliable
    if patterns["switch_count_6"] >= 5:
        expected = "SMALL" if sides[0] == "BIG" else "BIG"
        if l1_pred != expected:
            override = expected
            override_reason = "strong_zigzag"
    
    # RULE 4: Pattern repetition - if we see a repeating pattern, follow it
    if patterns["pattern_repeats"] >= 3:
        # Analyze what follows the repeated pattern
        pattern_str = "".join("B" if s == "BIG" else "S" for s in sides[:6])
        # If last 2 match a previous occurrence, predict what came next
        for length in [2, 3]:
            if len(pattern_str) >= length * 2:
                last_sub = pattern_str[:length]
                # Find previous occurrence
                prev_idx = pattern_str.find(last_sub, length)
                if prev_idx != -1 and prev_idx + length < len(pattern_str):
                    next_char = pattern_str[prev_idx + length]
                    expected = "BIG" if next_char == "B" else "SMALL"
                    if l1_pred != expected:
                        override = expected
                        override_reason = f"pattern_repeat_{length}"
                        break
    
    # RULE 5: Low confidence from Level 1 -> check recent momentum
    if l1_strength < 60:
        # Level 1 is uncertain, use momentum
        last3 = sides[:3]
        if last3.count("BIG") >= 2:
            override = "BIG"
            override_reason = "l1_low_confidence_momentum_big"
        elif last3.count("SMALL") >= 2:
            override = "SMALL"
            override_reason = "l1_low_confidence_momentum_small"
    
    # RULE 6: Anti-streak protection
    if patterns["current_streak"]["length"] >= 4:
        # Long streak - high chance of break
        streak_side = patterns["current_streak"]["side"]
        expected = "SMALL" if streak_side == "BIG" else "BIG"
        if l1_pred != expected:
            override = expected
            override_reason = f"anti_streak_{streak_side.lower()}_{patterns['current_streak']['length']}"
    
    # RULE 7: Dominance + recent shift detection
    # If last 10 is heavily one-sided but last 3 shows shift
    if patterns["last10"]["BIG"] >= 7 and patterns["last3"]["SMALL"] >= 2:
        if l1_pred != "SMALL":
            override = "SMALL"
            override_reason = "dominance_shift_to_small"
    elif patterns["last10"]["SMALL"] >= 7 and patterns["last3"]["BIG"] >= 2:
        if l1_pred != "BIG":
            override = "BIG"
            override_reason = "dominance_shift_to_big"
    
    # Apply override if exists
    if override:
        final_pred = override
        final_strength = min(95.0, l1_strength + 10.0)  # Boost confidence when overriding with pattern
        source = "level2_override"
    else:
        final_pred = l1_pred
        final_strength = l1_strength
        source = "level1"
    
    return {
        "prediction": final_pred,
        "strength": round(final_strength, 1),
        "source": source,
        "override_reason": override_reason,
        "level1_prediction": l1_pred,
        "level1_strength": round(l1_strength, 1)
    }

def analyze(rows):
    """Main analysis function with 2-level prediction."""
    recent = rows[:10]
    sides = [row["side"] for row in recent]
    
    # Detect all patterns
    patterns = detect_patterns(sides)
    
    # Level 1: Primary prediction
    level1 = level1_prediction(sides, patterns)
    
    # Level 2: Validation/override
    level2 = level2_prediction(sides, patterns, level1)
    
    # Build evidence list
    evidence = list(level1["evidence"])
    if level2["override_reason"]:
        evidence.append(f"L2_{level2['override_reason']}")
    
    # Counts for reference
    big_count = sides.count("BIG")
    small_count = sides.count("SMALL")
    
    return {
        "prediction": level2["prediction"],
        "strength": level2["strength"],
        "strength_note": "2-level heuristic score. Not a guarantee.",
        "prediction_source": level2["source"],
        "level1": {
            "prediction": level1["prediction"],
            "strength": level1["strength"],
            "scores": level1["scores"],
        },
        "level2": {
            "override_reason": level2["override_reason"],
            "applied": level2["source"] == "level2_override",
        },
        "analyzed_count": len(recent),
        "recent_10": recent,
        "recent_5": recent[:5],
        "big_count_last10": big_count,
        "small_count_last10": small_count,
        "current_streak": patterns["current_streak"],
        "patterns_detected": {
            "last5": patterns["last5"],
            "last3": patterns["last3"],
            "zigzag_6": patterns["zigzag_6"],
            "switch_count_6": patterns["switch_count_6"],
            "double_count": patterns["double_count"],
            "triple_count": patterns["triple_count"],
            "last10": patterns["last10"],
            "odd_positions": patterns["odd_positions"],
            "even_positions": patterns["even_positions"],
            "since_last_switch": patterns["since_last_switch"],
            "pattern_repeats": patterns["pattern_repeats"],
        },
        "evidence": evidence,
    }

@app.get("/")
async def root():
    return {
        "service": "K3 1M JSON Prediction API (2-Level)",
        "status": "ok",
        "version": "2.0.0",
        "endpoints": {
            "health": "/api/health",
            "prediction": "/api/predict",
            "docs": "/api/docs"
        }
    }

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "k3-1m-json-prediction-api", "version": "2.0.0"}

@app.get("/predict")
@app.get("/api/predict")
async def predict():
    rows = await fetch_history()
    current_issue = rows[0]["issueNumber"]
    result = analyze(rows)
    now = datetime.now(timezone.utc)
    
    return {
        "success": True,
        "market": "K3_1M",
        "period": current_issue,
        "prediction_issue": next_issue(current_issue),
        "prediction": result["prediction"],
        "strength": result["strength"],
        "source": result["prediction_source"],
        "level1_prediction": result["level1"]["prediction"],
        "level2_override": result["level2"]["applied"],
        "override_reason": result["level2"]["override_reason"],
        "history_count_used": result["analyzed_count"],
        "recent_10": result["recent_10"],
        "recent_5": result["recent_5"],
        "big_count_last10": result["big_count_last10"],
        "small_count_last10": result["small_count_last10"],
        "current_streak": result["current_streak"],
        "patterns_detected": result["patterns_detected"],
        "evidence": result["evidence"],
        "server_time_utc": now.isoformat(),
        "timer": {
            "period_seconds": 60,
            "seconds_remaining": seconds_remaining(now),
            "market_offset_seconds": MARKET_OFFSET_SECONDS,
            "sync_note": "Wall-clock minute estimate; set MARKET_OFFSET_SECONDS to match the market boundary."
        },
        "strength_note": result["strength_note"],
                                     }
