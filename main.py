"""WinGo 30S history analyzer JSON API.

Endpoints:
  GET /             -> health/info JSON
  GET /api/predict  -> fetch latest 10 results and return a BIG/SMALL analysis JSON
  GET /api/history  -> raw parsed latest history JSON

This is an experimental pattern analyzer, not a guaranteed betting signal.
"""
from collections import Counter
from datetime import datetime, timezone
from typing import Any
import logging
import os
import sqlite3
from pathlib import Path

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("wingo-json-api")

HISTORY_URL = os.getenv(
    "WINGO_HISTORY_URL",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)
HISTORY_COUNT = 10
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "12"))
DB_PATH = os.getenv("PREDICTION_DB_PATH", str(Path(__file__).with_name("prediction_state.db")))

app = FastAPI(title="WinGo 30S Prediction JSON API", version="1.0.0")


def fetch_history(count: int = HISTORY_COUNT) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 Chrome/131.0.0.0 Mobile Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://hgnice9.com/",
        "Origin": "https://hgnice9.com",
        "Cache-Control": "no-cache",
    }
    with httpx.Client(timeout=TIMEOUT, headers=headers, follow_redirects=True) as client:
        response = client.get(HISTORY_URL, params={"pageNo": 1, "pageSize": max(count, 10)})
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("History API returned non-object JSON")
    if payload.get("code") not in (0, "0", None):
        raise ValueError(f"History API code={payload.get('code')!r}, msg={payload.get('msg')!r}")
    data = payload.get("data") or {}
    rows = data.get("list") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError("History API response missing data.list array")
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            issue = str(row.get("issueNumber", "")).strip()
            number = int(row.get("number"))
            if issue and 0 <= number <= 9:
                parsed.append({
                    "period": issue,
                    "number": number,
                    "size": "BIG" if number >= 5 else "SMALL",
                    "color": str(row.get("color", "")),
                })
        except (TypeError, ValueError):
            continue
    if len(parsed) < 5:
        raise ValueError(f"History API returned only {len(parsed)} usable records; need at least 5")
    # API is expected to return newest first. Keep only requested count.
    return parsed[:count]


def run_analysis(history: list[dict[str, Any]]) -> dict[str, Any]:
    # Convert into chronological order for pattern calculations.
    chronological = list(reversed(history))
    nums = [int(x["number"]) for x in chronological]
    sides = ["BIG" if n >= 5 else "SMALL" for n in nums]
    recent_sides = sides[-5:]
    recent_nums = nums[-5:]
    votes: list[tuple[str, str, float]] = []

    def vote(label: str, side: str, weight: float) -> None:
        if side in ("BIG", "SMALL"):
            votes.append((label, side, weight))

    # 1-2: recent trend and last-five dominance
    b5, s5 = recent_sides.count("BIG"), recent_sides.count("SMALL")
    if b5 != s5:
        vote("recent_5_trend", "BIG" if b5 > s5 else "SMALL", 1.4)
    # 3: current streak follow, weak evidence only
    streak_side = sides[-1]
    streak_len = 1
    for side in reversed(sides[:-1]):
        if side == streak_side:
            streak_len += 1
        else:
            break
    if streak_len >= 2:
        vote("streak_follow", streak_side, min(1.3, 0.55 + 0.15 * streak_len))
    # 4: streak break based on recent streak transitions; if long run, give mild opposite vote
    if streak_len >= 4:
        vote("long_streak_break", "SMALL" if streak_side == "BIG" else "BIG", 0.45)
    # 5: overall 10-result dominance
    b10, s10 = sides.count("BIG"), sides.count("SMALL")
    if b10 != s10:
        vote("last_10_dominance", "BIG" if b10 > s10 else "SMALL", 0.9)
    # 6: one-step transition counts from chronological sequence
    transitions: dict[str, Counter] = {"BIG": Counter(), "SMALL": Counter()}
    for a, b in zip(sides[:-1], sides[1:]):
        transitions[a][b] += 1
    next_counts = transitions[sides[-1]]
    if next_counts["BIG"] != next_counts["SMALL"] and sum(next_counts.values()) >= 2:
        vote("transition_pattern", "BIG" if next_counts["BIG"] > next_counts["SMALL"] else "SMALL", 1.0)
    # 7: zigzag/alternation pattern across latest 4 outcomes
    last4 = sides[-4:]
    alternations = sum(1 for a, b in zip(last4[:-1], last4[1:]) if a != b)
    if alternations >= 3:
        vote("zigzag_pattern", "SMALL" if sides[-1] == "BIG" else "BIG", 0.65)
    elif alternations <= 1 and last4.count("BIG") != last4.count("SMALL"):
        vote("run_pattern", "BIG" if last4.count("BIG") > last4.count("SMALL") else "SMALL", 0.55)
    # 8: repeating side pattern: find earlier matching 2-side suffix and inspect what followed it
    seq = sides
    repeat_follow = Counter()
    for i in range(0, len(seq) - 2):
        if seq[i:i+2] == seq[-2:] and i + 2 < len(seq):
            repeat_follow[seq[i+2]] += 1
    if repeat_follow["BIG"] != repeat_follow["SMALL"] and sum(repeat_follow.values()) > 0:
        vote("repeated_pair", "BIG" if repeat_follow["BIG"] > repeat_follow["SMALL"] else "SMALL", 0.7)
    # 9: digit distribution as very weak contextual vote only if clear conditional repeat
    digit_counts = Counter(nums)
    frequent_digits = [n for n, c in digit_counts.items() if c >= 2]
    if frequent_digits:
        frequent_side_counts = Counter("BIG" if n >= 5 else "SMALL" for n in frequent_digits)
        if frequent_side_counts["BIG"] != frequent_side_counts["SMALL"]:
            vote("repeated_digit_side", "BIG" if frequent_side_counts["BIG"] > frequent_side_counts["SMALL"] else "SMALL", 0.35)
    # 10: short momentum in last 3
    b3, s3 = sides[-3:].count("BIG"), sides[-3:].count("SMALL")
    if b3 != s3:
        vote("last_3_momentum", "BIG" if b3 > s3 else "SMALL", 0.85)
    # 11: recent numeric average is only a weak size signal
    avg = sum(recent_nums) / len(recent_nums)
    if avg != 4.5:
        vote("recent_number_average", "BIG" if avg > 4.5 else "SMALL", 0.35)
    # 12: parity context, deliberately weak and not used as a reliable causal signal
    odd_count = sum(n % 2 for n in nums[-6:])
    even_count = 6 - odd_count
    if odd_count != even_count:
        # No direct Big/Small relationship; expose as a pattern stat rather than a vote.
        pass

    score = {"BIG": 0.0, "SMALL": 0.0}
    for _, side, weight in votes:
        score[side] += weight
    total = score["BIG"] + score["SMALL"]
    if total == 0:
        prediction = "BIG" if b10 > s10 else "SMALL" if s10 > b10 else ("SMALL" if sides[-1] == "BIG" else "BIG")
    elif score["BIG"] == score["SMALL"]:
        prediction = "BIG" if b5 > s5 else "SMALL" if s5 > b5 else ("SMALL" if sides[-1] == "BIG" else "BIG")
    else:
        prediction = "BIG" if score["BIG"] > score["SMALL"] else "SMALL"
    agreement = max(score.values()) / total if total else 0.5
    # Evidence index only; not a probability of winning. Cap to avoid overstating a tiny sample.
    evidence_score = max(35, min(75, round(100 * agreement * min(1.0, len(votes) / 5)))) if votes else 35

    # Pick a plausible digit from the chosen side using frequency and recency; this is a suggestion, not a guaranteed number.
    valid_digits = [n for n in range(5, 10)] if prediction == "BIG" else [n for n in range(0, 5)]
    side_digit_counts = Counter(n for n in nums if n in valid_digits)
    if side_digit_counts:
        max_count = max(side_digit_counts.values())
        candidates = [n for n, c in side_digit_counts.items() if c == max_count]
        # Prefer the most recently observed among tied frequent digits.
        predicted_number = next((n for n in reversed(nums) if n in candidates), candidates[0])
    else:
        predicted_number = 7 if prediction == "BIG" else 2

    latest_period = str(history[0]["period"])
    try:
        next_period = str(int(latest_period) + 1)
    except ValueError:
        next_period = None

    return {
        "prediction": prediction,
        "predicted_number": predicted_number,
        "evidence_score": evidence_score,
        "score_type": "experimental_pattern_evidence_not_win_probability",
        "next_period": next_period,
        "latest_settled_period": latest_period,
        "history_count": len(history),
        "history": history,
        "patterns": {
            "last_5": [{"number": n, "size": s} for n, s in zip(reversed(recent_nums), reversed(recent_sides))],
            "big_count_last_10": b10,
            "small_count_last_10": s10,
            "current_streak": {"side": streak_side, "length": streak_len},
            "last_4_alternations": alternations,
            "recent_5_big": b5,
            "recent_5_small": s5,
            "last_5_number_average": round(avg, 2),
            "odd_count_last_6": odd_count,
            "even_count_last_6": even_count,
            "transition_after_big": dict(transitions["BIG"]),
            "transition_after_small": dict(transitions["SMALL"]),
            "signals": [{"name": name, "prediction": side, "weight": weight} for name, side, weight in votes],
            "signal_votes": len(votes),
            "weighted_score": {k: round(v, 2) for k, v in score.items()},
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Past results do not guarantee the next result. Evidence score is not a win probability.",
    }


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        period TEXT PRIMARY KEY, prediction TEXT NOT NULL,
        source_period TEXT NOT NULL, actual TEXT, status TEXT, created_at TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def settle_predictions(history: list[dict[str, Any]]) -> None:
    """Settle any saved prediction whose period now appears in official history."""
    with db_connect() as conn:
        for item in history:
            period = str(item["period"])
            actual = str(item["size"])
            row = conn.execute("SELECT prediction, status FROM predictions WHERE period=?", (period,)).fetchone()
            if row and row["status"] is None:
                status = "WIN" if row["prediction"] == actual else "LOSS"
                conn.execute("UPDATE predictions SET actual=?, status=? WHERE period=?", (actual, status, period))
        conn.commit()


def get_previous_result(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return outcome for the most recently settled period with a saved prediction."""
    with db_connect() as conn:
        for item in history:
            row = conn.execute(
                "SELECT prediction, actual, status FROM predictions WHERE period=?",
                (str(item["period"]),),
            ).fetchone()
            if row and row["status"]:
                return {
                    "period": str(item["period"]),
                    "prediction": row["prediction"],
                    "actual": row["actual"],
                    "result": row["status"],
                }
    latest = history[0]
    return {
        "period": str(latest["period"]),
        "prediction": None,
        "actual": str(latest["size"]),
        "result": "NO_SAVED_PREDICTION",
    }


def save_or_get_prediction(period: str, source_period: str, prediction: str) -> str:
    """Keep a prediction stable for each target period to make WIN/LOSS tracking valid."""
    with db_connect() as conn:
        row = conn.execute("SELECT prediction FROM predictions WHERE period=?", (period,)).fetchone()
        if row:
            return str(row["prediction"])
        conn.execute(
            "INSERT INTO predictions(period,prediction,source_period,created_at) VALUES(?,?,?,?)",
            (period, prediction, source_period, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    return prediction


@app.get("/")
def root() -> dict[str, Any]:
    return {"status": "ok", "service": "WinGo 30S JSON prediction API", "endpoints": ["/api/predict", "/api/history"]}


@app.get("/api/history")
def history_endpoint(count: int = Query(default=10, ge=5, le=50)) -> dict[str, Any]:
    try:
        history = fetch_history(count)
        return {"success": True, "count": len(history), "history": history, "source": HISTORY_URL}
    except Exception as exc:
        log.exception("History fetch failed")
        return JSONResponse(status_code=502, content={"success": False, "error": "history_api_error", "detail": f"{type(exc).__name__}: {exc}", "source": HISTORY_URL})


@app.get("/api/predict")
def predict_endpoint() -> Any:
    try:
        history = fetch_history(HISTORY_COUNT)
        settle_predictions(history)
        analysis = run_analysis(history)
        next_period = analysis.get("next_period")
        if not next_period:
            raise ValueError("Could not calculate next period from latest history period")

        # Only expose BIG/SMALL, period, and the previous period's recorded WIN/LOSS.
        # The full pattern analysis remains internal to the service.
        prediction = save_or_get_prediction(
            str(next_period), str(history[0]["period"]), str(analysis["prediction"])
        )
        previous = get_previous_result(history)
        return {
            "success": True,
            "period": str(next_period),
            "prediction": prediction,
            "previous_period": previous["period"],
            "previous_prediction": previous["prediction"],
            "previous_actual": previous["actual"],
            "previous_result": previous["result"],
            "history_count_used": len(history),
            "source": HISTORY_URL,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": "WIN/LOSS is reported only when this service has a saved prediction for that settled period. Pattern analysis cannot guarantee a win.",
        }
    except Exception as exc:
        log.exception("Prediction failed")
        return JSONResponse(status_code=502, content={"success": False, "error": "history_api_error", "detail": f"{type(exc).__name__}: {exc}", "source": HISTORY_URL, "generated_at_utc": datetime.now(timezone.utc).isoformat()})
