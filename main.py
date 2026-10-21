
import os
import json
import time
from datetime import datetime, timezone
from urllib.parse import quote

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="WinGo 30S Prediction API", version="1.0.0")

SOURCE_URL = (
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/"
    "GetHistoryIssuePage.json?pageNo=1&pageSize=10"
)

PROXY_URL = "https://api.codetabs.com/v1/proxy/?quest="

# Store prediction records while this process is running.
prediction_records = {}
last_prediction = None


def fetch_history():
    """Fetch the latest WinGo history through CodeTabs proxy."""
    proxy_target = PROXY_URL + quote(SOURCE_URL, safe="")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
    }

    with httpx.Client(timeout=20, follow_redirects=True) as client:
        response = client.get(proxy_target, headers=headers)
        response.raise_for_status()
        data = response.json()

    if not isinstance(data, dict):
        raise ValueError("Unexpected response format from proxy")

    if data.get("success") is False:
        raise ValueError("Source API returned success=false")

    # Common history response formats
    payload = data.get("data", data)

    if isinstance(payload, dict):
        records = (
            payload.get("list")
            or payload.get("records")
            or payload.get("history")
            or payload.get("data")
        )
    elif isinstance(payload, list):
        records = payload
    else:
        records = None

    if not isinstance(records, list) or not records:
        raise ValueError(
            "No history records found. Proxy response: "
            + json.dumps(data, ensure_ascii=False)[:500]
        )

    return records[:10]


def get_field(record, *names):
    """Read a field while allowing common API naming variations."""
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def normalize_history(records):
    results = []

    for item in records:
        if not isinstance(item, dict):
            continue

        issue = get_field(
            item, "issueNumber", "issue", "period", "periodNumber"
        )
        number = get_field(
            item, "number", "result", "openNumber", "winNumber"
        )

        if issue is None or number is None:
            continue

        # Some APIs return the winning number as a string.
        number_text = str(number).strip()

        # Do not guess if the result is not a single digit.
        if not number_text.isdigit():
            continue

        number_int = int(number_text)
        if not 0 <= number_int <= 9:
            continue

        results.append({
            "period": str(issue),
            "number": number_int,
            "size": "BIG" if number_int >= 5 else "SMALL",
        })

    if not results:
        raise ValueError(
            "History was received, but its period/number fields "
            "could not be recognized."
        )

    return results


def next_period(period):
    """Increment numeric period IDs; report an error for unsupported IDs."""
    period_text = str(period).strip()

    if not period_text.isdigit():
        raise ValueError(
            f"Cannot increment non-numeric period ID: {period_text}"
        )

    width = len(period_text)
    return str(int(period_text) + 1).zfill(width)


def analyze_history(history):
    """
    Simple descriptive pattern analysis.
    This is not a guaranteed prediction of a random outcome.
    """
    sizes = [item["size"] for item in history]
    recent5 = sizes[:5]
    recent10 = sizes[:10]

    # Current streak begins at the newest result.
    current_size = sizes[0]
    streak_length = 0

    for size in sizes:
        if size == current_size:
            streak_length += 1
        else:
            break

    big_count = recent10.count("BIG")
    small_count = recent10.count("SMALL")

    # Use a transparent heuristic, not a claimed win probability.
    score_big = 0
    score_small = 0

    # Recent results receive more weight.
    for index, size in enumerate(recent10):
        weight = max(1, 5 - index // 2)
        if size == "BIG":
            score_big += weight
        else:
            score_small += weight

    # Modest streak signal.
    if streak_length >= 3:
        if current_size == "BIG":
            score_big += 2
        else:
            score_small += 2

    if score_big > score_small:
        prediction = "BIG"
    elif score_small > score_big:
        prediction = "SMALL"
    else:
        # No clear signal: follow the most recent result, but
        # explicitly identify the result as a weak heuristic.
        prediction = current_size

    confidence_label = (
        "LOW" if abs(score_big - score_small) <= 2
        else "MODERATE"
    )

    return {
        "prediction": prediction,
        "signal_strength": confidence_label,
        "analysis": {
            "current_streak": {
                "size": current_size,
                "length": streak_length,
            },
            "streak_break": (
                "Possible reversal after a long streak"
                if streak_length >= 4
                else "No long streak break signal"
            ),
            "trend_follow": (
                recent5.count(current_size) >= 3
            ),
            "zigzag_alternate": (
                len(recent5) >= 4
                and all(
                    recent5[i] != recent5[i + 1]
                    for i in range(len(recent5) - 1)
                )
            ),
            "recent_5": recent5,
            "recent_10": recent10,
            "big_count_last_10": big_count,
            "small_count_last_10": small_count,
            "dominance": (
                "BIG" if big_count > small_count
                else "SMALL" if small_count > big_count
                else "BALANCED"
            ),
            "weighted_big_score": score_big,
            "weighted_small_score": score_small,
        },
    }


@app.get("/")
def home():
    return {
        "service": "WinGo 30S Prediction API",
        "status": "online",
        "history_source": "CodeTabs proxy",
        "endpoints": {
            "prediction": "/predict",
            "history": "/history",
            "health": "/health",
        },
        "note": "Predictions are heuristic estimates, not guaranteed results.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/history")
def history_endpoint():
    try:
        raw = fetch_history()
        history = normalize_history(raw)
        return {
            "success": True,
            "count": len(history),
            "history": history,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "history_api_error",
                "detail": str(exc),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


@app.get("/predict")
def predict():
    global last_prediction

    try:
        raw = fetch_history()
        history = normalize_history(raw)

        # API records are normally newest first.
        latest = history[0]
        target_period = next_period(latest["period"])

        analysis = analyze_history(history)

        prediction = {
            "period": target_period,
            "prediction": analysis["prediction"],
            "signal_strength": analysis["signal_strength"],
            "based_on_period": latest["period"],
            "based_on_latest_result": latest["size"],
            "analysis": analysis["analysis"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Only evaluate a previous prediction once its period appears
        # in the fetched history.
        previous_result = None
        if last_prediction:
            for item in history:
                if item["period"] == last_prediction["period"]:
                    outcome = (
                        "WIN"
                        if item["size"] == last_prediction["prediction"]
                        else "LOSS"
                    )
                    previous_result = {
                        "period": item["period"],
                        "prediction": last_prediction["prediction"],
                        "actual": item["size"],
                        "number": item["number"],
                        "result": outcome,
                    }
                    prediction_records[item["period"]] = previous_result
                    break

        last_prediction = {
            "period": target_period,
            "prediction": analysis["prediction"],
        }

        return {
            "success": True,
            "prediction": prediction,
            "previous_prediction_result": previous_result,
            "recent_settled_results": list(
                prediction_records.values()
            )[-20:],
            "history_count": len(history),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "warning": (
                "BIG/SMALL outcomes may be random. "
                "This analysis cannot guarantee a win."
            ),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "prediction_api_error",
                "detail": str(exc),
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
