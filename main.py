
import os
import json
import asyncio
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(
    title="WinGo 30S JSON API",
    version="2.0.0",
)

# Allows websites to call YOUR Render API from a browser.
# This does not bypass restrictions imposed by the upstream API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

SOURCE_URL = os.getenv(
    "WINGO_API_URL",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/"
    "GetHistoryIssuePage.json?pageNo=1&pageSize=10",
)

# Optional: set FALLBACK_API_URL only if you have another
# authorized API that returns compatible WinGo history JSON.
FALLBACK_URL = os.getenv("FALLBACK_API_URL", "").strip()

TIMEOUT = httpx.Timeout(15.0, connect=8.0)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 10) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Mobile Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://draw.ar-lottery01.com/",
}

# In-memory tracking. Render restarts may clear these records.
predictions = {}
settled_results = {}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def validate_source_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("API URL must be a valid HTTPS URL")


def extract_records(payload):
    """Support common JSON response structures."""
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("API response is not a JSON object or list")

    # Some providers wrap their results under data.
    current = payload

    for _ in range(4):
        if isinstance(current, list):
            return current

        if not isinstance(current, dict):
            break

        for key in ("list", "records", "history", "items"):
            value = current.get(key)
            if isinstance(value, list):
                return value

        next_value = current.get("data")
        if isinstance(next_value, (dict, list)):
            current = next_value
            continue

        break

    raise ValueError(
        "Could not find a history list in the API response"
    )


async def request_history(url):
    validate_source_url(url)

    async with httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers=HEADERS,
    ) as client:
        response = await client.get(url)

        if response.status_code != 200:
            raise ValueError(
                f"Upstream HTTP {response.status_code} "
                f"from {urlparse(url).netloc}: "
                f"{response.text[:250]}"
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise ValueError(
                "Upstream returned non-JSON content: "
                + response.text[:250]
            ) from exc

    if isinstance(payload, dict):
        # Do not treat a provider's explicit failure as history.
        if payload.get("success") is False:
            raise ValueError(
                "Upstream API reported success=false: "
                + str(payload.get("msg") or payload.get("message")
                      or payload.get("error") or "unknown error")
            )

        # Some APIs use code=0 for success.
        code = payload.get("code")
        if code not in (None, 0, "0", 200, "200"):
            raise ValueError(
                f"Upstream API returned code={code}: "
                + str(payload.get("msg") or payload.get("message") or "")
            )

    records = extract_records(payload)

    if not records:
        raise ValueError("The upstream API returned an empty history list")

    return records


async def fetch_history():
    """Try the direct API first, then an optional configured fallback."""
    sources = [("primary", SOURCE_URL)]

    if FALLBACK_URL and FALLBACK_URL != SOURCE_URL:
        sources.append(("fallback", FALLBACK_URL))

    errors = []

    for source_name, url in sources:
        try:
            records = await request_history(url)
            return records, source_name
        except Exception as exc:
            errors.append({
                "source": source_name,
                "error": str(exc),
            })

    raise RuntimeError(
        json.dumps(errors, ensure_ascii=False)
    )


def field_value(record, *names):
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def normalize_history(records):
    history = []

    for record in records:
        if not isinstance(record, dict):
            continue

        period = field_value(
            record,
            "issueNumber",
            "issue",
            "period",
            "periodNumber",
            "drawId",
        )

        number = field_value(
            record,
            "number",
            "openNumber",
            "result",
            "winNumber",
        )

        if period is None or number is None:
            continue

        number_text = str(number).strip()

        # Do not guess a result if the provider's format is unknown.
        if not number_text.isdigit():
            continue

        number_int = int(number_text)

        if not 0 <= number_int <= 9:
            continue

        history.append({
            "period": str(period).strip(),
            "number": number_int,
            "size": "BIG" if number_int >= 5 else "SMALL",
        })

    # Remove duplicate periods.
    unique = {}
    for item in history:
        unique[item["period"]] = item

    history = list(unique.values())

    if not history:
        raise ValueError(
            "History received, but period/number fields "
            "were not recognized. Inspect the source JSON."
        )

    # Sort newest-first only when period IDs are numeric.
    if all(item["period"].isdigit() for item in history):
        history.sort(
            key=lambda item: int(item["period"]),
            reverse=True,
        )

    return history[:10]


def get_next_period(period):
    value = str(period).strip()

    if not value.isdigit():
        raise ValueError(
            "Cannot safely calculate next period from "
            f"non-numeric period ID: {value}"
        )

    return str(int(value) + 1).zfill(len(value))


def analyze(history):
    sizes = [item["size"] for item in history]
    recent5 = sizes[:5]
    recent10 = sizes[:10]

    current_size = sizes[0]
    streak = 0

    for size in sizes:
        if size != current_size:
            break
        streak += 1

    big_count = recent10.count("BIG")
    small_count = recent10.count("SMALL")

    # Weighted descriptive heuristic, not a trained probability model.
    big_score = 0
    small_score = 0

    for index, size in enumerate(recent10):
        weight = max(1, 5 - index // 2)
        if size == "BIG":
            big_score += weight
        else:
            small_score += weight

    if streak >= 3:
        if current_size == "BIG":
            big_score += 2
        else:
            small_score += 2

    if big_score > small_score:
        prediction = "BIG"
    elif small_score > big_score:
        prediction = "SMALL"
    else:
        prediction = "NO_CLEAR_SIGNAL"

    difference = abs(big_score - small_score)

    if prediction == "NO_CLEAR_SIGNAL" or difference <= 2:
        strength = "LOW"
    elif difference <= 6:
        strength = "MODERATE"
    else:
        strength = "STRONGER_SIGNAL"

    alternating = (
        len(recent5) >= 4
        and all(
            recent5[i] != recent5[i + 1]
            for i in range(len(recent5) - 1)
        )
    )

    return {
        "prediction": prediction,
        "signal_strength": strength,
        "analysis": {
            "current_streak": {
                "size": current_size,
                "length": streak,
            },
            "streak_break": (
                "Long streak detected; reversal is possible"
                if streak >= 4
                else "No long streak detected"
            ),
            "trend_follow": (
                recent5.count(current_size) >= 3
            ),
            "zigzag_alternate": alternating,
            "recent_5": recent5,
            "recent_10": recent10,
            "big_count_last_10": big_count,
            "small_count_last_10": small_count,
            "dominance": (
                "BIG" if big_count > small_count
                else "SMALL" if small_count > big_count
                else "BALANCED"
            ),
            "weighted_big_score": big_score,
            "weighted_small_score": small_score,
        },
    }


def update_settlements(history):
    """Resolve any saved prediction whose period has now appeared."""
    by_period = {item["period"]: item for item in history}

    for period, saved in list(predictions.items()):
        actual = by_period.get(period)

        if actual and period not in settled_results:
            settled_results[period] = {
                "period": period,
                "prediction": saved["prediction"],
                "actual": actual["size"],
                "number": actual["number"],
                "result": (
                    "WIN"
                    if saved["prediction"] == actual["size"]
                    else "LOSS"
                ),
            }


@app.get("/")
async def home():
    return {
        "service": "WinGo 30S JSON Prediction API",
        "status": "online",
        "endpoints": ["/health", "/history", "/predict"],
        "note": "Predictions are heuristic estimates, not guarantees.",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp_utc": now_utc(),
    }


@app.get("/history")
async def history_endpoint():
    try:
        raw, source = await fetch_history()
        history = normalize_history(raw)

        return {
            "success": True,
            "source_used": source,
            "count": len(history),
            "history": history,
            "generated_at_utc": now_utc(),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "history_api_error",
                "detail": str(exc),
                "generated_at_utc": now_utc(),
            },
        )


@app.get("/predict")
async def predict_endpoint():
    try:
        raw, source = await fetch_history()
        history = normalize_history(raw)

        update_settlements(history)

        latest = history[0]
        target_period = get_next_period(latest["period"])
        analysis = analyze(history)

        # Avoid overwriting an existing prediction for the same period.
        if target_period not in predictions:
            predictions[target_period] = {
                "period": target_period,
                "prediction": analysis["prediction"],
                "based_on_period": latest["period"],
            }

        saved_prediction = predictions[target_period]

        return {
            "success": True,
            "source_used": source,
            "prediction": {
                "period": target_period,
                "prediction": saved_prediction["prediction"],
                "signal_strength": analysis["signal_strength"],
                "based_on_period": latest["period"],
                "latest_actual": {
                    "number": latest["number"],
                    "size": latest["size"],
                },
                "analysis": analysis["analysis"],
            },
            "previous_prediction_result": (
                settled_results.get(
                    str(int(target_period) - 1).zfill(len(target_period))
                )
                if target_period.isdigit()
                else None
            ),
            "recent_settled_results": list(
                settled_results.values()
            )[-20:],
            "history_count": len(history),
            "generated_at_utc": now_utc(),
            "warning": (
                "BIG/SMALL outcomes may be random. "
                "No prediction method guarantees a win."
            ),
        }

    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": "prediction_api_error",
                "detail": str(exc),
                "generated_at_utc": now_utc(),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )
