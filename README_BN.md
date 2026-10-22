# K3 1M JSON-only Prediction API

This version intentionally has NO `index.html` and no visual dashboard. Every endpoint returns JSON.

## Files
- `cloud-functions/api/index.py`
- `cloud-functions/requirements.txt`

## Deploy from an Android phone
1. Extract this ZIP.
2. Create a GitHub repository and upload the extracted files/folders, keeping the same structure.
3. Open https://pages.edgeone.ai/ and create a project connected to that repository.
4. Deploy using EdgeOne's Python/FastAPI function template/runtime. Confirm the runtime supports Python and the `cloud-functions/api` convention for this project.
5. Test the deployed domain:
   - `/api/health`
   - `/api/predict`
   - `/api/docs` (FastAPI-generated docs; this is not a custom dashboard)

## API behaviour
- Fetches the K3 history endpoint from `K3_HISTORY_URL` or the default source URL.
- Reads `data.list`, `issueNumber`, and `sum`.
- Uses latest 10 valid rows.
- SUM 3-10 = SMALL; SUM 11-18 = BIG.
- Returns prediction for latest numeric issueNumber + 1.
- Uses recency weighting, last-five trend, current streak, zigzag and dominance.
- Timer field is an estimated 60-second wall-clock countdown. If the official market timer has a different boundary, set `MARKET_OFFSET_SECONDS` in EdgeOne to a value from 0 to 59. An issueNumber/sum-only API does not provide exact remaining seconds.

## Errors
If the upstream source blocks EdgeOne or returns non-JSON/HTTP 403, the endpoint will return a visible HTTP error with detail. Hosting does not guarantee the source API will permit server-side requests.

Prediction scores are heuristic only; they are not a probability or guarantee of winning.
