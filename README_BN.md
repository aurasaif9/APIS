# WinGo 30S JSON Prediction API

Python/FastAPI service: last 10 history records থেকে BIG/SMALL prediction তৈরি করে এবং আগের saved prediction-এর settled WIN/LOSS দেখায়। শুধু BIG/SMALL দেখায়—predicted digit প্রকাশ করে না।

## Endpoints
- `/` — service health JSON
- `/api/predict` — target period, BIG/SMALL prediction, এবং আগের settled period-এর WIN/LOSS
- `/api/history` — source history JSON
- `/api/history?count=20` — ৫–৫০টি history দেখতে পারবেন

## Example `/api/predict` response
```json
{
  "success": true,
  "period": "2026102112346",
  "prediction": "SMALL",
  "previous_period": "2026102112345",
  "previous_prediction": "BIG",
  "previous_actual": "SMALL",
  "previous_result": "LOSS",
  "history_count_used": 10
}
```

Example values are illustrative. প্রথমবার চালালে আগে prediction save না থাকায় `previous_result` হবে `NO_SAVED_PREDICTION`; পরবর্তী period-এ history update হলে saved prediction-এর সঙ্গে actual result মিলিয়ে WIN/LOSS হবে। একই period-এর prediction বারবার request করলেও বদলাবে না, যাতে result tracking ঠিক থাকে।

## Render deploy
1. ZIP extract করে ফাইলগুলো GitHub repository-তে upload করুন।
2. Render-এ New + → Web Service দিয়ে repository connect করুন।
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Render-এ disk persistence না থাকলে restart/redeploy-এ SQLite state হারাতে পারে। স্থায়ী WIN/LOSS tracking চাইলে persistent disk যুক্ত করে `PREDICTION_DB_PATH` disk path-এ সেট করুন।

## Important
- History source: `https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json`
- `period` = latest settled period + 1 (numeric period format ধরে)।
- শেষ ১০টি history থেকে pattern analysis করা হয়, কিন্তু কোনও pattern-ই loss-free result বা 2 level-এর মধ্যে win নিশ্চিত করতে পারে না।
- Upstream API hosting IP-কে 403 দিলে endpoint error detail ফেরত দেবে।
