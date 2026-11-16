# Service interaction transcript (local test)

Service: Flask app from `predict.py` at http://localhost:9696

> Screenshots of interacting with the running service via its demo page
> (`GET /`) are in [`deployment/`](deployment/):
> `demo_bankrupt_prediction.png` (real bankrupt company -> prediction 1)
> and `demo_healthy_prediction.png` (real healthy company -> prediction 0).

## ping — `GET /ping` -> HTTP 200

Response:
```json
PONG
```

## predict (bankrupt company) — `POST /predict` -> HTTP 200

Request:
```json
{
  "ROA(C) before interest and depreciation before interest": 0.370496758153366,
  "ROA(A) before interest and % after tax": 0.410488443087658,
  "ROA(B) before interest and depreciation after tax": 0.416349911665507,
  "Operating Gross Margin": 0.595158477349054,
  "Realized Sales Gross Margin": 0.595353060724427,
  "Operating Profit Rate": 0.998879307456147,
  "Pre-tax net Interest Rate": 0.796
  ... (94 features total, see test_sample.json)
```

Response:
```json
{
  "bankruptcy_prediction": 1,
  "bankruptcy_probability": 1.0,
  "model": "RandomForestClassifier (tuned)"
}

```

## predict (healthy company) — `POST /predict` -> HTTP 200

Request:
```json
{
  "ROA(C) before interest and depreciation before interest": 0.547408960171598,
  "ROA(A) before interest and % after tax": 0.593054949847362,
  "ROA(B) before interest and depreciation after tax": 0.592483537662616,
  "Operating Gross Margin": 0.609737816918664,
  "Realized Sales Gross Margin": 0.609737816918664,
  "Operating Profit Rate": 0.999111788474626,
  "Pre-tax net Interest Rate": 0.797
  ... (94 features total, see test_sample.json)
```

Response:
```json
{
  "bankruptcy_prediction": 0,
  "bankruptcy_probability": 0.0,
  "model": "RandomForestClassifier (tuned)"
}

```

## predict (batch) — `POST /predict` -> HTTP 200

Request:
```json
{
  "records": [
    {
      "ROA(C) before interest and depreciation before interest": 0.370496758153366,
      "ROA(A) before interest and % after tax": 0.410488443087658,
      "ROA(B) before interest and depreciation after tax": 0.416349911665507,
      "Operating Gross Margin": 0.595158477349054,
      "Realized Sales Gross Margin": 0.595353060724427,
      "Operating Profit Rate": 0.99887930
  ... (94 features total, see test_sample.json)
```

Response:
```json
{
  "predictions": [
    {
      "bankruptcy_prediction": 1,
      "bankruptcy_probability": 1.0,
      "model": "RandomForestClassifier (tuned)"
    },
    {
      "bankruptcy_prediction": 0,
      "bankruptcy_probability": 0.0,
      "model": "RandomForestClassifier (tuned)"
    }
  ]
}

```

## predict (validation error) — `POST /predict` -> HTTP 400

Request:
```json
{
  "Equity to Liability": 0.03
}
```

Response:
```json
{
  "error": "missing 93 required feature(s): ['ROA(C) before interest and depreciation before interest', 'ROA(A) before interest and % after tax', 'ROA(B) before interest and depreciation after tax', 'Operating Gross Margin', 'Realized Sales Gross Margin'] ..."
}

```


---

# Docker container test (gunicorn)

Image built with `docker build -t bankruptcy-prediction .` (model trained during the build), then run with `docker run -d --rm -p 9697:9696 bankruptcy-prediction`.

## ping — `GET /ping` -> HTTP 200

Response:
```json
PONG
```

## predict (bankrupt company) — `POST /predict` -> HTTP 200

Request:
```json
{
  "ROA(C) before interest and depreciation before interest": 0.370496758153366,
  "ROA(A) before interest and % after tax": 0.410488443087658,
  "ROA(B) before interest and depreciation after tax": 0.416349911665507,
  "Operating Gross Margin": 0.595158477349054,
  "Realized Sales Gross Margin": 0.595353060724427,
  "Operating Profit Rate": 0.998879307456147,
  "Pre-tax net Interest Rate": 0.796
  ... (94 features total, see test_sample.json)
```

Response:
```json
{"bankruptcy_prediction":1,"bankruptcy_probability":1.0,"model":"RandomForestClassifier (tuned)"}

```

## predict (healthy company) — `POST /predict` -> HTTP 200

Request:
```json
{
  "ROA(C) before interest and depreciation before interest": 0.547408960171598,
  "ROA(A) before interest and % after tax": 0.593054949847362,
  "ROA(B) before interest and depreciation after tax": 0.592483537662616,
  "Operating Gross Margin": 0.609737816918664,
  "Realized Sales Gross Margin": 0.609737816918664,
  "Operating Profit Rate": 0.999111788474626,
  "Pre-tax net Interest Rate": 0.797
  ... (94 features total, see test_sample.json)
```

Response:
```json
{"bankruptcy_prediction":0,"bankruptcy_probability":0.0,"model":"RandomForestClassifier (tuned)"}

```

## predict (batch) — `POST /predict` -> HTTP 200

Request:
```json
{
  "records": [
    {
      "ROA(C) before interest and depreciation before interest": 0.370496758153366,
      "ROA(A) before interest and % after tax": 0.410488443087658,
      "ROA(B) before interest and depreciation after tax": 0.416349911665507,
      "Operating Gross Margin": 0.595158477349054,
      "Realized Sales Gross Margin": 0.595353060724427,
      "Operating Profit Rate": 0.99887930
  ... (94 features total, see test_sample.json)
```

Response:
```json
{"predictions":[{"bankruptcy_prediction":1,"bankruptcy_probability":1.0,"model":"RandomForestClassifier (tuned)"},{"bankruptcy_prediction":0,"bankruptcy_probability":0.0,"model":"RandomForestClassifier (tuned)"}]}

```

## predict (validation error) — `POST /predict` -> HTTP 400

Request:
```json
{
  "Equity to Liability": 0.03
}
```

Response:
```json
{"error":"missing 93 required feature(s): ['ROA(C) before interest and depreciation before interest', 'ROA(A) before interest and % after tax', 'ROA(B) before interest and depreciation after tax', 'Operating Gross Margin', 'Realized Sales Gross Margin'] ..."}

```

