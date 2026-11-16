"""
Web service for the Company Bankruptcy Prediction project.

Loads the trained model from model.pkl (created by train.py) and serves
predictions with Flask.

Endpoints:
    GET  /ping    -> health check ("PONG")
    POST /predict -> bankruptcy prediction

Request body for /predict: JSON with the 94 financial indicators of one
company, e.g.
    {
        "ROA(C) before interest and depreciation before interest": 0.45,
        ... 93 more features ...,
        "Equity to Liability": 0.03
    }
Or a list of such objects (or {"records": [...]}) for batch predictions.
See test_sample.json for ready-to-use examples.

Response:
    {
        "bankruptcy_probability": 0.87,
        "bankruptcy_prediction": 1,
        "model": "RandomForestClassifier (tuned)"
    }

Run:
    python train.py      # once, to create model.pkl
    python predict.py    # starts the service on http://localhost:9696

In production / Docker it is served with:
    gunicorn --bind=0.0.0.0:9696 predict:app
"""

import json
import math
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = "model.pkl"
SAMPLE_PATH = "test_sample.json"
PORT = 9696

with open(MODEL_PATH, "rb") as f_in:
    artifact = pickle.load(f_in)

model = artifact["model"]
scaler = artifact["scaler"]
columns = artifact["columns"]
model_name = artifact["model_name"]
threshold = artifact.get("threshold", 0.5)

# optional: ready-to-use sample payloads for the demo page
samples = {}
try:
    with open(SAMPLE_PATH, encoding="utf-8") as f_in:
        samples = json.load(f_in)
except (OSError, ValueError):
    pass

DEMO_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>Company Bankruptcy Prediction</title>
  <style>
    body {{ font-family: Consolas, monospace; background: #10141f; color: #d7dce8;
           max-width: 860px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ color: #7aa2f7; font-size: 22px; }}
    h2 {{ color: #9ece6a; font-size: 15px; margin-top: 30px; }}
    button {{ background: #7aa2f7; color: #10141f; border: 0; padding: 10px 18px;
              margin: 6px 8px 6px 0; font-family: inherit; font-size: 14px;
              border-radius: 6px; cursor: pointer; }}
    button:hover {{ background: #9ece6a; }}
    pre {{ background: #1a2030; border: 1px solid #2c3550; border-radius: 8px;
           padding: 14px; overflow-x: auto; min-height: 20px; }}
    .url {{ color: #bb9af7; }}
  </style>
</head>
<body>
  <h1>Company Bankruptcy Prediction</h1>
  <p>Random Forest trained on 94 financial indicators of Taiwanese companies
     (holdout F1 = 0.8246, MCC = 0.6364, ROC AUC = 0.8638).</p>

  <h2>API</h2>
  <pre><span class="url">GET  /ping</span>    -> PONG
<span class="url">POST /predict</span> -> JSON with 94 features -> {{"bankruptcy_probability": ..., "bankruptcy_prediction": ...}}</pre>

  <h2>Try it</h2>
  <button onclick="predict('single_bankrupt_company')">Predict: real bankrupt company</button>
  <button onclick="predict('single_healthy_company')">Predict: real healthy company</button>
  <pre id="out">Click a button to send a POST /predict request...</pre>

<script>
async function predict(key) {{
  const out = document.getElementById("out");
  out.textContent = "POST /predict ...";
  try {{
    const resp = await fetch("/predict", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(SAMPLES[key]),
    }});
    const data = await resp.json();
    out.textContent = "HTTP " + resp.status + "\\n\\n" + JSON.stringify(data, null, 2);
  }} catch (err) {{
    out.textContent = "Error: " + err;
  }}
}}
const SAMPLES = {samples_json};
</script>
</body>
</html>"""

app = Flask("bankruptcy")


def to_native(value):
    """Convert numpy types to native Python so they are JSON serializable."""
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def predict_one(features: dict) -> dict:
    """Predict bankruptcy for a single company.

    `features` must be a dict mapping feature name -> value. The model was
    trained on the 94 features listed in `columns` (column names stripped,
    'Net Income Flag' removed).
    """
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"missing {len(missing)} required feature(s): {missing[:5]} ...")

    # every value must be a finite number -> otherwise return a clean 400
    row = []
    for c in columns:
        try:
            value = float(features[c])
        except (TypeError, ValueError):
            raise ValueError(
                f"feature '{c}' must be a number "
                f"(got {features[c]!r}: null, strings and other non-numeric values are not allowed)"
            )
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"feature '{c}' must be a finite number (got {value})")
        row.append(value)

    x = pd.DataFrame([row], columns=columns)
    x_scaled = scaler.transform(x)

    probability = model.predict_proba(x_scaled)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "bankruptcy_probability": to_native(round(float(probability), 4)),
        "bankruptcy_prediction": prediction,
        "model": model_name,
    }


@app.route("/ping", methods=["GET"])
def ping():
    return "PONG"


@app.route("/", methods=["GET"])
def demo():
    """Small demo page: lets you call /predict with real sample companies."""
    return DEMO_PAGE.format(samples_json=json.dumps(samples))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "request body must be valid JSON"}), 400

    # accept a single record, a list of records, or {"records": [...]}
    if isinstance(data, dict) and "records" in data:
        data = data["records"]

    if isinstance(data, list):
        try:
            results = [predict_one(record) for record in data]
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"predictions": results})

    if isinstance(data, dict):
        try:
            return jsonify(predict_one(data))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    return jsonify({"error": "unsupported payload type"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)
