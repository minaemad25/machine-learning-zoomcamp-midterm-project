# Company Bankruptcy Prediction

**ML Zoomcamp — Midterm Project**

Predict whether a company will go **bankrupt** based on its financial indicators.

---

## Description of the problem

Bankruptcy prediction (financial distress prediction) is a classic binary
classification problem in corporate finance. Using 94 financial ratios and
indicators — profitability, leverage, liquidity, and activity metrics —
extracted from the annual reports of Taiwanese companies, the goal is to
estimate the probability that a company will go bankrupt.

This matters in the real world for:

- **banks and investors**, who want to avoid lending to / investing in companies likely to default;
- **regulators and auditors**, who monitor systemic risk;
- **the companies themselves**, as an early-warning signal.

### The dataset

- **Source:** [Company Bankruptcy Prediction for Taiwanese Companies](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction-dataset) (Kaggle),
  based on the Taiwan Economic Journal for the years 1999–2009.
- **Size:** 6,819 companies × 96 columns (95 financial indicators + the target `Bankrupt?`).
- **Target:** `Bankrupt?` — `1` = bankrupt, `0` = financially stable.
- **Challenge:** the dataset is **highly imbalanced** — only **220 (3.2 %)** of the companies are bankrupt.

The dataset is included in this repository at [`data/company_bankruptcy.csv`](data/company_bankruptcy.csv).

If you prefer to download it yourself:
1. Go to the Kaggle page above (free account required).
2. Download and unzip `archive.zip`.
3. Place the CSV at `data/company_bankruptcy.csv`.

### Approach

1. **Data cleaning** — all column names start with a space (stripped);
   `Net Income Flag` is constant and carries no information (dropped);
   the dataset has no missing values and no duplicates.
2. **EDA** — Spearman correlation heatmap, box plots, class-distribution analysis,
   outlier analysis, and distributions of key indicators per class.
3. **Handling imbalance** — the classes are balanced by **downsampling**:
   all 220 bankrupt companies + 220 randomly sampled healthy companies.
4. **Model selection** — 8 models compared on the balanced dataset
   (MinMax-scaled, 75/25 train/test split, seed 74):
   SVC, KNN, Decision Tree, Random Forest, Logistic Regression,
   Gaussian Naive Bayes, Gradient Boosting and a sklearn MLP —
   each first with default parameters, then tuned with GridSearchCV /
   RandomizedSearchCV and compared on precision / recall / **F1** / **MCC**.

| Model (tuned)         | Precision | Recall | F1     | MCC     |
|-----------------------|-----------|--------|--------|---------|
| **Random Forest**     | 0.8103    | 0.8393 | **0.8246** | **0.6364** |
| Gradient Boosting     | 0.8070    | 0.8214 | 0.8142 | 0.6180  |
| Logistic Regression   | 0.7833    | 0.8393 | 0.8103 | 0.6009  |
| Decision Tree         | 0.7627    | 0.8036 | 0.7826 | 0.5457  |
| SVC                   | 0.7818    | 0.7679 | 0.7748 | 0.5455  |
| MLP                   | 0.7544    | 0.7679 | 0.7611 | 0.5089  |
| KNN                   | 0.7170    | 0.6786 | 0.6972 | 0.4010  |
| Gaussian NB           | 0.8571    | 0.3214 | 0.4675 | 0.3382  |

*(A PyTorch neural network was also explored in the notebook but did not outperform the tree ensembles.)*

5. **Final model** — a **tuned RandomForest**
   (`n_estimators=117, max_depth=30, min_samples_split=5, min_samples_leaf=5,
   max_features='sqrt', class_weight='balanced_subsample'`), selected by
   RandomizedSearchCV (60 iterations, 5-fold CV, F1 scoring).
   On the holdout set it reaches **F1 = 0.8246, MCC = 0.6364, ROC AUC = 0.8638**.

### Project structure

```
├── data/
│   └── company_bankruptcy.csv      # the dataset (committed)
├── Machine-Learning-Zoomcamp-Midtrem-Project.ipynb   # EDA + model selection + tuning
├── train.py                        # trains the final model, saves model.pkl
├── predict.py                      # Flask web service (loads model.pkl)
├── test_sample.json                # ready-to-use prediction requests
├── requirements.txt                # dependencies
├── Dockerfile                      # containerizes the service (trains + serves)
├── deployment/                     # screenshots of interacting with the service
│   ├── demo_bankrupt_prediction.png
│   └── demo_healthy_prediction.png
├── deployment_evidence.md          # transcript of a live test of the service
└── README.md
```

---

## Instructions on how to run the project

### 1. Locally (without Docker)

**Requirements:** Python 3.10+ (tested on 3.14).

```bash
# 1. clone / download the project and enter the folder
cd backrupcty

# 2. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate            # Windows   (use `. .venv/bin/activate` on Linux/macOS)

# 3. install the dependencies
pip install -r requirements.txt

# 4. train the final model (creates model.pkl)
python train.py

# 5. start the web service (listens on port 9696)
python predict.py
```

> To re-run the notebook you additionally need Jupyter: `pip install jupyter`.

### 2. With Docker

```bash
docker build -t bankruptcy-prediction .
docker run -it --rm -p 9696:9696 bankruptcy-prediction
```

The image trains the model during the build (`RUN python train.py`),
so no pickle needs to be committed, then serves it with gunicorn.

### 3. Testing the service

Once the service is running (locally or in Docker):

- open the **demo page** at `http://localhost:9696/` in a browser — it has
  buttons that send real bankrupt / healthy company payloads to `/predict`
  and show the raw response (screenshots: [`deployment/`](deployment/));
- or use the API directly:

```bash
# health check
curl http://localhost:9696/ping
# -> PONG

# predict a single company: extract one payload from test_sample.json and POST it
python -c "import json;print(json.dumps(json.load(open('test_sample.json'))['single_bankrupt_company']))" | \
    curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @-

# batch: two companies at once
python -c "import json;s=json.load(open('test_sample.json'));print(json.dumps({'records':[s['single_bankrupt_company'],s['single_healthy_company']]}))" | \
    curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d @-
```

`test_sample.json` contains ready-to-use payloads:

- `single_bankrupt_company` — a real bankrupt company from the dataset;
- `single_healthy_company` — a real healthy company from the dataset;
- `_expected` — their true labels for reference.

You can also send a **batch** of companies or an incomplete payload to see
the validation error — full request/response examples are in
[`deployment_evidence.md`](deployment_evidence.md).

**Python example:**

```python
import json
import urllib.request

sample = json.load(open("test_sample.json"))["single_bankrupt_company"]

req = urllib.request.Request(
    "http://localhost:9696/predict",
    json.dumps(sample).encode(),
    {"Content-Type": "application/json"},
)
print(urllib.request.urlopen(req).read().decode())
# {"bankruptcy_probability": 1.0, "bankruptcy_prediction": 1, "model": "RandomForestClassifier (tuned)"}
```

**Response format:**

```json
{
  "bankruptcy_probability": 0.87,   // P(bankrupt) from the Random Forest
  "bankruptcy_prediction": 1,       // 1 = bankrupt (probability >= 0.5), 0 = stable
  "model": "RandomForestClassifier (tuned)"
}
```

---

## Deployment

The service is containerized with the included `Dockerfile`, so it can be
deployed on any platform that runs Docker images
(Render, Google Cloud Run, AWS, Azure, a VPS, …).

**Deploying to Render (free):**
1. Push this repository to GitHub.
2. On [render.com](https://render.com) choose **New → Web Service** and connect the repo
   (Render auto-detects the `Dockerfile`; environment: *Docker*).
3. Expose port **9696** (the `Dockerfile` already handles the rest) and deploy.

**Deployment status:**

- **Service URL:** `http://localhost:9696` (locally) — *add your public cloud URL here after deploying*
- **Evidence of interaction with the running service:**
  - screenshots of the service's demo page making live `/predict` calls:
    [`deployment/demo_bankrupt_prediction.png`](deployment/demo_bankrupt_prediction.png)
    (real bankrupt company → `bankruptcy_prediction: 1`, probability `1.0`) and
    [`deployment/demo_healthy_prediction.png`](deployment/demo_healthy_prediction.png)
    (real healthy company → `bankruptcy_prediction: 0`, probability `0.0`);
  - transcript of real HTTP requests (`/ping`, single and batch `/predict`
    calls, and a validation-error case): [`deployment_evidence.md`](deployment_evidence.md).

---

## Notes on methodology

- **Why downsample instead of oversampling / SMOTE?** It keeps training fast,
  avoids synthetic samples, and — most importantly — it is the approach used
  consistently across the whole model comparison in the notebook.
  The cost is losing information from the ~6,600 healthy companies, which is
  why the holdout metrics above (not training accuracy) are the reference numbers.
- **Why F1 / MCC?** With a 3.2 % positive class, accuracy is misleading;
  F1 (and MCC) measure the balance between catching bankruptcies (recall)
  and avoiding false alarms (precision).
- **Feature importance** of the final Random Forest highlights
  `Non-industry income and expenditure/revenue`,
  `Net profit before tax/Paid-in capital` and
  `Persistent EPS in the Last Four Seasons` as the strongest predictors.
