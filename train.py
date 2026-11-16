"""
Training script for the Company Bankruptcy Prediction project.

Replicates the final pipeline of Machine-Learning-Zoomcamp-Midtrem-Project.ipynb:
  1. load the dataset (data/company_bankruptcy.csv)
  2. clean it (strip column names, drop the constant 'Net Income Flag')
  3. build a balanced training set (all 220 bankrupt + 220 random healthy companies)
  4. split into train/test, scale with MinMaxScaler
  5. train the tuned RandomForest (best model of the notebook's comparison)
  6. report holdout metrics, retrain on the full balanced data
  7. save model + scaler + feature names to model.pkl

Run:
    python train.py
"""

import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

DATA_PATH = "data/company_bankruptcy.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 74

# best hyperparameters found by RandomizedSearchCV in the notebook
BEST_PARAMS = {
    "n_estimators": 117,
    "max_depth": 30,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
}


def load_and_clean(path: str) -> pd.DataFrame:
    """Load the raw data and apply the cleaning steps from the notebook."""
    df = pd.read_csv(path)

    # every column name starts with a space -> strip it
    df.columns = df.columns.str.strip()

    # 'Net Income Flag' is constant (1 for every company) -> no information
    if "Net Income Flag" in df.columns:
        df = df.drop(columns=["Net Income Flag"])

    return df


def make_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance the classes by downsampling, exactly like the notebook:
    keep every bankrupt company and sample an equal number of healthy ones."""
    bankrupt = df[df["Bankrupt?"] == 1]
    healthy = df[df["Bankrupt?"] == 0].sample(frac=1, random_state=RANDOM_STATE)[
        : len(bankrupt)
    ]
    return pd.concat([bankrupt, healthy], axis=0).sample(frac=1, random_state=RANDOM_STATE)


def evaluate(model, scaler, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the metrics used in the notebook on a holdout set."""
    x_test_scaled = scaler.transform(x_test)
    y_pred = model.predict(x_test_scaled)
    y_prob = model.predict_proba(x_test_scaled)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "mcc": round(matthews_corrcoef(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }
    print(classification_report(y_test, y_pred))
    for name, value in metrics.items():
        print(f"{name:10s}: {value}")
    return metrics


def main() -> None:
    print(f"Loading data from {DATA_PATH} ...")
    df = load_and_clean(DATA_PATH)
    print(f"Dataset: {df.shape[0]} companies, {df.shape[1] - 1} features")

    balanced_df = make_balanced_dataset(df)
    print(f"Balanced dataset: {len(balanced_df)} companies "
          f"({balanced_df['Bankrupt?'].sum()} bankrupt / "
          f"{(balanced_df['Bankrupt?'] == 0).sum()} healthy)")

    y = balanced_df["Bankrupt?"]
    x = balanced_df.drop(columns=["Bankrupt?"])

    # holdout split for an honest estimate of the generalization error
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=RANDOM_STATE
    )

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(**BEST_PARAMS)
    model.fit(x_train_scaled, y_train)

    print("\n--- Tuned RandomForest on the holdout set (110 companies) ---")
    metrics = evaluate(model, scaler, x_test, y_test)

    print("\nTop 10 most important features:")
    importances = pd.Series(model.feature_importances_, index=x.columns)
    print(importances.sort_values(ascending=False).head(10).round(4))

    # retrain the final model on ALL balanced data so the deployed service
    # uses every available balanced sample
    print("\nRetraining the final model on the full balanced dataset ...")
    scaler_final = MinMaxScaler()
    x_scaled = scaler_final.fit_transform(x)

    final_model = RandomForestClassifier(**BEST_PARAMS)
    final_model.fit(x_scaled, y)

    artifact = {
        "model": final_model,
        "scaler": scaler_final,
        "columns": list(x.columns),
        "model_name": "RandomForestClassifier (tuned)",
        "holdout_metrics": metrics,
        "threshold": 0.5,
    }
    with open(MODEL_PATH, "wb") as f_out:
        pickle.dump(artifact, f_out)

    print(f"Final model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
