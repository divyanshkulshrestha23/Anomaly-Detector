"""Train and compare fraud detectors on the credit card dataset.

Models:
    isolation_forest      unsupervised anomaly detector (learns normal transactions only)
    logistic_regression   supervised baseline; coefficients show which features push toward fraud
    gradient_boosting     supervised model; usually the strongest of the three

Expects the dataset at data/creditcard.csv (not committed; see README).
Run from the repository root:
    python model/train.py

Split (all by Time, so evaluation uses transactions later than training):
    earliest 70%  -> fit (80%) and validation (20%)
    latest 30%    -> holdout, used only for the final report

Alert thresholds are set on the validation slice and applied to the holdout set.

Writes:
    model/<name>.joblib   each fitted model with its feature order and thresholds
    model/metrics.json    holdout metrics for every model and alert rate
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TRAIN_FRACTION = 0.7
FIT_FRACTION_OF_TRAIN = 0.8
ALERT_RATES = (0.002, 0.005, 0.01)  # share of transactions flagged
DEFAULT_ALERT_RATE = 0.005

FEATURES = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
LABEL = "Class"

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "metrics.json"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (fit, validation, holdout), split in time order."""
    df = pd.read_csv(DATA_PATH).sort_values("Time").reset_index(drop=True)
    train_cut = int(len(df) * TRAIN_FRACTION)
    train, holdout = df.iloc[:train_cut], df.iloc[train_cut:]
    fit_cut = int(len(train) * FIT_FRACTION_OF_TRAIN)
    return train.iloc[:fit_cut], train.iloc[fit_cut:], holdout


def fit_isolation_forest(fit: pd.DataFrame):
    """Fit on normal transactions only. Higher anomaly score means more suspicious."""
    normal = fit[fit[LABEL] == 0]
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED)
    model.fit(normal[FEATURES])

    def score(X: pd.DataFrame) -> np.ndarray:
        return -model.score_samples(X)  # score_samples is higher for normal points

    return score, model


def fit_logistic_regression(fit: pd.DataFrame):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )
    model.fit(fit[FEATURES], fit[LABEL])

    def score(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    return score, model


def fit_gradient_boosting(fit: pd.DataFrame):
    model = HistGradientBoostingClassifier(random_state=SEED)
    model.fit(fit[FEATURES], fit[LABEL])

    def score(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    return score, model


def evaluate(score_fn, validation: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    """Thresholds come from validation scores; the holdout set is only scored, never tuned on."""
    val_scores = score_fn(validation[FEATURES])
    thresholds = {rate: float(np.quantile(val_scores, 1 - rate)) for rate in ALERT_RATES}

    hold_scores = score_fn(holdout[FEATURES])
    y = holdout[LABEL].to_numpy()

    by_rate = {}
    for rate, threshold in thresholds.items():
        flagged = hold_scores >= threshold
        by_rate[str(rate)] = {
            "alert_rate_actual": round(float(flagged.mean()), 5),
            "precision": round(float(precision_score(y, flagged, zero_division=0)), 4),
            "recall": round(float(recall_score(y, flagged, zero_division=0)), 4),
            "threshold": threshold,
        }

    return {
        "pr_auc": round(float(average_precision_score(y, hold_scores)), 4),
        "by_alert_rate": by_rate,
        "thresholds": thresholds,
    }


def top_coefficients(logreg_pipeline, n: int = 5) -> list[dict]:
    """Largest logistic regression coefficients by magnitude, on standardized features."""
    coefs = logreg_pipeline.named_steps["logisticregression"].coef_[0]
    order = np.argsort(-np.abs(coefs))[:n]
    return [{"feature": FEATURES[i], "coefficient": round(float(coefs[i]), 4)} for i in order]


def main() -> None:
    fit, validation, holdout = load_splits()

    builders = {
        "isolation_forest": fit_isolation_forest,
        "logistic_regression": fit_logistic_regression,
        "gradient_boosting": fit_gradient_boosting,
    }

    results = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in builders.items():
        score_fn, fitted = build(fit)
        report = evaluate(score_fn, validation, holdout)
        results[name] = report
        joblib.dump(
            {
                "model": fitted,
                "features": FEATURES,
                "thresholds": report["thresholds"],
                "default_alert_rate": DEFAULT_ALERT_RATE,
            },
            MODEL_DIR / f"{name}.joblib",
        )

    results["logistic_regression"]["top_coefficients"] = top_coefficients(
        joblib.load(MODEL_DIR / "logistic_regression.joblib")["model"]
    )

    metrics = {
        "fit_rows": len(fit),
        "validation_rows": len(validation),
        "holdout_rows": len(holdout),
        "holdout_frauds": int(holdout[LABEL].sum()),
        "holdout_fraud_rate": round(float(holdout[LABEL].mean()), 6),
        "default_alert_rate": DEFAULT_ALERT_RATE,
        "models": results,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved models and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
