"""Train the anomaly detector on synthetic data and save it to model/anomaly_model.joblib.

Run from the repository root:
    python model/train.py
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

SEED = 42
N_SAMPLES = 5000
CONTAMINATION = 0.02
FEATURES = ("temperature_c", "fan_rpm", "power_w")
NORMAL_MEAN = np.array([45.0, 3000.0, 250.0])
NORMAL_STD = np.array([3.0, 150.0, 15.0])

OUTPUT_PATH = Path(__file__).parent / "anomaly_model.joblib"


def make_normal_readings(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate synthetic sensor readings that represent normal operation."""
    return rng.normal(loc=NORMAL_MEAN, scale=NORMAL_STD, size=(n, len(FEATURES)))


def train(output_path: Path = OUTPUT_PATH) -> IsolationForest:
    rng = np.random.default_rng(SEED)
    data = make_normal_readings(N_SAMPLES, rng)

    model = IsolationForest(contamination=CONTAMINATION, random_state=SEED)
    model.fit(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return model


if __name__ == "__main__":
    trained = train()
    print(f"Saved model to {OUTPUT_PATH}")
    print(f"Features: {', '.join(FEATURES)}")
