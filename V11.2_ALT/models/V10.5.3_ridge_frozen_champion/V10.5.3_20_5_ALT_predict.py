"""
V10.5.3_20_5_ALT_predict.py — self-contained loader for the frozen ALT champion
(V10.5.3 ridge, V11.2-validated).

Library use:
    bundle = load_bundle()
    out = predict(df, bundle)   # df must contain the 6 champion feature columns

CLI:
    py -3 V10.5.3_20_5_ALT_predict.py <features_csv> [--out <predictions_csv>]

Requires only: numpy, pandas, scikit-learn, joblib (packaged with sklearn 1.8.0).
NaNs in feature columns are fine — the pipeline imputes with TRAINING medians.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE_PATH = HERE / "V10.5.3_20_5_ALT_champion_bundle.joblib"


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def load_bundle(path=BUNDLE_PATH):
    return joblib.load(path)


def predict(df, bundle=None):
    """Score trucks. df: DataFrame containing the champion feature columns.
    Returns DataFrame with prob_raw, predicted_class, alert_tier."""
    if bundle is None:
        bundle = load_bundle()
    feats = bundle["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    X = df[feats].values.astype(float)
    prob = sigmoid(bundle["pipeline"].decision_function(X))
    thr = bundle["threshold"]
    bands = bundle["tier_bands"]
    tier = np.where(prob >= bands["red_ge"], "red",
                    np.where(prob >= bands["amber_ge"], "amber", "green"))
    out = pd.DataFrame({
        "VIN": df["truck_id"] if "truck_id" in df.columns else np.arange(len(df)),
        "prob_raw": np.round(prob, 4),
        "predicted_class": (prob >= thr).astype(int),
        "alert_tier": tier,
        "threshold": thr,
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_csv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.features_csv)
    out = predict(df)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out} ({len(out)} rows)")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
