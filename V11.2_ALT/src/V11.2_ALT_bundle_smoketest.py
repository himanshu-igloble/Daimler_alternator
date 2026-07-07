"""
V11.2_ALT_bundle_smoketest.py — load-and-predict smoke test for the packaged
ALT champion bundle (frozen V10.5.3 ridge, V11.2-validated).
Exits non-zero on any failure.

Run: py -3 "V11.2_ALT/src/V11.2_ALT_bundle_smoketest.py"
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"D:\Daimler-starter_motor_alternator_battery")
DEP = ROOT / "V11.2_ALT" / "models" / "V10.5.3_ridge_frozen_champion"
BUNDLE = DEP / "V10.5.3_20_5_ALT_champion_bundle.joblib"
MATRIX = DEP / "V10.5.3_20_5_ALT_training_matrix.csv"
PREDICT = DEP / "V10.5.3_20_5_ALT_predict.py"

EXPECT_FEATURES = [
    "vsi_std_ratio_30d", "vsi_dominant_freq", "vsi_spectral_entropy",
    "bat_charge_delta_trend_right", "vsi_range_trend_last30d", "progressive_drift",
]
EXPECT_KEYS = {"component", "champion_version", "validated_by", "created", "features",
               "pipeline", "score_mapping", "threshold", "tier_bands", "tier_score",
               "frozen_metrics", "training", "environment"}
MIN_RESUB_AUROC = 0.90


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def main():
    if not BUNDLE.exists():
        print(f"BUNDLE MISSING: {BUNDLE}")
        return 1

    b = joblib.load(BUNDLE)
    missing = EXPECT_KEYS - set(b)
    assert not missing, f"bundle missing keys: {missing}"
    assert b["champion_version"] == "V10.5.3_20_5_ALT"
    assert b["features"] == EXPECT_FEATURES, f"feature mismatch: {b['features']}"
    assert abs(b["threshold"] - 0.4456) < 1e-9, f"threshold drift: {b['threshold']}"
    assert b["frozen_metrics"]["lovo_auroc"] == 0.9267

    df = pd.read_csv(MATRIX)
    X = df[b["features"]].values.astype(float)
    y = df["failed"].values.astype(int)
    probs = sigmoid(b["pipeline"].decision_function(X))
    assert probs.shape == (25,) and np.all(np.isfinite(probs))
    assert np.all((probs >= 0) & (probs <= 1))
    resub_auroc = roc_auc_score(y, probs)
    # resubstitution is optimistic; must comfortably clear the LOVO estimate's floor
    assert resub_auroc > MIN_RESUB_AUROC, \
        f"resubstitution AUROC suspiciously low: {resub_auroc:.4f}"

    # CLI loader round-trip — prove the CLI's numeric output equals in-process.
    # Write to a temp CSV via --out and compare prob_raw against the in-process
    # probabilities (loader rounds prob_raw to 4 dp; row order matches MATRIX order).
    fd, cli_out = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        r = subprocess.run([sys.executable, str(PREDICT), str(MATRIX), "--out", cli_out],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"predict.py CLI failed:\n{r.stderr}"
        assert "VIN" in r.stdout and "tier" in r.stdout
        cli = pd.read_csv(cli_out)
        assert len(cli) == 25, f"CLI produced {len(cli)} rows, expected 25"
        cli_prob_raw = cli["prob_raw"].values.astype(float)
        assert np.allclose(cli_prob_raw, np.round(probs, 4), atol=1e-9), \
            "CLI prob_raw does not match in-process probabilities"
    finally:
        os.remove(cli_out)

    print(f"SMOKE PASS  resub_auroc={resub_auroc:.4f}  n=25  threshold={b['threshold']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
