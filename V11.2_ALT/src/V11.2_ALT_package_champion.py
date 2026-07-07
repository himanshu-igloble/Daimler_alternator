"""
V11.2_ALT_package_champion.py — package the frozen ALT champion (V10.5.3 ridge:
alpha=1.0, 6 features, Youden threshold 0.4456, LOVO AUROC 0.9267; validated by
the V11.2 dossier) into a loadable joblib bundle under
V11.2_ALT/models/V10.5.3_ridge_frozen_champion/.

Parity gates (script aborts on any failure):
  P1  25-fold LOVO re-run (exact manual-impute replica) AUROC rounds to 0.9267
  P2  per-VIN LOVO probs match frozen ridge_predictions.csv within 6e-5 (4-dp CSV)
  P3  SimpleImputer(median) statistics == np.nanmedian on the all-25 fit (1e-12)
  P4  joblib round-trip returns bit-identical decision values

Outputs -> V11.2_ALT/models/V10.5.3_ridge_frozen_champion/:
  V10.5.3_20_5_ALT_champion_bundle.joblib
  V10.5.3_20_5_ALT_training_matrix.csv      (provenance copy)
  V10.5.3_20_5_ALT_ridge_spec.json          (provenance copy)
  V10.5.3_20_5_ALT_lovo_predictions.csv     (provenance copy)
  V11.2_ALT_metric_suite.json               (provenance copy — V11.2 validation)
  V10.5.3_20_5_ALT_verification.json
  V10.5.3_20_5_ALT_MANIFEST.json

Run: py -3 "V11.2_ALT/src/V11.2_ALT_package_champion.py"
"""
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"D:\Daimler-starter_motor_alternator_battery")
MATRIX_CSV = ROOT / "V5.2_ALT" / "features" / "V5.2_20_5_ALT_selected_features.csv"
SPEC_JSON = ROOT / "V5.2_ALT" / "models" / "classification" / "V10.5.3_20_5_ALT_ridge_spec.json"
FROZEN_PREDS = ROOT / "V5.2_ALT" / "results" / "V10.5.3_20_5_ALT_ridge_predictions.csv"
V112_SUITE = ROOT / "V11.2_ALT" / "results" / "V11.2_ALT_metric_suite.json"
OUT = ROOT / "V11.2_ALT" / "models" / "V10.5.3_ridge_frozen_champion"

FEATURES = [
    "vsi_std_ratio_30d", "vsi_dominant_freq", "vsi_spectral_entropy",
    "bat_charge_delta_trend_right", "vsi_range_trend_last30d", "progressive_drift",
]
RIDGE_ALPHA = 1.0
RANDOM_SEED = 42
THRESHOLD = 0.4456          # frozen Youden from the original LOVO — never recompute
FROZEN_AUROC = 0.9267
TIER_AMBER, TIER_RED = 0.35, 0.55
PROB_TOL = 6e-5             # frozen CSV probs are rounded to 4 dp


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def impute_median_from_train(X_train, X_test):
    """Exact replica of the original script's manual impute (NaN median -> 0.0)."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for j in range(X_tr.shape[1]):
        med = np.nanmedian(X_tr[:, j])
        if np.isnan(med):
            med = 0.0
        X_tr[np.isnan(X_tr[:, j]), j] = med
        X_te[np.isnan(X_te[:, j]), j] = med
    return X_tr, X_te


def lovo_probs(X_raw, y):
    """Exact replica of lovo_ridge() in V10.5.3_20_5_ALT_ridge_optimized.py."""
    n = len(y)
    probs = np.full(n, np.nan)
    for i in range(n):
        tr = np.concatenate([np.arange(0, i), np.arange(i + 1, n)])
        X_tr, X_te = impute_median_from_train(X_raw[tr], X_raw[i:i + 1])
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        model = RidgeClassifier(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)
        model.fit(X_tr_sc, y[tr])
        probs[i] = sigmoid(model.decision_function(X_te_sc))[0]
    return probs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MATRIX_CSV)
    assert len(df) == 25 and int(df["failed"].sum()) == 10, "frozen matrix changed"
    missing = [f for f in FEATURES if f not in df.columns]
    assert not missing, f"champion features missing from matrix: {missing}"
    X = df[FEATURES].values.astype(float)
    y = df["failed"].values.astype(int)
    assert all(np.isfinite(X[:, j]).any() for j in range(X.shape[1])), \
        "all-NaN champion column — SimpleImputer equivalence would break"

    # ── P1 + P2: LOVO parity vs frozen predictions ──────────────────────────
    print("[P1] 25-fold LOVO parity re-run ...")
    probs = lovo_probs(X, y)
    auroc = float(roc_auc_score(y, probs))
    print(f"     LOVO AUROC = {auroc:.6f} (frozen {FROZEN_AUROC})")
    assert round(auroc, 4) == FROZEN_AUROC, f"P1 FAIL: {auroc:.6f}"

    frozen = pd.read_csv(FROZEN_PREDS)
    merged = df[["truck_id"]].assign(prob_new=probs).merge(
        frozen[["VIN_LABEL", "ridge_prob"]],
        left_on="truck_id", right_on="VIN_LABEL", validate="1:1")
    assert len(merged) == 25, "VIN alignment with frozen predictions failed"
    max_diff = float(np.max(np.abs(merged["prob_new"] - merged["ridge_prob"])))
    print(f"[P2] max |prob - frozen ridge_prob| = {max_diff:.2e} (tol {PROB_TOL})")
    assert max_diff <= PROB_TOL, f"P2 FAIL: {max_diff:.2e}"

    # ── production fit on all 25 ────────────────────────────────────────────
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=RIDGE_ALPHA, random_state=RANDOM_SEED)),
    ])
    pipe.fit(X, y)
    med_check = np.nanmedian(X, axis=0)
    assert np.allclose(pipe.named_steps["impute"].statistics_, med_check, atol=1e-12), \
        "P3 FAIL: SimpleImputer != nanmedian"
    print("[P3] SimpleImputer(median) == np.nanmedian  OK")
    dec_train = pipe.decision_function(X)

    # ── bundle ──────────────────────────────────────────────────────────────
    spec = json.loads(SPEC_JSON.read_text())
    env = {"python": platform.python_version(), "sklearn": sklearn.__version__,
           "numpy": np.__version__, "pandas": pd.__version__,
           "joblib": joblib.__version__, "platform": platform.platform()}
    bundle = {
        "component": "alternator",
        "champion_version": "V10.5.3_20_5_ALT",
        "validated_by": "V11.2_ALT dossier (metric suite 139/150 pairs, PR-AUC 0.94); "
                        "frozen by V11.1; unbeaten through V12 GED challenge",
        "created": date.today().isoformat(),
        "features": FEATURES,
        "pipeline": pipe,
        "score_mapping": "prob = sigmoid(pipeline.decision_function(X[features]))",
        "threshold": THRESHOLD,
        "tier_bands": {"amber_ge": TIER_AMBER, "red_ge": TIER_RED},
        "tier_score": "prob_raw",
        "frozen_metrics": {"lovo_auroc": spec["lovo_auroc"],
                           "lovo_recall": spec["lovo_recall"],
                           "lovo_specificity": spec["lovo_specificity"],
                           "threshold_youden": spec["threshold_youden"]},
        "training": {"matrix": str(MATRIX_CSV.relative_to(ROOT)),
                     "matrix_sha256": sha256(MATRIX_CSV),
                     "n_trucks": 25, "n_failed": 10,
                     "fit_scope": "all 25 trucks (production fit; LOVO was evaluation-only)",
                     "git_head": git_head()},
        "environment": env,
    }
    bundle_path = OUT / "V10.5.3_20_5_ALT_champion_bundle.joblib"
    joblib.dump(bundle, bundle_path)

    # ── P4: round trip ──────────────────────────────────────────────────────
    b2 = joblib.load(bundle_path)
    dec_2 = b2["pipeline"].decision_function(X)
    assert np.array_equal(dec_train, dec_2), "P4 FAIL: round-trip drift"
    print("[P4] joblib round-trip bit-identical  OK")

    # ── provenance copies ───────────────────────────────────────────────────
    shutil.copy2(MATRIX_CSV, OUT / "V10.5.3_20_5_ALT_training_matrix.csv")
    shutil.copy2(SPEC_JSON, OUT / "V10.5.3_20_5_ALT_ridge_spec.json")
    shutil.copy2(FROZEN_PREDS, OUT / "V10.5.3_20_5_ALT_lovo_predictions.csv")
    shutil.copy2(V112_SUITE, OUT / "V11.2_ALT_metric_suite.json")

    # ── verification + manifest ─────────────────────────────────────────────
    verification = {
        "created": date.today().isoformat(),
        "P1_lovo_auroc": {"value": round(auroc, 6), "frozen": FROZEN_AUROC, "pass": True},
        "P2_prob_parity": {"max_abs_diff": max_diff, "tol": PROB_TOL, "pass": True},
        "P3_imputer_equivalence": {"atol": 1e-12, "pass": True},
        "P4_roundtrip": {"pass": True},
        "environment": env,
    }
    (OUT / "V10.5.3_20_5_ALT_verification.json").write_text(
        json.dumps(verification, indent=2))

    files = sorted(p for p in OUT.iterdir()
                   if p.is_file() and p.name != "V10.5.3_20_5_ALT_MANIFEST.json")
    manifest = {
        "artifact": "ALT frozen champion: V10.5.3 ridge, V11.2-validated",
        "created": date.today().isoformat(),
        "git_head": git_head(),
        "environment": env,
        "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in files],
        "inputs": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                   for p in (MATRIX_CSV, SPEC_JSON, FROZEN_PREDS, V112_SUITE)],
    }
    (OUT / "V10.5.3_20_5_ALT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nPACKAGED OK -> {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
