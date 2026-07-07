"""Task 2 — LOVO noise-null distribution (R3 independent recompute).

200 single-noise-column LOVO trials  → single-feature null distribution
100 additive noise trials            → delta null relative to FAMILY_A baseline

Writes: V12_ALT_GED/results/r3_lovo_null.json
"""
import sys
import json
import pathlib
import importlib.util

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

_SRC = pathlib.Path(__file__).resolve().parent
RESULTS = _SRC.parent / "results"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / f"{name}.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C    = _load("ged_common")
T4   = _load("trial_4_lovo")

FAMILY_A = T4.FAMILY_A
lovo_auroc = T4.lovo_auroc

# ── load baseline matrix ──────────────────────────────────────────────────────
base = pd.read_csv(C.ROOT / "V5.2_ALT" / "features" / "V5.2_20_5_ALT_selected_features.csv")
# apply the same join-key fix as trial_4_lovo.main()
if "VIN" not in base.columns or not base["VIN"].astype(str).str.contains("_ALT").any():
    for col in ("truck_id", "VIN_LABEL", "vin"):
        if col in base.columns and base[col].astype(str).str.contains("_ALT").any():
            if "VIN" in base.columns:
                base = base.drop(columns=["VIN"])
            base = base.rename(columns={col: "VIN"})
            break

df = base.copy()
# add a dummy failed column (truck_id already removed from baseline; failed present)
# ensure 'failed' dtype is int
df["failed"] = df["failed"].astype(int)

n_rows = len(df)
assert n_rows == 25, f"Expected 25 rows, got {n_rows}"

# ── baseline sanity ───────────────────────────────────────────────────────────
print("Computing baseline FAMILY_A AUROC (sanity check)...")
a0 = lovo_auroc(df, FAMILY_A)
print(f"  FAMILY_A baseline = {a0:.4f}  (expect ~0.9267)")

# ── single-noise 200 trials ───────────────────────────────────────────────────
TRIALS_SINGLE = 200
single_aurocs = []
print(f"Running {TRIALS_SINGLE} single-noise-column LOVO trials...")
for seed in range(TRIALS_SINGLE):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_rows)
    df2 = df.copy()
    df2["noise"] = noise
    a = lovo_auroc(df2, ["noise"])
    single_aurocs.append(a)
    if (seed + 1) % 50 == 0:
        print(f"  ... {seed + 1}/{TRIALS_SINGLE}  running mean={np.mean(single_aurocs):.4f}")

single_arr = np.array(single_aurocs)
single_stats = {
    "n_trials": TRIALS_SINGLE,
    "mean":  float(np.mean(single_arr)),
    "std":   float(np.std(single_arr)),
    "p5":    float(np.percentile(single_arr, 5)),
    "p25":   float(np.percentile(single_arr, 25)),
    "p50":   float(np.percentile(single_arr, 50)),
    "p75":   float(np.percentile(single_arr, 75)),
    "p95":   float(np.percentile(single_arr, 95)),
}
print()
print("SINGLE-NOISE NULL DISTRIBUTION:")
for k, v in single_stats.items():
    print(f"  {k:<12s} = {v:.4f}" if isinstance(v, float) else f"  {k:<12s} = {v}")

# ── additive-noise 100 trials ─────────────────────────────────────────────────
TRIALS_ADD = 100
add_deltas = []
print()
print(f"Running {TRIALS_ADD} additive-noise LOVO trials (FAMILY_A + noise)...")
for seed in range(TRIALS_ADD):
    rng = np.random.default_rng(seed + 10000)
    noise = rng.standard_normal(n_rows)
    df2 = df.copy()
    df2["noise"] = noise
    a = lovo_auroc(df2, FAMILY_A + ["noise"])
    add_deltas.append(a - a0)
    if (seed + 1) % 25 == 0:
        print(f"  ... {seed + 1}/{TRIALS_ADD}  running mean-delta={np.mean(add_deltas):+.4f}")

delta_arr = np.array(add_deltas)
additive_stats = {
    "n_trials":        TRIALS_ADD,
    "baseline_auroc":  float(a0),
    "mean_delta":      float(np.mean(delta_arr)),
    "p5_delta":        float(np.percentile(delta_arr, 5)),
    "p95_delta":       float(np.percentile(delta_arr, 95)),
}
print()
print("ADDITIVE-NOISE DELTA (FAMILY_A+noise minus baseline):")
for k, v in additive_stats.items():
    print(f"  {k:<20s} = {v:.4f}" if isinstance(v, float) else f"  {k:<20s} = {v}")

# calibration comment
print()
print("CALIBRATION INTERPRETATION:")
print(f"  Single-noise AUROC 50th pct = {single_stats['p50']:.4f}")
print(f"  => A single-feature AUROC of 0.11-0.16 sits {'BELOW chance' if single_stats['p50'] > 0.16 else 'at/above chance'} median")
print(f"  Additive mean delta = {additive_stats['mean_delta']:+.4f}")
print(f"  => Red-team's claimed delta of -0.0067 sits {'within' if additive_stats['p5_delta'] < -0.0067 < additive_stats['p95_delta'] else 'outside'} [p5,p95] range")

# ── write json ────────────────────────────────────────────────────────────────
result = {
    "single_noise_null": single_stats,
    "additive_noise_delta": additive_stats,
}
out_path = RESULTS / "r3_lovo_null.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print()
print(f"JSON written: {out_path}")
