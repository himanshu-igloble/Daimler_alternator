"""
V11.2_ALT — data-driven honesty audit of the per-VIN evidence-stack figures/guide.

For every VIN it (a) checks figure inputs against the AUTHORITATIVE source files
(no trust in build_bundle's own reads), and (b) stress-tests the *derived* claims —
especially precursor "lead" honesty (is the breach a genuine late-life deterioration
that is still elevated at failure, or an early-life baseline artifact?). Run: py -3
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
import V11_2_ALT_rul_evidence_stack as E

ALL = [f"VIN{i}_F_ALT" for i in range(1, 11)] + [f"VIN{i}_NF_ALT" for i in range(1, 16)]

# authoritative sources
LC = pd.read_parquet(E.cfg.LIFECYCLE_PARQUET).set_index("vin_label")
FR = pd.read_csv(os.path.join(E.cfg.RUL_CACHE, "final_rul_per_vin.csv")).set_index("vin_label")
EMG = pd.read_csv(os.path.join(E.cfg.EMERG_CACHE, "emergency_per_vin.csv")).set_index("vin_label")
FW = json.loads(open(os.path.join(ROOT, "V10.6.2_ALT/cache/rul/fleet_window.json")).read())

fidelity_fail, honesty_flags = [], []
prec_rows = []

for vin in ALL:
    b = E.build_bundle(vin)
    # ---- (a) external-value fidelity ----
    checks = {
        "failed": b["failed"] == bool(LC.loc[vin, "failed_flag"]),
        "risk_band": b["risk_band"] == str(FR.loc[vin, "risk_band"]),
        "median_rul": abs(b["median_rul"] - float(FR.loc[vin, "median_rul_days"])) < 1e-6,
        "km_per_day": abs(b["km_per_day"] - float(FR.loc[vin, "km_per_day_est"])) < 1e-6,
        "fleet_med": abs(b["fleet_med"] - float(FW["median_ttf_days"])) < 1e-6,
        "ged_fired": b["ged_fired"] == (bool(EMG.loc[vin, "ged_fired"]) if vin in EMG.index else False),
    }
    if b["failed"]:
        jc_par = pd.Timestamp(str(b["daily"]  # JCOPENDATE crosscheck via common + parquet already in bundle
                                  ).__class__ and E.JCOPENDATE.get(vin))
        checks["jc_matches_common"] = (b["jc"] is not None and str(b["jc"].date()) == E.JCOPENDATE[vin])
        if b["ged_fired"]:
            checks["ged_lead"] = b["ged_lead"] == int(EMG.loc[vin, "ged_lead_days"])
    for k, ok in checks.items():
        if not ok:
            fidelity_fail.append((vin, k, b.get(k, "?")))

    # ---- (b) invariants ----
    rm = b["rul_med"]
    if not np.all(np.diff(rm) <= 1e-6):
        honesty_flags.append((vin, "RUL median not monotone non-increasing"))
    if not (np.nanmin(b["m5"]) >= -1e-9 and np.nanmax(b["m5"]) <= 1.0 + 1e-9):
        honesty_flags.append((vin, "M5 out of [0,1]"))

    # ---- (c) precursor lead honesty ----
    end_i = len(b["daily"]) - 1
    for col, info in b["prec"].items():
        if info["breach"] is None:
            continue
        z = np.asarray(info["z"], float); bi = info["breach"]
        lead = (b["end"] - pd.Timestamp(b["daily"]["date"].values[bi])).days
        z_end = float(z[end_i])
        frac_sustained = float(np.mean(z[bi:] >= 1.0)) if end_i >= bi else 0.0
        # genuine late precursor: still elevated near failure OR sustained majority of the way
        genuine = (z_end >= 1.5) or (frac_sustained >= 0.6)
        verdict = "genuine" if genuine else "EARLY-ARTIFACT"
        prec_rows.append(dict(vin=b["dname"], failed=b["failed"], signal=info["label"],
                              lead=int(lead), z_end=round(z_end, 2),
                              sustained=round(frac_sustained, 2), verdict=verdict))

    # M5 transition transience: did it recover below the entered zone by the end?
    if b["m5_trans"]:
        worst = max(E._ZORDER[t["to"]] for t in b["m5_trans"])
        final = E._ZORDER[E.zone_of_m5(float(b["m5"][-1]))]
        if final < worst:
            honesty_flags.append((b["dname"], f"M5 recovered: worst zone reached then ended lower "
                                              f"(worst idx {worst} -> final {final}) — transition may be transient"))

# ---- report ----
print("=" * 78)
print("DATA-FIDELITY (figure inputs vs authoritative source files)")
print("=" * 78)
print("FAILURES:", fidelity_fail if fidelity_fail else "NONE — all 25 figures use correct source values")

print("\n" + "=" * 78)
print("GED2 STORM (authoritative emergency channel)")
print("=" * 78)
gf = [(v, int(EMG.loc[v, "ged_lead_days"])) for v in ALL if v in EMG.index and bool(EMG.loc[v, "ged_fired"])]
print("fired:", gf, "| expected exactly VIN1_F(21), VIN10_F(1)")

print("\n" + "=" * 78)
print("PRECURSOR LEAD HONESTY  (per breaching signal)")
print("=" * 78)
pr = pd.DataFrame(prec_rows)
if len(pr):
    pr = pr.sort_values(["failed", "lead"], ascending=[False, False])
    print(pr.to_string(index=False))
    art = pr[pr.verdict == "EARLY-ARTIFACT"]
    print(f"\nEARLY-ARTIFACT breaches (long lead but NOT elevated at end => misleading lead): {len(art)}")
    # which FAILED VINs have their *reported first-deviation* (longest lead) be an artifact?
    bad = []
    for v in pr[pr.failed].vin.unique():
        sub = pr[pr.vin == v].sort_values("lead", ascending=False)
        if len(sub) and sub.iloc[0].verdict == "EARLY-ARTIFACT":
            bad.append((v, sub.iloc[0].signal, int(sub.iloc[0].lead), float(sub.iloc[0].z_end)))
    print("FAILED VINs whose headline 'first deviation' is an early-artifact:", bad if bad else "none")

print("\n" + "=" * 78)
print("OTHER HONESTY FLAGS")
print("=" * 78)
for v, msg in honesty_flags:
    print(f"  {v}: {msg}")
if not honesty_flags:
    print("  none")
