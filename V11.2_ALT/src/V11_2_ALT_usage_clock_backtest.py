"""
V11.2_ALT usage-clock RUL prototype + leave-one-VIN-out (LOVO) backtest.

Question: does a per-truck USAGE clock (km-to-failure, converted to calendar days via each
truck's km/day) predict remaining life better than the fleet CALENDAR clock (days-to-failure)?

Honest design:
  * LOVO: when scoring truck i, fit the Weibull on the OTHER 9 failures only (no self-leak).
  * Three INDEPENDENT inputs so the conversion is not trivially circular:
        TTF_days = JCOPENDATE - t0     (calendar, from dates)
        KTF_km   = est_km              (odometer / usage estimate)
        kpd      = km_per_day_est      (separate per-truck rate estimate)
  * Predict conditional MEDIAN remaining at install (f=0) and mid-life (f=0.5, 0.8).
        calendar:  remaining_days = condMedRemaining(TTF_weibull | age   > f*TTF_i)
        usage:     remaining_km   = condMedRemaining(KTF_weibull | usage > f*KTF_i) ; /kpd_i -> days
  * Actual remaining = (1-f)*TTF_i.  Metric = MAE over the 10 failures.

The deciding diagnostic is CV(KTF) vs CV(TTF): the usage clock can only win if trucks fail at a
more consistent MILEAGE than a more consistent AGE.

Run: py -3
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
from scipy.stats import weibull_min

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
sys.path.insert(0, os.path.join(ROOT, "V11.2_ALT", "src"))
import V11_2_ALT_rul_evidence_stack as E

cfg = E.cfg
OUTD = os.path.join(ROOT, "V11.2_ALT", "results", "usage_clock")
os.makedirs(OUTD, exist_ok=True)
REPORT = os.path.join(ROOT, "V11.2_ALT", "reports", "V11.2_ALT_usage_clock_backtest.md")
FRACS = (0.0, 0.5, 0.8)


def load_failed():
    lc = pd.read_parquet(cfg.LIFECYCLE_PARQUET).set_index("vin_label")
    fr = pd.read_csv(os.path.join(cfg.RUL_CACHE, "final_rul_per_vin.csv")).set_index("vin_label")
    rows = []
    for vin in E.ALL_VINS:
        if not bool(lc.loc[vin, "failed_flag"]) or vin not in E.JCOPENDATE:
            continue
        t0 = pd.Timestamp(lc.loc[vin, "alt_t0"])
        ttf = (pd.Timestamp(E.JCOPENDATE[vin]) - t0).days
        rows.append(dict(vin=vin, dname=E.display_name(vin), ttf=float(ttf),
                         ktf=float(lc.loc[vin, "est_km"]), kpd=float(fr.loc[vin, "km_per_day_est"])))
    return pd.DataFrame(rows)


def fit_w(x):
    c, _loc, s = weibull_min.fit(np.asarray(x, float), floc=0)
    return c, s


def cond_med_remaining(c, s, a):
    """Conditional median remaining beyond threshold a for Weibull(c, scale=s)."""
    w = weibull_min(c, scale=s)
    sf_a = float(w.sf(a))
    if sf_a <= 1e-9:
        return 0.0
    return max(float(w.isf(0.5 * sf_a)) - a, 0.0)


def backtest(df):
    recs = []
    for f in FRACS:
        for _, r in df.iterrows():
            others = df[df.vin != r.vin]
            cT, sT = fit_w(others.ttf)
            cK, sK = fit_w(others.ktf)
            cal = cond_med_remaining(cT, sT, f * r.ttf)
            usg = cond_med_remaining(cK, sK, f * r.ktf) / r.kpd
            actual = (1 - f) * r.ttf
            recs.append(dict(frac=f, vin=r.dname, actual=actual, cal_pred=cal, usg_pred=usg,
                             cal_err=cal - actual, usg_err=usg - actual))
    return pd.DataFrame(recs)


def main():
    df = load_failed()
    cvT = df.ttf.std(ddof=1) / df.ttf.mean()
    cvK = df.ktf.std(ddof=1) / df.ktf.mean()

    print(f"Failed trucks used: {len(df)}")
    print("\nPer-truck inputs (3 independent sources):")
    show = df.copy()
    show["impliedKpd"] = show.ktf / show.ttf
    print(show[["dname", "ttf", "ktf", "kpd", "impliedKpd"]].to_string(
        index=False, float_format=lambda v: f"{v:,.1f}"))
    print(f"\nDISPATCH DIAGNOSTIC  ->  CV(days-to-failure) = {cvT:.3f}   CV(km-to-failure) = {cvK:.3f}")
    print("   usage clock can only win if CV(km) < CV(days):",
          "YES, km is tighter" if cvK < cvT else "NO, days is tighter")

    bt = backtest(df)
    bt.to_csv(os.path.join(OUTD, "usage_clock_backtest.csv"), index=False)

    print("\nLOVO conditional-median RUL MAE (days), by prediction horizon:")
    print(f"  {'horizon':<22}{'calendar MAE':>14}{'usage MAE':>12}{'winner':>10}")
    per = []
    for f in FRACS:
        d = bt[bt.frac == f]
        cmae, umae = d.cal_err.abs().mean(), d.usg_err.abs().mean()
        per.append((f, cmae, umae))
        lbl = {0.0: "install (predict life)", 0.5: "mid-life (50%)", 0.8: "late (80%)"}[f]
        print(f"  {lbl:<22}{cmae:>14.1f}{umae:>12.1f}{('usage' if umae < cmae else 'calendar'):>10}")
    omae_c, omae_u = bt.cal_err.abs().mean(), bt.usg_err.abs().mean()
    print(f"  {'OVERALL':<22}{omae_c:>14.1f}{omae_u:>12.1f}{('usage' if omae_u < omae_c else 'calendar'):>10}")
    print(f"\n  (historical fleet calendar-clock baseline was ~50 d; per-truck beta attempt ~142 d)")

    verdict = ("USAGE clock WINS" if omae_u < omae_c else "CALENDAR clock WINS") + \
              f" overall ({omae_u:.0f} vs {omae_c:.0f} d MAE)"
    print(f"\nVERDICT: {verdict}")

    _write_report(df, bt, cvT, cvK, per, omae_c, omae_u, verdict)
    print(f"\nWrote: {REPORT}\n       {os.path.join(OUTD, 'usage_clock_backtest.csv')}")


def _write_report(df, bt, cvT, cvK, per, omae_c, omae_u, verdict):
    L = []
    L.append("---\ntitle: \"Usage-clock RUL prototype + LOVO backtest\"\nstatus: complete\n"
             "created: 2026-06-26\n---\n")
    L.append("# Can a per-truck USAGE clock beat the fleet CALENDAR clock?\n")
    L.append("**Question.** Section 1's RUL is fleet-based (a function of calendar age only). A usage "
             "clock models km-to-failure and converts to days via each truck's km/day, so a hard-working "
             "truck burns RUL faster. Does it predict remaining life better?\n")
    L.append("## Method (honest)\n")
    L.append("- Leave-one-VIN-out: score each failure with a Weibull fit on the **other 9** only.\n"
             "- Three independent inputs: `TTF_days = JCOPENDATE - t0`, `KTF_km = est_km`, "
             "`kpd = km_per_day_est`.\n"
             "- Conditional median remaining at install (f=0), mid-life (f=0.5), late (f=0.8); "
             "MAE vs actual `(1-f)*TTF`.\n")
    L.append("## Deciding diagnostic\n")
    L.append(f"| spread across the 10 failures | CV |\n|---|---|\n"
             f"| days-to-failure (calendar) | **{cvT:.3f}** |\n"
             f"| km-to-failure (usage) | **{cvK:.3f}** |\n")
    L.append(f"\nThe usage clock can only win if km-to-failure is the tighter quantity. "
             f"Here **{'km is tighter -> usage has a chance' if cvK < cvT else 'days is tighter -> calendar should win'}**.\n")
    L.append("## Result — LOVO conditional-median RUL MAE (days)\n")
    L.append("| horizon | calendar MAE | usage MAE | winner |\n|---|---|---|---|")
    names = {0.0: "install (predict full life)", 0.5: "mid-life (50%)", 0.8: "late (80%)"}
    for f, c, u in per:
        L.append(f"| {names[f]} | {c:.1f} | {u:.1f} | {'**usage**' if u < c else '**calendar**'} |")
    L.append(f"| **overall** | **{omae_c:.1f}** | **{omae_u:.1f}** | "
             f"**{'usage' if omae_u < omae_c else 'calendar'}** |")
    L.append(f"\nHistorical fleet calendar-clock baseline ≈ 50 d; the per-truck *feature* (beta) attempt ≈ 142 d.\n")
    L.append("## Verdict\n")
    L.append(f"**{verdict}.**\n")
    if omae_u < omae_c:
        L.append("The usage clock improves per-truck RUL — adopt it for Section 1 (re-axis the schedule on "
                 "usage, convert to days per truck). Keep the fleet window as the honest uncertainty band.\n")
    else:
        L.append("The usage clock does **not** beat the calendar fleet clock at n=10 failures: alternators here "
                 "fail at a more consistent **age** than mileage, so personalizing the clock by usage adds error. "
                 "This re-confirms the standing finding — at this data scale the RUL **number** stays the fleet "
                 "window (601 d / ~120k km); per-truck intelligence lives in Section 1b (condition), Section 0 "
                 "(effective alert) and the GED emergency channel, not in a per-truck RUL number. "
                 "A usage clock becomes worth revisiting only with many more failure events (the 500-truck scale-up).\n")
    L.append("## Per-truck detail (install horizon, f=0 = lifespan prediction)\n")
    L.append("| truck | actual life (d) | calendar pred | usage pred | cal err | usage err |\n|---|---|---|---|---|---|")
    d0 = bt[bt.frac == 0.0]
    for _, r in d0.iterrows():
        L.append(f"| {r.vin} | {r.actual:.0f} | {r.cal_pred:.0f} | {r.usg_pred:.0f} | "
                 f"{r.cal_err:+.0f} | {r.usg_err:+.0f} |")
    L.append("\n_Inputs: TTF=JCOPENDATE−t0, KTF=est_km, kpd=km_per_day_est. n=25 data ceiling. Confidential._\n")
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
