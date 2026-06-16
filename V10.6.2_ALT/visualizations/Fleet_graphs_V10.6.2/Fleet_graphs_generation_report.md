---
title: "V10.6.2 Alternator Fleet Overlay -- Generation Report"
status: "complete"
created: "2026-06-12"
---

# V10.6.2 Alternator Fleet-Level Visualization Suite

**Generated:** 2026-06-12 09:44  
**Output directory:** `D:\Daimler-starter_motor_alternator_battery\V10.6.2_ALT\visualizations\Fleet_graphs_V10.6.2`

## 1. Processing summary

- **Total VINs processed:** 25 of 25
- **Failed:** 10  |  **In-service:** 15
- **Excluded VINs:** none (all VINs had >=3 weekly snapshots)

## 2. Graphs generated

| # | View | Files |
|---|------|-------|
| 1 | Failed Vehicle Fleet Overlay | `failed_vehicle_fleet_overlay.png`, `failed_vehicle_fleet_overlay.pdf`, `failed_vehicle_fleet_overlay_hd.png` |
| 2 | Non-Failed Vehicle Fleet Overlay | `non_failed_vehicle_fleet_overlay.png`, `non_failed_vehicle_fleet_overlay.pdf`, `non_failed_vehicle_fleet_overlay_hd.png` |
| 3 | Combined Failed vs Non-Failed Comparison | `fleet_comparison_overlay.png`, `fleet_comparison_overlay.pdf`, `fleet_comparison_overlay_hd.png` |

Each view: **standard PNG (150 dpi)**, **vector PDF**, **HD PNG (300 dpi)**.

## 3. Data deliverables

- `fleet_statistics_summary.csv` -- per-VIN analytical summary (CSV)
- `fleet_statistics_summary.xlsx` -- same, formatted workbook (XLSX)

Columns: VIN, Failure_Status, **Data_Start_Date, Data_End_Date, Observed_Span_days** (real per-VIN window), Current_RUL_median_days, Predicted_RUL_days, RUL_p10/p90, Forecast_Failure_Date, **EndOfLife_est_km, EndOfLife_est_engine_hrs**, km_per_day_est, zone-entry dates, Current_Zone, Final_Zone_Reached, Risk_prob, Risk_tier, GED2_emergency.

## 4. Scaling methodology

- **Shared x-axis = REAL calendar date.** Each VIN is drawn over its own observed window (`alt_t0` -> `alt_t1`). The fleet did not start on a common day and no truck has data from "day 0 / 0 km", so the lines are genuinely staggered in time -- that is the real fleet picture.
- **x-limits (combined view):** 2023-12-13 -> 2026-09-08 -- covers every VIN start, end and forecast (no clipping). The two standalone graphs ZOOM to their own data span (failed vs in-service) to use maximum plotting area; the combined view uses this union scale so both panels mirror exactly.
- **y-limits:** [-12, 765] d RUL, mirrored across all panels; tight headroom so the degradation patterns fill the plotting area.
- **Horizontal RUL zones (y-axis, calendar-independent):** GREEN >180d, YELLOW 90-180d, ORANGE 30-90d, BLACK <30d.
- **Fleet wear-out window** (578-652 d, median 601 d) is age-based, so on the calendar axis it is expressed per-VIN by where each line ends, not as one global band.
- **km & engine-hours:** carried per-VIN in the legend and CSV as each truck's REAL end-of-life estimate (a single global distance axis is meaningless once trucks share calendar time at different km/day rates).
- **Legend placement:** moved OUTSIDE below the plot (multi-column) so the axes use the full figure width.

## 5. Per-VIN summary table

| VIN | Failure_Status | Data_Start_Date | Data_End_Date | Observed_Span_days | Predicted_RUL_days | Forecast_Failure_Date | EndOfLife_est_km | EndOfLife_est_engine_hrs | Current_Zone | Risk_prob | Risk_tier |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VIN1_F_ALT | FAILED | 2024-04-01 | 2025-11-18 | 596.0 | 0.0 |  | 135365.0 | 5328.0 | EVENT | 0.6058 | RED |
| VIN2_F_ALT | FAILED | 2024-04-01 | 2025-12-16 | 627.0 | 0.0 |  | 104473.0 | 3887.0 | EVENT | 0.6085 | RED |
| VIN3_F_ALT | FAILED | 2024-02-19 | 2025-09-27 | 591.0 | 0.0 |  | 152634.0 | 4787.0 | EVENT | 0.7603 | RED |
| VIN4_F_ALT | FAILED | 2024-02-05 | 2025-11-25 | 664.0 | 0.0 |  | 115517.0 | 3798.0 | EVENT | 0.4456 | AMBER |
| VIN5_F_ALT | FAILED | 2024-02-05 | 2025-11-22 | 661.0 | 0.0 |  | 108462.0 | 4501.0 | EVENT | 0.2799 | GREEN |
| VIN6_F_ALT | FAILED | 2024-04-01 | 2025-09-30 | 552.0 | 0.0 |  | 119166.0 | 3980.0 | EVENT | 0.7112 | RED |
| VIN7_F_ALT | FAILED | 2024-02-05 | 2025-12-04 | 673.0 | 0.0 |  | 112798.0 | 4758.0 | EVENT | 0.7178 | RED |
| VIN8_F_ALT | FAILED | 2024-05-06 | 2025-11-24 | 573.0 | 0.0 |  | 121846.0 | 4951.0 | EVENT | 0.8923 | RED |
| VIN9_F_ALT | FAILED | 2024-05-06 | 2025-12-27 | 606.0 | 0.0 |  | 172911.0 | 5921.0 | EVENT | 0.4919 | AMBER |
| VIN10_F_ALT | FAILED | 2024-09-02 | 2025-12-16 | 472.0 | 0.0 |  | 123993.0 | 4635.0 | EVENT | 0.6263 | RED |
| VIN11_NF_ALT | IN_SERVICE | 2024-02-05 | 2026-02-18 | 747.0 | 97.3 | 2026-05-26 | 138649.0 | 6662.0 | YELLOW | 0.2553 | LOW_RISK |
| VIN12_NF_ALT | IN_SERVICE | 2024-01-22 | 2026-02-18 | 758.0 | 89.5 | 2026-05-18 | 131416.0 | 4882.0 | ORANGE | 0.3956 | LOW_RISK |
| VIN13_NF_ALT | IN_SERVICE | 2024-03-25 | 2026-02-13 | 693.0 | 116.2 | 2026-06-09 | 154586.0 | 5365.0 | YELLOW | 0.4906 | HIGH_RISK |
| VIN14_NF_ALT | IN_SERVICE | 2024-04-08 | 2026-02-17 | 685.0 | 123.5 | 2026-06-20 | 122911.0 | 6444.0 | YELLOW | 0.4257 | LOW_RISK |
| VIN15_NF_ALT | IN_SERVICE | 2024-03-11 | 2026-02-18 | 721.0 | 105.5 | 2026-06-03 | 132007.0 | 5157.0 | YELLOW | 0.325 | LOW_RISK |
| VIN16_NF_ALT | IN_SERVICE | 2024-02-26 | 2026-02-18 | 727.0 | 102.4 | 2026-05-31 | 193415.0 | 5756.0 | YELLOW | 0.4133 | LOW_RISK |
| VIN17_NF_ALT | IN_SERVICE | 2024-03-04 | 2026-02-18 | 719.0 | 106.4 | 2026-06-04 | 149346.0 | 4721.0 | YELLOW | 0.4116 | LOW_RISK |
| VIN18_NF_ALT | IN_SERVICE | 2024-01-01 | 2025-05-27 | 513.0 | 234.5 | 2026-01-16 | 167484.0 | 7355.0 | GREEN | 0.3934 | LOW_RISK |
| VIN19_NF_ALT | IN_SERVICE | 2024-01-01 | 2025-07-28 | 574.0 | 189.5 | 2026-02-02 | 207978.0 | 6757.0 | GREEN | 0.1053 | LOW_RISK |
| VIN20_NF_ALT | IN_SERVICE | 2024-01-22 | 2026-02-18 | 761.0 | 88.5 | 2026-05-17 | 130381.0 | 4715.0 | ORANGE | 0.4389 | LOW_RISK |
| VIN21_NF_ALT | IN_SERVICE | 2024-01-15 | 2026-02-16 | 766.0 | 87.0 | 2026-05-14 | 155794.0 | 7745.0 | ORANGE | 0.2559 | LOW_RISK |
| VIN22_NF_ALT | IN_SERVICE | 2024-07-22 | 2026-02-18 | 581.0 | 183.0 | 2026-08-20 | 154402.0 | 5417.0 | GREEN | 0.0421 | LOW_RISK |
| VIN23_NF_ALT | IN_SERVICE | 2024-03-25 | 2026-02-11 | 691.0 | 118.4 | 2026-06-09 | 137706.0 | 5520.0 | YELLOW | 0.2104 | LOW_RISK |
| VIN24_NF_ALT | IN_SERVICE | 2024-05-06 | 2026-02-18 | 656.0 | 137.7 | 2026-07-05 | 157397.0 | 5786.0 | YELLOW | 0.3162 | LOW_RISK |
| VIN25_NF_ALT | IN_SERVICE | 2024-01-01 | 2025-09-26 | 634.0 | 151.0 | 2026-02-24 | 185716.0 | 6861.0 | YELLOW | 0.3987 | LOW_RISK |

## 6. Anomalies & honesty notes

- **GED2 excitation-storm precursors:** 2 truck(s) (VIN1_F_ALT, VIN10_F_ALT). Other failed trucks have NO usable per-truck precursor (flat telemetry).
- RUL is a function of **age + fleet survival model**, not per-truck telemetry -- no fabricated per-truck wear trend. Failed trucks land at RUL=0 on their real failure date.
- Risk tier is a **static** classifier (AUROC 0.927 = which, not when). km / engine-hours are speed-integrated estimates ("est."), not odometer.

## 7. Validation checklist

- [x] Each VIN plotted on its REAL calendar window (real start/length/end dates).
- [x] RUL values + Hermite forecast wedge imported verbatim from the V10.6.2 engine.
- [x] x/y limits cover all VIN starts, endpoints, forecasts (no clipping).
- [x] Legend moved OUTSIDE below the plot; axes use full figure width.
- [x] No annotation boxes / text overlays inside the plot area.
- [x] Standard (150 dpi) + HD (300 dpi) PNG + vector PDF saved for all 3 views.

🟢 Fleet overlay suite generated and validated.