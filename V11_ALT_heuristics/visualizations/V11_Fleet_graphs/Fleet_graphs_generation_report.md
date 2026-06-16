# V11 Fleet & Per-VIN Graphs - generation report

_V11-native: plots precursor signals, NOT RUL (V11 has no RUL data; see V10.6.2 for RUL)._

- Per-VIN precursor dashboards: 25 VINs in V11_per_vin_graphs/ (5 vote-channel panels each, healthy p05-p95 band).

- Fleet overlays (failed / non-failed / comparison) of crank_recovery_t vs vehicle age, healthy p95 line; PNG+HD+PDF.

- fleet_statistics_summary.csv/.xlsx: per-VIN status, span, earliest detection, detecting feature, #discriminative features, compound fired, GED2 total.


## Honesty notes

- crank_recovery_t is episodic (isolated spikes) against a near-zero healthy baseline (~0.05s); guard is 0/15 NF false alarms, not z-magnitude.

- Failed plotted vs days-before-failure (per-VIN) / vehicle age (overlays); non-failed vs vehicle age.

- No RUL/Weibull curves: not available for V11.
