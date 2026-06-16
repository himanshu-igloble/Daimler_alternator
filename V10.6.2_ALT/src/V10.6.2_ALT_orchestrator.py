"""
V10.6.2 Alternator — Master Orchestrator (alpha honest-baseline)
================================================================
Runs the V10.6.2 pipeline in dependency order.  verify runs LAST so it can
check the generated reports (B1/B2 gates).

Order:
  1. weibull_fleet     (Tier A fleet survival, cohort=25, posterior CI)
  2. predictive_rul    (survival-conditioned per-VIN RUL band)
  3. backtest          (LOVO + dummies + coverage  <- centerpiece)
  4. ged_emergency     (daily GED=2 alert, out-of-sample)
  5. decisions         (2x2 risk x time)
  6. assemble_rul      (final table + empirical fleet window)
  7. rul_graphs        (figures)
  8. narrative_rul     (per-VIN narratives)
  9. markdown_report   (customer-facing md)
  10. excel_report     (workbook)
  11. verify           (honest gates; exits 1 on gating failure)
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import traceback

_src = pathlib.Path(__file__).resolve().parent
_cfg_spec = importlib.util.spec_from_file_location(
    "V10_6_2_ALT_config", str(_src / "V10.6.2_ALT_config.py"))
cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(cfg)

SCRIPTS = [
    "V10.6.2_ALT_weibull_fleet",
    "V10.6.2_ALT_predictive_rul",
    "V10.6.2_ALT_backtest",
    "V10.6.2_ALT_ged_emergency",
    "V10.6.2_ALT_decisions",
    "V10.6.2_ALT_assemble_rul",
    "V10.6.2_ALT_rul_graphs",
    "V10.6.2_ALT_narrative_rul",
    "V10.6.2_ALT_verify",            # before reports so the gate table embeds
    "V10.6.2_ALT_markdown_report",
    "V10.6.2_ALT_excel_report",
]


def _run(name):
    path = _src / f"{name}.py"
    if not path.exists():
        return "SKIP", 0.0, "missing"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    t0 = time.perf_counter()
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return "OK", time.perf_counter() - t0, None
    except SystemExit as e:
        # verify raises SystemExit(1) on gating failure — capture, don't crash
        return ("OK" if (e.code in (0, None)) else "GATE_FAIL"), time.perf_counter() - t0, f"exit={e.code}"
    except Exception as e:
        print(f"\n{'!'*70}\nERROR in {name}:\n{traceback.format_exc()}{'!'*70}\n")
        return "FAIL", time.perf_counter() - t0, str(e)


def main():
    print("=" * 70)
    print(f"V10.6.2 ALTERNATOR — ORCHESTRATOR  ({cfg.VERSION})")
    print("=" * 70)
    results = []
    for i, name in enumerate(SCRIPTS, 1):
        print(f"\n{'='*70}\n[{i}/{len(SCRIPTS)}] {name}\n{'='*70}")
        status, el, err = _run(name)
        results.append((name, status, el, err))
        print(f"\n  >> {name}: {status} ({el:.1f}s){'' if not err else ' - ' + str(err)}")

    print(f"\n\n{'='*70}\nSUMMARY\n{'='*70}")
    nok = nfail = 0
    for i, (n, s, el, err) in enumerate(results, 1):
        print(f"{i:<3} {n:<34} {s:<10} {el:>6.1f}s  {err or ''}")
        nok += s == "OK"
        nfail += s in ("FAIL", "GATE_FAIL")
    print(f"\n{nok}/{len(SCRIPTS)} OK, {nfail} failed/gate-fail")
    if nfail:
        sys.exit(1)


if __name__ == "__main__":
    main()
