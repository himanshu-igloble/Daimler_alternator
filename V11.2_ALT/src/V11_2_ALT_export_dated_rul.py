"""
Export the current (badged) per-VIN RUL evidence-stack figures to a DATED snapshot folder with
datestamped filenames, per the project naming convention (YYYY-MM-DD-<file>).

Why: the canonical figures (e.g. VIN1_F_ALT_evidence_stack.png) are referenced by the report, deck
and reading guide, so their names must stay stable. This makes a dated, shareable copy of the latest
state (now carrying the ABRUPT / SHORT-LEAD / GRADUAL failure-mode badges) without touching them.

Output: V11.2_ALT/visualizations/rul_evidence_stack_dated/<YYYY-MM-DD>-<original_name>
Run: py -3
"""
import os
import shutil
from datetime import date

ROOT = r"D:/Daimler-starter_motor_alternator_battery"
SRC = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack")
OUT = os.path.join(ROOT, "V11.2_ALT", "visualizations", "rul_evidence_stack_dated")
os.makedirs(OUT, exist_ok=True)

STAMP = date.today().isoformat()   # YYYY-MM-DD (project convention)


def main():
    n_png = n_svg = 0
    for f in sorted(os.listdir(SRC)):
        if "_evidence_stack" not in f:
            continue
        if f.endswith(".png"):
            n_png += 1
        elif f.endswith(".svg"):
            n_svg += 1
        else:
            continue
        shutil.copy2(os.path.join(SRC, f), os.path.join(OUT, f"{STAMP}-{f}"))
    print(f"date stamp : {STAMP}")
    print(f"exported   : {n_png} PNG + {n_svg} SVG -> {OUT}")
    assert n_png >= 25, f"expected >=25 PNGs, copied {n_png}"


if __name__ == "__main__":
    main()
