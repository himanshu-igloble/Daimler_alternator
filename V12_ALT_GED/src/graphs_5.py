"""Phase 5: investigation figures."""
import importlib.util, pathlib
import polars as pl, pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_SRC = pathlib.Path(__file__).resolve().parent


def _load(n):
    s = importlib.util.spec_from_file_location(n, str(_SRC / f"{n}.py"))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


C = _load("ged_common")
G = C.RESULTS.parent / "graphs"


def state_timeline(vin="VIN1_F_ALT"):
    d = (
        pl.read_parquet(C.DAILY_CACHE)
        .filter(pl.col("vin") == vin)
        .sort("day")
        .to_pandas()
    )
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(d["day"], d["ged_cnt_2"], color="#c0392b", lw=1.2)
    ax.axhline(200, ls="--", color="gray", label="emergency threshold (200/day)")
    ax.set_title(f"GED=2 daily count - {vin}")
    ax.set_ylabel("GED=2 count/day")
    ax.legend()
    fig.tight_layout()
    out = G / f"ged_state_timeline_{vin}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def heatmap(pop):
    m = pd.read_csv(C.RESULTS / f"2c_transitions_{pop}.csv").set_index("from")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(m.values, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([0, 1, 2, 3])
    ax.set_yticks(range(4))
    ax.set_yticklabels([0, 1, 2, 3])
    ax.set_xlabel("to state")
    ax.set_ylabel("from state")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    ax.set_title(f"GED transition P - {pop}")
    fig.colorbar(im)
    fig.tight_layout()
    out = G / f"transition_heatmap_{pop}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def regulation_overlay():
    r = pd.read_csv(C.RESULTS / "3b_regulation_features.csv")
    # Robust color mapping: handle bool, "True"/"False", "true"/"false"
    def _color(v):
        if isinstance(v, bool):
            return "#c0392b" if v else "#2980b9"
        s = str(v).strip().lower()
        return "#c0392b" if s == "true" else "#2980b9"

    colors = r["failed"].apply(_color)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(r["resid_mean"], r["resid_slope_30d"], c=colors, edgecolors="none", s=55, alpha=0.85)
    # Add manual legend patches
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#c0392b', markersize=8, label='failed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=8, label='non-failed'),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel("resid_mean")
    ax.set_ylabel("resid_slope_30d")
    ax.set_title("Regulation-effort proxy (red=failed)")
    fig.tight_layout()
    out = G / "regulation_residual_overlay.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def trigger_importance():
    t = pd.read_csv(C.RESULTS / "2d_trigger_importance.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(t["feature"], t["perm_importance"], color="#16a085")
    ax.set_title("GED=2 trigger - permutation importance")
    ax.set_xlabel("perm importance")
    fig.tight_layout()
    out = G / "trigger_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    G.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.append(state_timeline("VIN1_F_ALT"))
    outputs.append(heatmap("failed"))
    outputs.append(heatmap("nonfailed"))
    outputs.append(regulation_overlay())
    outputs.append(trigger_importance())
    print("figures written to", G)
    for p in outputs:
        size = pathlib.Path(p).stat().st_size
        print(f"  {p.name}  {size:,} bytes")


if __name__ == "__main__":
    main()
