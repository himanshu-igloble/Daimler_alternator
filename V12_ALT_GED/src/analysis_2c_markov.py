"""Phase 2C: GED Markov transition structure + dwell times (vectorized)."""
import importlib.util, pathlib, json
from collections import Counter
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
MAX_GAP_S = 60

def _prep(df, dt_col, max_gap_s):
    return (df.sort(dt_col)
        .with_columns(pl.col("GED").fill_null(3).cast(pl.Int64).alias("g"))
        .with_columns([
            pl.col(dt_col).cast(pl.Datetime).diff().dt.total_seconds().alias("gap_s"),
            pl.col("GED").fill_null(3).cast(pl.Int64).shift(1).alias("g_prev"),
        ]))

def transition_counts(df: pl.DataFrame, dt_col=C.DT_COL, max_gap_s=MAX_GAP_S) -> Counter:
    d = _prep(df, dt_col, max_gap_s).filter(
        pl.col("g_prev").is_not_null() & (pl.col("gap_s") <= max_gap_s))
    grp = d.group_by(["g_prev","g"]).len()
    cm = Counter()
    for row in grp.iter_rows(named=True):
        cm[(int(row["g_prev"]), int(row["g"]))] = int(row["len"])
    return cm

def matrix_from_counts(cm: Counter):
    states=[0,1,2,3]; rows={}
    for i in states:
        tot=sum(cm[(i,j)] for j in states)
        rows[i]={j:(cm[(i,j)]/tot if tot else 0.0) for j in states}
    return rows

def dwell_summary(df: pl.DataFrame, dt_col=C.DT_COL, max_gap_s=MAX_GAP_S) -> dict:
    """Mean consecutive run-length per state (vectorized run-length encoding)."""
    d = _prep(df, dt_col, max_gap_s).with_columns(
        ((pl.col("g")!=pl.col("g_prev")) | (pl.col("gap_s")>max_gap_s)
         | pl.col("g_prev").is_null()).alias("new_run"))
    d = d.with_columns(pl.col("new_run").cast(pl.Int64).cum_sum().alias("run_id"))
    runs = d.group_by("run_id").agg([pl.col("g").first().alias("state"), pl.len().alias("rl")])
    out = runs.group_by("state").agg(pl.col("rl").mean().alias("mean_rl"))
    res={s:0.0 for s in [0,1,2,3]}
    for row in out.iter_rows(named=True): res[int(row["state"])]=float(row["mean_rl"])
    return res

def main():
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    pops={"failed":C.FAILED_VINS,"nonfailed":C.NONFAILED_VINS}; dwell_out={}
    for pop,vins in pops.items():
        agg=Counter(); ds=[]
        for vin in vins:
            df=C.load_vin(vin)
            agg.update(transition_counts(df)); ds.append(dwell_summary(df))
        mat=matrix_from_counts(agg)
        pl.DataFrame([{"from":i, **{f"to_{j}":mat[i][j] for j in [0,1,2,3]}} for i in [0,1,2,3]]) \
          .write_csv(C.RESULTS/f"2c_transitions_{pop}.csv")
        dwell_out[pop]={str(s):sum(x[s] for x in ds)/len(ds) for s in [0,1,2,3]}
        print(pop,"P(0->2)=",round(mat[0][2],8),"P(2->2)=",round(mat[2][2],6),"P(2->0)=",round(mat[2][0],6))
    (C.RESULTS/"2c_dwell.json").write_text(json.dumps(dwell_out,indent=2)); print(dwell_out)

if __name__=="__main__":
    main()
