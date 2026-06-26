"""Phase 2B: per-VIN GED occupancy."""
import importlib.util, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def main():
    df = pl.read_parquet(C.DAILY_CACHE)
    occ = df.group_by("vin","failed").agg([
        pl.col("ged_cnt_0").sum().alias("c0"), pl.col("ged_cnt_1").sum().alias("c1"),
        pl.col("ged_cnt_2").sum().alias("c2"), pl.col("ged_cnt_3").sum().alias("c3"),
        pl.col("ged_null").sum().alias("nnull"), pl.col("n_rows").sum().alias("ntot"),
    ]).with_columns([
        (pl.col("c2")/(pl.col("c0")+pl.col("c1")+pl.col("c2")+pl.col("c3"))).alias("ged2_frac_obs"),
        (pl.col("c3")/(pl.col("c0")+pl.col("c1")+pl.col("c2")+pl.col("c3"))).alias("ged3_frac_obs"),
        (pl.col("nnull")/pl.col("ntot")).alias("null_frac"),
    ]).sort(["failed","vin"])
    occ.write_csv(C.RESULTS/"2b_occupancy.csv")
    print(occ)
    v1 = occ.filter(pl.col("vin")=="VIN1_F_ALT")["c2"].item()
    print("VIN1_F_ALT GED2:", v1, "(expect ~82357)")

if __name__ == "__main__":
    main()
