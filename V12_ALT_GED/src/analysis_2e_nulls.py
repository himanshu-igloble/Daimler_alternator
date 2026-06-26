"""Phase 2E: is GED missingness informative?"""
import importlib.util, pathlib, json
import polars as pl
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")

def main():
    df=pl.read_parquet(C.DAILY_CACHE)
    by_fail=df.group_by("failed").agg((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("null_frac"))
    fail=df.filter(pl.col("failed"))
    near=fail.filter(pl.col("dtf")<=30).select((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("nf"))
    base=fail.filter(pl.col("dtf")>30).select((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("bf"))
    per_vin=df.group_by("vin","failed").agg((pl.col("ged_null").sum()/pl.col("n_rows").sum()).alias("null_frac")).sort("null_frac",descending=True)
    per_vin.write_csv(C.RESULTS/"2e_null_structure.csv")
    rep={"by_failed":by_fail.to_dicts(),
         "failed_near_le30d_null":float(near["nf"][0]) if near.height else None,
         "failed_base_gt30d_null":float(base["bf"][0]) if base.height else None}
    (C.RESULTS/"2e_null_report.json").write_text(json.dumps(rep,indent=2)); print(rep)

if __name__ == "__main__":
    main()
