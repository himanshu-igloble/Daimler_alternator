"""V12.1: engine-torque (ANR) alternator electrical-LOAD proxy (idle-conditioned).

At idle (engine on, stationary, not cranking) propulsion torque ~ 0, so ANR reflects accessory
load; the idle-ANR residual vs the healthy fleet is an honest alternator electrical-LOAD proxy
(NOT duty cycle / field current). Confounded by other accessories; alternator drag is small.
"""
import importlib.util, pathlib
import polars as pl, numpy as np
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common")
RPM_IDLE_LO, RPM_IDLE_HI, RPM_BIN = 500.0, 800.0, 50.0

def idle_rows(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter((pl.col("RPM")>=RPM_IDLE_LO)&(pl.col("RPM")<=RPM_IDLE_HI)
                     &(pl.col("CSP")<3.0)&(pl.col("SMA")==0)&pl.col("ANR").is_not_null())

def fit_idle_baseline(nf_idle: pl.DataFrame) -> pl.DataFrame:
    d=nf_idle.with_columns((pl.col("RPM")//RPM_BIN).alias("rb"))
    return d.group_by("rb").agg(pl.col("ANR").median().alias("anr_exp"))

def idle_residual(df_idle: pl.DataFrame, base: pl.DataFrame) -> pl.DataFrame:
    d=df_idle.with_columns((pl.col("RPM")//RPM_BIN).alias("rb")).join(base,on="rb",how="inner")
    return d.with_columns((pl.col("ANR")-pl.col("anr_exp")).alias("anr_resid"))

def per_vin_features(vin: str, base: pl.DataFrame) -> dict:
    di=idle_rows(C.load_vin(vin))
    if di.height==0:
        return {"VIN":vin,"anr_idle_n":0,"anr_idle_mean":float("nan"),
                "anr_idle_resid_mean":float("nan"),"anr_idle_resid_slope_30d":float("nan"),
                "anr_idle_resid_oscillation":float("nan")}
    r=idle_residual(di,base)
    daily=r.group_by("day").agg(pl.col("anr_resid").mean().alias("rd")).sort("day")
    y=daily["rd"].to_numpy(); k=min(30,len(y))
    slope=float(np.polyfit(np.arange(k),y[-k:],1)[0]) if k>=2 else float("nan")
    return {"VIN":vin,"anr_idle_n":di.height,"anr_idle_mean":float(di["ANR"].mean()),
            "anr_idle_resid_mean":float(np.nanmean(r["anr_resid"].to_numpy())),
            "anr_idle_resid_slope_30d":slope,
            "anr_idle_resid_oscillation":float(np.nanstd(y))}

def main():
    import sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    nf_idle=pl.concat([idle_rows(C.load_vin(v)) for v in C.NONFAILED_VINS])
    base=fit_idle_baseline(nf_idle)
    rows=[{**per_vin_features(v,base),"failed":C.is_failed(v)} for v in C.ALL_VINS]
    out=pl.DataFrame(rows); out.write_csv(C.RESULTS/"v121_anr_features.csv")
    print("baseline rpm-bins:",base.height,"| total NF idle rows:",nf_idle.height)
    print(out.sort("anr_idle_resid_mean"))

if __name__=="__main__": main()
