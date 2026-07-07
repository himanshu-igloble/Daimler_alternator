"""V12.1 LOVO trial: does the idle-ANR load proxy separate failed/NF or add over 0.9267?"""
import importlib.util, pathlib
import pandas as pd
_SRC=pathlib.Path(__file__).resolve().parent
def _load(n):
    s=importlib.util.spec_from_file_location(n,str(_SRC/f"{n}.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
C=_load("ged_common"); T=_load("trial_4_lovo")
NEW=["anr_idle_mean","anr_idle_resid_mean","anr_idle_resid_slope_30d","anr_idle_resid_oscillation"]

def main():
    base=pd.read_csv(C.ROOT/"V5.2_ALT"/"features"/"V5.2_20_5_ALT_selected_features.csv")
    if "VIN" not in base.columns or not base["VIN"].astype(str).str.contains("_ALT").any():
        for col in ("truck_id","VIN_LABEL","vin"):
            if col in base.columns and base[col].astype(str).str.contains("_ALT").any():
                if "VIN" in base.columns:
                    base=base.drop(columns=["VIN"])
                base=base.rename(columns={col:"VIN"}); break
    cand=pd.read_csv(C.RESULTS/"v121_anr_features.csv")
    # drop 'failed' from cand to avoid failed_x/failed_y collision (baseline is authoritative)
    if "failed" in cand.columns: cand=cand.drop(columns=["failed"])
    df=base.merge(cand, on="VIN", how="left")
    assert len(df)==25, f"expected 25 rows, got {len(df)}"
    a0=T.lovo_auroc(df, T.FAMILY_A); print(f"BASELINE FAMILY_A AUROC={a0:.4f} (expect ~0.9267)")
    assert abs(a0-0.9267)<0.005, f"baseline not reproduced ({a0:.4f})"
    rows=[{"set":"FAMILY_A","auroc_additive":round(a0,4),"delta":0.0,"single_feat_auroc":""}]
    for f in NEW:
        add=T.lovo_auroc(df, T.FAMILY_A+[f]); single=T.lovo_auroc(df, [f])
        rows.append({"set":f"+{f}","auroc_additive":round(add,4),"delta":round(add-a0,4),
                     "single_feat_auroc":round(single,4)})
    allf=T.lovo_auroc(df, T.FAMILY_A+NEW)
    rows.append({"set":"+ALL_ANR","auroc_additive":round(allf,4),"delta":round(allf-a0,4),"single_feat_auroc":""})
    out=pd.DataFrame(rows); out.to_csv(C.RESULTS/"v121_anr_lovo_trial.csv",index=False)
    print(out.to_string(index=False))
    print("\nGroup means failed(1) vs NF(0):"); print(df.groupby("failed")[NEW].mean().to_string())

if __name__=="__main__": main()
