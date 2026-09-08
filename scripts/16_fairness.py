"""Step 16 (amendment v3.1 Section F): fairness metrics by subgroup on the test era at the single overall 20% threshold. Writes results/fairness.json."""
import json, pathlib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
P = pd.read_parquet(D/"test_predictions.parquet"); F = pd.read_parquet(D/"dp_features.parquet")[["decision_id","gender","age","any_bh","sud"]]; df = P.merge(F, on="decision_id")
df["age_band"] = pd.cut(df.age, [-1, 17, 44, 64, 200], labels=["under 18","18 to 44","45 to 64","65 and over"]).astype(str); df["sex"] = df.gender.where(df.gender.isin(["Female","Male"]), "Other or not recorded"); df["behavioral_health"] = np.where(df.any_bh == 1, "behavioral health diagnosis", "none")
df["race_group"] = df.race.where(df.race.isin(["Black or African American","White","Hispanic","Asian"]), "Other or unknown")
def calib(y, p):
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1,1), y); return round(float(lr.coef_[0][0]), 3), round(float(lr.intercept_[0]), 3)
out = {"threshold_rule": "single overall threshold at the 80th percentile of each model's test-era predictions", "groups": {}}
rng = np.random.default_rng(20260908)
for model in ["M5_full","M2_structured_history"]:
    thr = np.quantile(df[model], 0.8); df["flag_"+model] = (df[model] >= thr).astype(int)
for dim, ref in [("sex","Female"),("race_group","White"),("state","VIRGINIA"),("age_band","18 to 44"),("behavioral_health","none")]:
    out["groups"][dim] = {"reference": ref, "levels": {}}
    for lev, g in df.groupby(dim):
        if len(g) < 200 or g.y.nunique() < 2: out["groups"][dim]["levels"][str(lev)] = {"n": int(len(g)), "note": "insufficient"}; continue
        rec = {"n": int(len(g)), "members": int(g.person_id.nunique()), "event_rate": round(float(g.y.mean()), 4)}
        for model in ["M5_full","M2_structured_history"]:
            sl, ic = calib(g.y.values, g[model].values); f = g["flag_"+model] == 1
            rec[model] = {"auroc": round(float(roc_auc_score(g.y, g[model])), 3), "auprc": round(float(average_precision_score(g.y, g[model])), 3), "calibration_slope": sl, "calibration_intercept": ic, "flag_rate": round(float(f.mean()), 3), "sensitivity": round(float(g.y[f].sum()/max(g.y.sum(),1)), 3), "ppv": round(float(g.y[f].mean()), 3) if f.sum() else None}
        out["groups"][dim]["levels"][str(lev)] = rec
    # bootstrap differences vs reference for the full model (sensitivity and PPV at threshold)
    for lev in out["groups"][dim]["levels"]:
        if lev == ref or "note" in out["groups"][dim]["levels"][lev]: continue
        a, b = df[df[dim] == lev], df[df[dim] == ref]; ds, dp = [], []
        ua, ub = a.person_id.unique(), b.person_id.unique(); ia = {p: np.where(a.person_id.values == p)[0] for p in ua}; ib = {p: np.where(b.person_id.values == p)[0] for p in ub}
        for _ in range(300):
            xa = np.concatenate([ia[p] for p in rng.choice(ua, len(ua))]); xb = np.concatenate([ib[p] for p in rng.choice(ub, len(ub))]); A, B = a.iloc[xa], b.iloc[xb]
            fa, fb = A.flag_M5_full == 1, B.flag_M5_full == 1
            if A.y.sum() and B.y.sum() and fa.sum() and fb.sum(): ds.append(A.y[fa].sum()/A.y.sum() - B.y[fb].sum()/B.y.sum()); dp.append(A.y[fa].mean() - B.y[fb].mean())
        out["groups"][dim]["levels"][lev]["M5_full"]["sensitivity_diff_vs_ref_ci95"] = [round(float(np.percentile(ds, 2.5)), 3), round(float(np.percentile(ds, 97.5)), 3)]; out["groups"][dim]["levels"][lev]["M5_full"]["ppv_diff_vs_ref_ci95"] = [round(float(np.percentile(dp, 2.5)), 3), round(float(np.percentile(dp, 97.5)), 3)]
json.dump(out, open(R/"fairness.json", "w"), indent=1); print(json.dumps(out, indent=1)[:3500])
