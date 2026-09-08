"""Step 17 (amendment v3.1 Section B, quantitative companion): acute care events (ED and inpatient, from ADT feeds) in days 1-90 after test-era
decision points, disengaged vs retained, unadjusted and adjusted (Poisson, member-clustered). ADT message rows are de-duplicated to stays
by member, class code, and admit date. Writes results/clinical_stakes.json."""
import json, pathlib, warnings, numpy as np, pandas as pd, statsmodels.api as sm; warnings.filterwarnings("ignore")
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); D1 = np.timedelta64(1, "D")
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary","grad_90"]], on="decision_id"); df = df[(df.eligible == 1) & (df.training_era == 0)].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
adt = pd.read_parquet(D/"adt_events.parquet"); adt["admit_date"] = pd.to_datetime(adt.admit_date); adt = adt[adt.class_code.isin(["E","I"])].drop_duplicates(["person_id","class_code","admit_date"])
A = {p: (g.admit_date.values, g.class_code.values) for p, g in adt.groupby("person_id")}
ed, ip = np.zeros(len(df)), np.zeros(len(df))
for i, r in enumerate(df.itertuples()):
    d, c = A.get(r.person_id, (np.array([], dtype="datetime64[ns]"), np.array([]))); t = r.enc_date.to_datetime64(); m = (d > t) & (d <= t + 90*D1); ed[i] = (c[m] == "E").sum(); ip[i] = (c[m] == "I").sum()
df["ed90"], df["ip90"], df["acute90"] = ed, ip, ed + ip
def rate(g, col): return round(float(100*g[col].mean()), 2)
out = {"n_test_dps": int(len(df)), "members": int(df.person_id.nunique()), "unit": "events per 100 decision points in days 1 to 90", "rates": {f"y{yv}": {"n": int((df.y_primary == yv).sum()), "acute": rate(df[df.y_primary == yv], "acute90"), "ed": rate(df[df.y_primary == yv], "ed90"), "inpatient": rate(df[df.y_primary == yv], "ip90"), "any_acute_pct": round(float(100*(df[df.y_primary == yv].acute90 > 0).mean()), 1)} for yv in (1, 0)}}
Xa = pd.concat([df[["y_primary","risk_percentile","days_from_zd","adt_all_prior_90d","adt_all_prior_365d","age","any_bh","sud"]].astype(float).fillna(0), pd.get_dummies(df[["state","cur_rg"]], drop_first=True, dtype=float)], axis=1); Xa = sm.add_constant(Xa)
for col in ["acute90","ed90","ip90"]:
    m = sm.GLM(df[col].values, Xa, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df.person_id.values}); b, se = float(m.params["y_primary"]), float(m.bse["y_primary"])
    u = sm.GLM(df[col].values, sm.add_constant(df[["y_primary"]].astype(float)), family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": df.person_id.values}); bu, seu = float(u.params["y_primary"]), float(u.bse["y_primary"])
    out[f"irr_{col}"] = {"unadjusted": round(float(np.exp(bu)), 3), "unadjusted_ci_95": [round(float(np.exp(bu-1.96*seu)), 3), round(float(np.exp(bu+1.96*seu)), 3)], "adjusted": round(float(np.exp(b)), 3), "adjusted_ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "p_adjusted": round(float(m.pvalues["y_primary"]), 4)}
out["adjustment"] = "Poisson regression with member-clustered standard errors; covariates risk percentile at activation, days since activation, acute events in prior 90 and 365 days, age, behavioral health and substance use flags, state, discipline of the contact"
out["caveat"] = "ADT feed coverage is not universal and an event is counted only if a feed captured it; disengaged and retained members are compared on the same feeds"
json.dump(out, open(R/"clinical_stakes.json", "w"), indent=1); print(json.dumps(out, indent=1))
