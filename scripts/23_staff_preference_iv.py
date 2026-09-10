"""Step 23 (amendment v3.3): staff-assignment (inferred-preference) instruments for first-30-day actions.
One row per patient: assigned first-contact staff member; actions in days 0-30 (any in-person contact; second contact within 7 days;
any weekend contact); outcome = disengagement in days 31-120 (no completed contact and no program completion) among patients enrolled
for all of days 31-120. Instrument = assigned staff member's leave-one-patient-out action rate. 2SLS with market x activation-quarter
fixed effects, baseline covariates, staff-clustered SEs; falsification gates I4; subgroup LATEs; DRIV heterogeneity. Writes results/staff_iv.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
F = pd.read_parquet(D/"dp_features.parquet"); S = pd.read_parquet(D/"state_features.parquet")[["encounter_id","created_by_id","roles","zd"]].rename(columns={"encounter_id":"decision_id"})
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"])
sh = pd.read_parquet(D/"status_history.parquet"); sh["dt"] = pd.to_datetime(sh.updated_at)
sp = pd.read_parquet(D/"enrollment_spans.parquet"); sp["s"] = pd.to_datetime(sp.enrollment_start_date); sp["e"] = pd.to_datetime(sp.enrollment_end_date).fillna(pd.Timestamp("2026-12-31"))
d = F.merge(S, on="decision_id"); d["enc_date"] = pd.to_datetime(d.enc_date); d["zd"] = pd.to_datetime(d.zd)
INP = {"HOME_VISIT","OTHER_INPERSON","HOSPITAL","CBO","IN_COMMUNITY","PROVIDER_OFFICE"}; GRAD = {"GRADUATED","PREGRADUATION"}; D1 = np.timedelta64(1, "D")
first = d.sort_values("enc_date").groupby("person_id").head(1).set_index("person_id")   # first completed contact = assignment
E = {p: g for p, g in enc.groupby("person_id")}; SH = {p: (g.dt.values, g.new_status.values) for p, g in sh.groupby("person_id")}; SP = {p: (g.s.values, g.e.values) for p, g in sp.groupby("person_id")}
def enrolled_all(p, a, b):
    s, e = SP.get(p, (np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")))
    if len(s) == 0: return False
    lo = np.maximum(s, a); hi = np.minimum(e, b); days = ((hi - lo) / D1).astype(float) + 1; return float(np.clip(days, 0, None).sum()) >= float((b - a) / D1) + 1
rows = []
for p, r in first.iterrows():
    z = r.zd.to_datetime64() if pd.notna(r.zd) else r.enc_date.to_datetime64(); g = E.get(p)
    if g is None: continue
    dt = g.enc_date.values; ct = g.contact_type.values
    w30 = (dt >= z) & (dt <= z + 30*D1); w31 = (dt > z + 30*D1) & (dt <= z + 120*D1)
    if r.enc_date.to_datetime64() > z + 30*D1: continue                      # first contact must fall in the first 30 days
    if (z + 120*D1) > np.datetime64("2026-06-07"): continue                    # complete 120-day window
    if not enrolled_all(p, z + 31*D1, z + 120*D1): continue
    d30 = np.sort(dt[w30]); fu7 = int(len(d30) >= 2 and (d30[1] - d30[0]) / D1 <= 7)
    inp = int(np.isin(ct[w30], list(INP)).any()); wk = int(any(pd.Timestamp(x).weekday() >= 5 for x in d30))
    sd, ss = SH.get(p, (np.array([], dtype="datetime64[ns]"), np.array([]))); grad = bool(set(ss[(sd > z) & (sd <= z + 120*D1)]) & GRAD)
    y = int((not w31.any()) and (not grad))
    rows.append(dict(person_id=p, staff=r.created_by_id, role=r.roles, market=r.market, state=r.state, q=pd.Timestamp(z).to_period("Q").strftime("%Y-Q%q"), zd=pd.Timestamp(z), y=y, inperson=inp, fu7=fu7, weekend=wk,
                     age=r.age, female=int(r.gender == "Female"), race=r.race, risk=r.risk_percentile, any_bh=r.any_bh, sud=r.sud, diabetes=r.diabetes, htn=r.htn, chf=r.chf, copd=r.copd, asthma=r.asthma, polypharmacy=r.polypharmacy, high_ed_ip=r.high_ed_ip, no_pcp=r.no_pcp_last_10mo, adt365=r.adt_all_prior_365d, n30=int(w30.sum())))
X = pd.DataFrame(rows); X = X[X.staff.notna()].reset_index(drop=True)
staff_n = X.groupby("staff").size(); X = X[X.staff.map(staff_n) >= 20].reset_index(drop=True)   # staff with at least 20 assigned patients
print(f"patients {len(X)}; staff {X.staff.nunique()}; outcome rate {X.y.mean():.3f}; action rates inperson {X.inperson.mean():.3f} fu7 {X.fu7.mean():.3f} weekend {X.weekend.mean():.3f}", flush=True)
ACTIONS = ["inperson", "fu7", "weekend"]
for a in ACTIONS:  # leave-one-patient-out staff rate
    tot = X.groupby("staff")[a].transform("sum"); n = X.groupby("staff")[a].transform("size"); X[f"z_{a}"] = (tot - X[a]) / (n - 1)
COV = ["age","female","risk","any_bh","sud","diabetes","htn","chf","copd","asthma","polypharmacy","high_ed_ip","no_pcp","adt365"]
X["race_black"] = (X.race == "Black or African American").astype(int); X["race_hisp"] = (X.race == "Hispanic").astype(int); X["race_white"] = (X.race == "White").astype(int); COV += ["race_black","race_hisp","race_white"]
X[COV] = X[COV].astype(float).fillna(0)
FE = pd.get_dummies(X.market.astype(str) + "|" + X.q.astype(str), prefix="fe", drop_first=True, dtype=float); ROLE = pd.get_dummies(X.role.fillna("NA"), prefix="role", drop_first=True, dtype=float)
def resid(v, M): return v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
out = {"design": "one row per patient; instrument = assigned first-contact staff member's leave-one-patient-out action rate", "n_patients": int(len(X)), "n_staff": int(X.staff.nunique()), "outcome": "disengagement in days 31 to 120 after activation", "outcome_rate": round(float(X.y.mean()), 4), "actions": {}}
Mfe = np.column_stack([np.ones(len(X)), FE.values])
for a in ACTIONS:
    z = X[f"z_{a}"].values; A = X[a].values.astype(float); y = X.y.values.astype(float); rec = {"action_rate": round(float(A.mean()), 4), "instrument_sd": round(float(z.std()), 4)}
    exog = pd.concat([pd.Series(1.0, index=X.index, name="const"), X[COV], FE], axis=1)
    # OLS association
    ols = sm.OLS(y, pd.concat([exog, X[[a]]], axis=1).values).fit(cov_type="cluster", cov_kwds={"groups": X.staff.astype("category").cat.codes.values})
    rec["ols_rd"] = round(float(ols.params[-1]), 4); rec["ols_ci_95"] = [round(float(v), 4) for v in ols.conf_int()[-1]]
    # 2SLS
    iv = IV2SLS(y, exog, X[[a]], X[[f"z_{a}"]]).fit(cov_type="clustered", clusters=X.staff.astype("category").cat.codes.values)
    fs = iv.first_stage.diagnostics; rec["first_stage_coef"] = round(float(iv.first_stage.individual[a].params[f"z_{a}"]), 4); rec["first_stage_F"] = round(float(fs.loc[a, "f.stat"]), 1)
    rec["late_rd"] = round(float(iv.params[a]), 4); rec["late_ci_95"] = [round(float(v), 4) for v in iv.conf_int().loc[a]]; rec["late_se"] = round(float(iv.std_errors[a]), 4); rec["late_p"] = round(float(iv.pvalues[a]), 4)
    rf = sm.OLS(y, pd.concat([exog, X[[f"z_{a}"]]], axis=1).values).fit(cov_type="cluster", cov_kwds={"groups": X.staff.astype("category").cat.codes.values}); rec["reduced_form_coef"] = round(float(rf.params[-1]), 4); rec["reduced_form_p"] = round(float(rf.pvalues[-1]), 4)
    # Gate 2 balance: standardized covariate difference per SD of instrument, after FE
    zr = resid(z, Mfe); bal = {}
    for c in COV:
        cr = resid(X[c].values.astype(float), Mfe); b = np.polyfit(zr, cr, 1)[0] * zr.std() / (cr.std() if cr.std() > 0 else 1); bal[c] = round(float(b), 3)
    joint = sm.OLS(zr, sm.add_constant(np.column_stack([resid(X[c].values.astype(float), Mfe) for c in COV]))).fit(cov_type="cluster", cov_kwds={"groups": X.staff.astype("category").cat.codes.values})
    rec["balance_std_diff_per_sd"] = bal; rec["balance_max_abs"] = round(float(max(abs(v) for v in bal.values())), 3); rec["balance_joint_p"] = round(float(joint.f_pvalue), 4)
    # Gate 3 monotonicity: first-stage sign by market (n >= 200)
    signs = {}
    for m, g in X.groupby("market"):
        if len(g) < 200: continue
        zz = g[f"z_{a}"].values; 
        if zz.std() == 0: continue
        signs[m] = round(float(np.polyfit(zz, g[a].values.astype(float), 1)[0]), 3)
    rec["first_stage_by_market"] = signs; rec["gate3_same_sign"] = bool(len(signs) > 0 and (all(v > 0 for v in signs.values()) or all(v < 0 for v in signs.values())))
    # Gate 4 exclusion: add other staff rates + role as controls
    others = [f"z_{b}" for b in ACTIONS if b != a]; exog2 = pd.concat([exog, X[others], ROLE], axis=1)
    try:
        iv2 = IV2SLS(y, exog2, X[[a]], X[[f"z_{a}"]]).fit(cov_type="clustered", clusters=X.staff.astype("category").cat.codes.values)
        rec["late_rd_with_other_preferences_and_role"] = round(float(iv2.params[a]), 4); rec["late_ci_95_with_controls"] = [round(float(v), 4) for v in iv2.conf_int().loc[a]]; rec["gate4_change_lt_half_se"] = bool(abs(iv2.params[a] - iv.params[a]) < 0.5 * iv.std_errors[a])
    except Exception as e: rec["gate4_error"] = str(e)
    rec["gate1_F_ge_10"] = bool(rec["first_stage_F"] >= 10); rec["gate2_balance"] = bool(rec["balance_max_abs"] <= 0.10 and rec["balance_joint_p"] > 0.05)
    rec["all_gates_pass"] = bool(rec["gate1_F_ge_10"] and rec["gate2_balance"] and rec["gate3_same_sign"] and rec.get("gate4_change_lt_half_se", False))
    # subgroup LATEs
    sub = {}
    X["age45"] = (X.age >= 45).astype(int); X["risk_hi"] = (X.risk >= X.risk.median()).astype(int); X["adt_any"] = (X.adt365 > 0).astype(int)
    for name, mask in [("state", None), ("age45", None), ("any_bh", None), ("risk_hi", None), ("adt_any", None)]:
        for lev, g in X.groupby(name):
            if len(g) < 300 or g[f"z_{a}"].std() == 0: continue
            ex = pd.concat([pd.Series(1.0, index=g.index, name="const"), g[COV], pd.get_dummies(g.market.astype(str) + "|" + g.q.astype(str), drop_first=True, dtype=float)], axis=1)
            try:
                r_ = IV2SLS(g.y.values.astype(float), ex, g[[a]], g[[f"z_{a}"]]).fit(cov_type="clustered", clusters=g.staff.astype("category").cat.codes.values)
                sub[f"{name}={lev}"] = {"n": int(len(g)), "late_rd": round(float(r_.params[a]), 4), "ci_95": [round(float(v), 4) for v in r_.conf_int().loc[a]], "first_stage_F": round(float(r_.first_stage.diagnostics.loc[a, "f.stat"]), 1)}
            except Exception as e: sub[f"{name}={lev}"] = {"n": int(len(g)), "error": str(e)[:80]}
    rec["subgroups"] = sub
    # DRIV heterogeneity (econml), effects by quartile of learned score on cross-fitted folds
    try:
        from econml.iv.dr import LinearDRIV
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        W = np.column_stack([X[COV].values, FE.values]); Xh = X[["age","risk","any_bh","adt365","female","race_black","race_hisp"]].values
        est = LinearDRIV(model_y_xw=RandomForestRegressor(n_estimators=200, min_samples_leaf=50, random_state=1), model_t_xw=RandomForestClassifier(n_estimators=200, min_samples_leaf=50, random_state=1), model_z_xw=RandomForestRegressor(n_estimators=200, min_samples_leaf=50, random_state=1), model_tz_xw=RandomForestRegressor(n_estimators=200, min_samples_leaf=50, random_state=1), discrete_treatment=True, cv=3, random_state=1)
        est.fit(y, A, Z=z, X=Xh, W=W); te = est.effect(Xh); q = pd.qcut(te, 4, labels=False, duplicates="drop")
        rec["driv_effect_by_quartile"] = {str(int(k)): {"n": int((q == k).sum()), "mean_effect": round(float(te[q == k].mean()), 4)} for k in sorted(pd.unique(q))}
        rec["driv_ate"] = round(float(te.mean()), 4)
    except Exception as e: rec["driv_error"] = str(e)[:120]
    out["actions"][a] = rec; print(a, {k: rec[k] for k in ["action_rate","first_stage_F","late_rd","late_ci_95","ols_rd","balance_max_abs","balance_joint_p","gate3_same_sign","all_gates_pass"]}, flush=True)
json.dump(out, open(R/"staff_iv.json", "w"), indent=1, default=str); print("saved results/staff_iv.json")
