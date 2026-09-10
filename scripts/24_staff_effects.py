"""Step 24 (amendment v3.4): assigned-staff effects on disengagement in days 31-120, with split-sample validation, balance, heterogeneity,
and descriptive association with the staff member's action profile. Reuses the patient-level construction of script 23. Writes results/staff_effects.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import statsmodels.api as sm, statsmodels.formula.api as smf
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); rng = np.random.default_rng(20260910)
F = pd.read_parquet(D/"dp_features.parquet"); S = pd.read_parquet(D/"state_features.parquet")[["encounter_id","created_by_id","roles","zd"]].rename(columns={"encounter_id":"decision_id"})
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"])
sh = pd.read_parquet(D/"status_history.parquet"); sh["dt"] = pd.to_datetime(sh.updated_at)
sp = pd.read_parquet(D/"enrollment_spans.parquet"); sp["s"] = pd.to_datetime(sp.enrollment_start_date); sp["e"] = pd.to_datetime(sp.enrollment_end_date).fillna(pd.Timestamp("2026-12-31"))
d = F.merge(S, on="decision_id"); d["enc_date"] = pd.to_datetime(d.enc_date); d["zd"] = pd.to_datetime(d.zd)
INP = {"HOME_VISIT","OTHER_INPERSON","HOSPITAL","CBO","IN_COMMUNITY","PROVIDER_OFFICE"}; GRAD = {"GRADUATED","PREGRADUATION"}; D1 = np.timedelta64(1, "D")
first = d.sort_values("enc_date").groupby("person_id").head(1).set_index("person_id")
E = {p: g for p, g in enc.groupby("person_id")}; SH = {p: (g.dt.values, g.new_status.values) for p, g in sh.groupby("person_id")}; SP = {p: (g.s.values, g.e.values) for p, g in sp.groupby("person_id")}
def enrolled_all(p, a, b):
    s, e = SP.get(p, (np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")))
    if len(s) == 0: return False
    lo = np.maximum(s, a); hi = np.minimum(e, b); days = ((hi - lo) / D1).astype(float) + 1; return float(np.clip(days, 0, None).sum()) >= float((b - a) / D1) + 1
rows = []
for p, r in first.iterrows():
    z = r.zd.to_datetime64() if pd.notna(r.zd) else r.enc_date.to_datetime64(); g = E.get(p)
    if g is None or r.enc_date.to_datetime64() > z + 30*D1 or (z + 120*D1) > np.datetime64("2026-06-07") or not enrolled_all(p, z + 31*D1, z + 120*D1): continue
    dt = g.enc_date.values; ct = g.contact_type.values; w30 = (dt >= z) & (dt <= z + 30*D1); w31 = (dt > z + 30*D1) & (dt <= z + 120*D1); d30 = np.sort(dt[w30])
    sd, ss = SH.get(p, (np.array([], dtype="datetime64[ns]"), np.array([]))); grad = bool(set(ss[(sd > z) & (sd <= z + 120*D1)]) & GRAD)
    rows.append(dict(person_id=p, staff=r.created_by_id, role=r.roles, market=r.market, state=r.state, q=pd.Timestamp(z).to_period("Q").strftime("%Y-Q%q"), y=int((not w31.any()) and (not grad)),
                     inperson=int(np.isin(ct[w30], list(INP)).any()), fu7=int(len(d30) >= 2 and (d30[1] - d30[0]) / D1 <= 7), weekend=int(any(pd.Timestamp(x).weekday() >= 5 for x in d30)), n30=int(w30.sum()),
                     age=r.age, female=int(r.gender == "Female"), race=r.race, risk=r.risk_percentile, any_bh=r.any_bh, sud=r.sud, diabetes=r.diabetes, htn=r.htn, chf=r.chf, copd=r.copd, asthma=r.asthma, polypharmacy=r.polypharmacy, high_ed_ip=r.high_ed_ip, no_pcp=r.no_pcp_last_10mo, adt365=r.adt_all_prior_365d))
X = pd.DataFrame(rows); X = X[X.staff.notna()]; X = X[X.staff.map(X.groupby("staff").size()) >= 20].reset_index(drop=True)
COV = ["age","female","risk","any_bh","sud","diabetes","htn","chf","copd","asthma","polypharmacy","high_ed_ip","no_pcp","adt365"]
X["race_black"] = (X.race == "Black or African American").astype(int); X["race_hisp"] = (X.race == "Hispanic").astype(int); X["race_white"] = (X.race == "White").astype(int); COV += ["race_black","race_hisp","race_white"]
X[COV] = X[COV].astype(float).fillna(0); X["mq"] = X.market.astype(str) + "|" + X.q.astype(str); X["staff_c"] = X.staff.astype("category").cat.codes
print(f"patients {len(X)}; staff {X.staff.nunique()}; outcome rate {X.y.mean():.3f}", flush=True)

# ---- CHW-only assignment: first community health worker contact within 30 days of activation
dchw = d[d.roles.isin(["CHW", "Market CHW Lead"])].sort_values("enc_date")
firstchw = dchw.groupby("person_id").head(1).set_index("person_id")
rows_c = []
for p, r in firstchw.iterrows():
    z = r.zd.to_datetime64() if pd.notna(r.zd) else r.enc_date.to_datetime64(); g = E.get(p)
    if g is None or r.enc_date.to_datetime64() > z + 30*D1 or (z + 120*D1) > np.datetime64("2026-06-07") or not enrolled_all(p, z + 31*D1, z + 120*D1): continue
    dt = g.enc_date.values; ct = g.contact_type.values; w30 = (dt >= z) & (dt <= z + 30*D1); w31 = (dt > z + 30*D1) & (dt <= z + 120*D1); d30 = np.sort(dt[w30])
    sd, ss = SH.get(p, (np.array([], dtype="datetime64[ns]"), np.array([]))); grad = bool(set(ss[(sd > z) & (sd <= z + 120*D1)]) & GRAD)
    rows_c.append(dict(person_id=p, staff=r.created_by_id, role=r.roles, market=r.market, state=r.state, q=pd.Timestamp(z).to_period("Q").strftime("%Y-Q%q"), y=int((not w31.any()) and (not grad)),
                     inperson=int(np.isin(ct[w30], list(INP)).any()), fu7=int(len(d30) >= 2 and (d30[1] - d30[0]) / D1 <= 7), weekend=int(any(pd.Timestamp(x).weekday() >= 5 for x in d30)), n30=int(w30.sum()),
                     age=r.age, female=int(r.gender == "Female"), race=r.race, risk=r.risk_percentile, any_bh=r.any_bh, sud=r.sud, diabetes=r.diabetes, htn=r.htn, chf=r.chf, copd=r.copd, asthma=r.asthma, polypharmacy=r.polypharmacy, high_ed_ip=r.high_ed_ip, no_pcp=r.no_pcp_last_10mo, adt365=r.adt_all_prior_365d))
XC = pd.DataFrame(rows_c); XC = XC[XC.staff.notna()]; XC = XC[XC.staff.map(XC.groupby("staff").size()) >= 20].reset_index(drop=True)
XC["race_black"] = (XC.race == "Black or African American").astype(int); XC["race_hisp"] = (XC.race == "Hispanic").astype(int); XC["race_white"] = (XC.race == "White").astype(int)
XC[COV] = XC[COV].astype(float).fillna(0); XC["mq"] = XC.market.astype(str) + "|" + XC.q.astype(str); XC["staff_c"] = XC.staff.astype("category").cat.codes
X_all = X.copy(); X_all["mq"] = X_all.market.astype(str) + "|" + X_all.q.astype(str) + "|" + X_all.role.fillna("NA").astype(str)   # role in the fixed effects
def analyze(X, label):
    rng = np.random.default_rng(20260910)
    covf = " + ".join(COV)
    # ---- fixed-effects staff model
    fe = smf.ols(f"y ~ C(mq) + {covf} + C(staff)", data=X).fit(cov_type="cluster", cov_kwds={"groups": X.staff_c.values})
    base = smf.ols(f"y ~ C(mq) + {covf}", data=X).fit()
    eff = {k.split("[T.")[1].rstrip("]"): v for k, v in fe.params.items() if k.startswith("C(staff)[T.")}; ref = [s for s in X.staff.unique() if s not in eff][0]; eff[ref] = 0.0
    E_ = pd.Series(eff); mk = X.groupby("staff").market.agg(lambda s: s.mode().iloc[0]); E_c = E_ - E_.groupby(mk.reindex(E_.index)).transform("mean")   # centered within market
    # ---- mixed model shrinkage
    mm = smf.mixedlm(f"y ~ C(mq) + {covf}", data=X, groups=X["staff"]).fit(reml=True)
    re = pd.Series({k: float(v.iloc[0]) for k, v in mm.random_effects.items()}); var_staff = float(mm.cov_re.iloc[0, 0]); var_res = float(mm.scale)
    q25, q75 = re.quantile(.25), re.quantile(.75); lo_staff = re[re <= q25].index; hi_staff = re[re >= q75].index
    pred_base = base.fittedvalues
    out = {"design": label, "n_patients": int(len(X)), "n_staff": int(X.staff.nunique()), "outcome_rate": round(float(X.y.mean()), 4),
           "r2_without_staff": round(float(base.rsquared), 4), "r2_with_staff": round(float(fe.rsquared), 4), "f_test_staff_p": round(float(fe.compare_f_test(base)[1]), 6),
           "staff_effect_sd_fixed_centered": round(float(E_c.std()), 4), "staff_effect_sd_shrunken": round(float(np.sqrt(var_staff)), 4), "icc": round(var_staff / (var_staff + var_res), 4),
           "shrunken_effect_quantiles": {str(k): round(float(v), 4) for k, v in re.quantile([.1, .25, .5, .75, .9]).items()},
           "mean_outcome_bottom_quartile_staff": round(float(X[X.staff.isin(lo_staff)].y.mean()), 4), "mean_outcome_top_quartile_staff": round(float(X[X.staff.isin(hi_staff)].y.mean()), 4),
           "adjusted_gap_top_minus_bottom_quartile": round(float((X[X.staff.isin(hi_staff)].y - pred_base[X.staff.isin(hi_staff)]).mean() - (X[X.staff.isin(lo_staff)].y - pred_base[X.staff.isin(lo_staff)]).mean()), 4),
           "n_patients_bottom_quartile": int(X.staff.isin(lo_staff).sum()), "n_patients_top_quartile": int(X.staff.isin(hi_staff).sum())}
    ci = fe.conf_int(); sig = sum(1 for k in fe.params.index if k.startswith("C(staff)[T.") and (ci.loc[k, 0] > 0 or ci.loc[k, 1] < 0)); out["share_staff_differing_from_reference_95"] = round(sig / max(len(eff) - 1, 1), 3)
    # ---- split-sample validation and balance
    X["half"] = X.groupby("staff").cumcount().pipe(lambda s: rng.permutation(len(s)) % 2) if False else 0
    X["half"] = X.groupby("staff", group_keys=False).apply(lambda g: pd.Series(rng.permutation(np.arange(len(g)) % 2), index=g.index))
    A, B = X[X.half == 0], X[X.half == 1]
    mmA = smf.mixedlm(f"y ~ C(mq) + {covf}", data=A, groups=A["staff"]).fit(reml=True); reA = pd.Series({k: float(v.iloc[0]) for k, v in mmA.random_effects.items()})
    B = B.assign(effA=B.staff.map(reA)).dropna(subset=["effA"])
    val = smf.ols(f"y ~ C(mq) + {covf} + effA", data=B).fit(cov_type="cluster", cov_kwds={"groups": B.staff.astype("category").cat.codes.values})
    out["validation"] = {"n_B": int(len(B)), "forecast_coef": round(float(val.params["effA"]), 3), "ci_95": [round(float(v), 3) for v in val.conf_int().loc["effA"]], "p": round(float(val.pvalues["effA"]), 5), "interpretation": "1 = A-estimated staff effects forecast B outcomes one for one; 0 = noise"}
    Mfe = pd.get_dummies(B.mq, drop_first=False, dtype=float).values
    def resid(v, M): return v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
    zr = resid(B.effA.values, Mfe); bal = {}
    for c in COV:
        cr = resid(B[c].values.astype(float), Mfe); bal[c] = round(float(np.polyfit(zr, cr, 1)[0] * zr.std() / (cr.std() if cr.std() > 0 else 1)), 3)
    joint = sm.OLS(zr, sm.add_constant(np.column_stack([resid(B[c].values.astype(float), Mfe) for c in COV]))).fit(cov_type="cluster", cov_kwds={"groups": B.staff.astype("category").cat.codes.values})
    out["balance_B_on_effA"] = {"std_diff_per_sd": bal, "max_abs": round(float(max(abs(v) for v in bal.values())), 3), "joint_p": round(float(joint.f_pvalue), 4), "pass": bool(max(abs(v) for v in bal.values()) <= 0.10 and joint.f_pvalue > 0.05)}
    # ---- heterogeneity: interaction of effA with patient characteristics in B
    B = B.assign(age45=(B.age >= 45).astype(int), risk_hi=(B.risk >= X.risk.median()).astype(int), adt_any=(B.adt365 > 0).astype(int), bh=B.any_bh.astype(int), va=(B.state == "VIRGINIA").astype(int))
    het = {}
    for v in ["age45", "bh", "risk_hi", "adt_any", "race_black", "race_hisp", "female"]:
        m = smf.ols(f"y ~ C(mq) + {covf} + effA + effA:{v}", data=B).fit(cov_type="cluster", cov_kwds={"groups": B.staff.astype("category").cat.codes.values})
        het[v] = {"effA_coef_when_0": round(float(m.params["effA"]), 3), "interaction": round(float(m.params[f"effA:{v}"]), 3), "interaction_ci_95": [round(float(x), 3) for x in m.conf_int().loc[f"effA:{v}"]], "p": round(float(m.pvalues[f"effA:{v}"]), 4)}
    out["heterogeneity_B"] = het
    # ---- descriptive association of staff effect with action profile (full-sample shrunken effects; lower effect = less disengagement)
    prof = X.groupby("staff").agg(n=("y", "size"), n30=("n30", "mean"), inperson=("inperson", "mean"), fu7=("fu7", "mean"), weekend=("weekend", "mean"), role=("role", lambda s: s.mode().iloc[0] if s.notna().any() else "NA")).assign(effect=re)
    out["staff_profile_correlation_with_effect"] = {k: round(float(prof[k].corr(prof.effect)), 3) for k in ["n30", "inperson", "fu7", "weekend"]}
    out["staff_profile_by_effect_quartile"] = {lab: {k: round(float(prof.loc[idx, k].mean()), 3) for k in ["n30", "inperson", "fu7", "weekend"]} | {"n_staff": int(len(idx))} for lab, idx in [("lowest_disengagement_quartile", lo_staff), ("highest_disengagement_quartile", hi_staff)]}
    out["staff_effect_by_role"] = prof.groupby("role").effect.agg(["mean", "size"]).round(4).to_dict("index")

    return out

RES = {"all_staff_role_adjusted": analyze(X_all, "all staff; assigned first-contact staff member; fixed effects for market x quarter x role; baseline covariates"),
       "chw_only": analyze(XC, "community health workers only; assigned CHW = first CHW contact within 30 days; market x quarter fixed effects; baseline covariates")}
json.dump(RES, open(R/"staff_effects.json", "w"), indent=1, default=str)
for k, o in RES.items(): print(k, json.dumps({kk: o[kk] for kk in ["n_patients","n_staff","outcome_rate","icc","staff_effect_sd_shrunken","mean_outcome_bottom_quartile_staff","mean_outcome_top_quartile_staff","adjusted_gap_top_minus_bottom_quartile","validation","balance_B_on_effA","staff_profile_correlation_with_effect","staff_profile_by_effect_quartile"]}, default=str)[:1800]); print("  HET", {kk: (v["interaction"], v["interaction_ci_95"]) for kk, v in o["heterogeneity_B"].items()})
