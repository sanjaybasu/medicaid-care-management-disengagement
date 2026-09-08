"""Step 6: identified levers (Section 7). Exposures at t: L1 in-person vs remote; L2 pharmacist; L3 therapy; L4 discipline change.
Outcome: no completed contact in days 31-120 and not graduated, >=60 enrolled days in that window.
Within-patient conditional logit on switchers (equipoise-trimmed), MSM co-estimator, pre-trend gate on prior engagement bins, marginal sensitivity bounds.
Writes results/levers.json"""
import json, pathlib, warnings, numpy as np, pandas as pd, statsmodels.api as sm; warnings.filterwarnings("ignore")
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.ensemble import HistGradientBoostingClassifier
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
df = F.merge(O[["decision_id","lever_eligible","y_lever","enrolled_days_31_120"]], on="decision_id"); df = df[df.lever_eligible == 1].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); E = {p: g.enc_date.values for p, g in enc.dropna(subset=["enc_date"]).groupby("person_id")}
D1 = np.timedelta64(1, "D")
# prior engagement bins (contacts in -90..-61, -60..-31, -30..-1) for the pre-trend gate
pre = np.zeros((len(df), 3))
for i, r in enumerate(df.itertuples()):
    d = E.get(r.person_id); t = r.enc_date.to_datetime64()
    if d is None: continue
    for j, (lo, hi) in enumerate([(90, 61), (60, 31), (30, 1)]): pre[i, j] = ((d >= t - lo*D1) & (d <= t - hi*D1)).sum()
df["pre3"], df["pre2"], df["pre1"] = pre[:,0], pre[:,1], pre[:,2]
df["L1_inperson"] = df.cur_inp.astype(int); df["L2_pharm"] = (df.cur_rg == "PharmD").astype(int); df["L3_therapy"] = (df.cur_rg == "Therapist").astype(int); df["L4_disc_change"] = ((df.prev_rg != "none") & (df.prev_rg != df.cur_rg)).astype(int)
df["cal_month"] = df.enc_date.dt.to_period("M").astype(str); df["tstrat"] = pd.cut(df.days_from_zd, [-1, 30, 90, 366], labels=["0_30","31_90","91_365"]).astype(str)
COV = ["days_from_zd","n_prior_90","days_since_last_contact","adt_all_prior_90d","risk_percentile","goals_open","pre1","pre2","pre3"]
base_rate = float(df.y_lever.mean()); margin = 0.10 * base_rate
Xps = pd.concat([df[COV].astype(float), pd.get_dummies(df[["gender","race","state","cur_rg","prev_rg"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0)
out = {"base_rate": round(base_rate, 4), "equivalence_margin_abs": round(margin, 4), "n_lever_eligible": int(len(df)), "exposures": {}}
def boot_ci(fn, ids, B=500, seed=20260907):
    rng = np.random.default_rng(seed); up = np.unique(ids); idx = {p: np.where(ids == p)[0] for p in up}; vals = []
    for _ in range(B):
        ix = np.concatenate([idx[p] for p in rng.choice(up, len(up))]); v = fn(ix)
        if v is not None and np.isfinite(v): vals.append(v)
    vals = np.array(vals); return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)], round(float(2*min((vals >= 0).mean(), (vals <= 0).mean())), 4)
for L in ["L1_inperson","L2_pharm","L3_therapy","L4_disc_change"]:
    A = df[L].values; y = df.y_lever.values; res = {"prevalence": round(float(A.mean()), 4)}
    # trimming propensity fit on training era, applied to all
    tr = (df.training_era == 1).values; ps_model = HistGradientBoostingClassifier(**HP).fit(Xps.values[tr], A[tr]); ps = np.clip(ps_model.predict_proba(Xps.values)[:,1], 1e-3, 1-1e-3)
    keep = (ps >= 0.10) & (ps <= 0.90); sub = df[keep].copy(); sub["ps"] = ps[keep]; sub["A"] = A[keep]; sub["y"] = y[keep]
    if len(sub) < 200 or sub.A.nunique() < 2 or (sub.A == 1).sum() < 50:
        out["exposures"][L] = {"prevalence": round(float(A.mean()), 4), "equipoise_dps": int(keep.sum()), "note": "insufficient equipoise support (fewer than 200 decision points or fewer than 50 exposed); not estimable"}; print(L, out["exposures"][L], flush=True); continue
    sw = sub.groupby("person_id").A.transform(lambda s: s.nunique() > 1); sw2 = sub.groupby("person_id").y.transform(lambda s: s.nunique() > 1)
    inf = sub.loc[(sw & sw2).values].copy()  # informative for conditional logit: switch in exposure and in outcome
    res.update(equipoise_dps=int(keep.sum()), equipoise_share=round(float(keep.mean()), 3), switcher_patients=int(sub.loc[sw.values].person_id.nunique()), informative_patients=int(inf.person_id.nunique()), informative_dps=int(len(inf)))
    # Gate 1: pre-period engagement bins, within-patient de-meaned, exposed vs unexposed (contacts per 30 days -> rate difference as share of base contact rate)
    g1 = {}; pass1 = True
    for b in ["pre3","pre2","pre1"]:
        dm = sub[b] - sub.groupby("person_id")[b].transform("mean"); diff = float(dm[sub.A == 1].mean() - dm[sub.A == 0].mean())
        ci, p = boot_ci(lambda ix: float(dm.values[ix][sub.A.values[ix]==1].mean() - dm.values[ix][sub.A.values[ix]==0].mean()), sub.person_id.values, B=300)
        g1[b] = {"diff_contacts_per_30d": round(diff, 4), "ci_95": ci, "p": p}
        if not (ci[0] <= 0 <= ci[1]) or abs(diff) > 0.10 * float(sub[b].mean()): pass1 = False
    res["gate1_pretrend"] = {"bins": g1, "passed": bool(pass1), "rule": "each pre-bin difference within 10% of mean bin contacts and CI covering zero"}
    # Within-patient conditional logit (fixed effects) with time-varying covariates
    if len(inf) >= 200 and inf.person_id.nunique() >= 50:
        fitted = None
        for cov in (COV, ["days_from_zd","n_prior_90","days_since_last_contact","risk_percentile","goals_open"], []):
            try:
                Xc = inf[["A"] + cov].astype(float).fillna(0); Xc = pd.concat([Xc, pd.get_dummies(inf.tstrat, drop_first=True, dtype=float)], axis=1); Xc = Xc.loc[:, Xc.std() > 0]
                cl = ConditionalLogit(inf.y.values, Xc.values, groups=inf.person_id.values).fit(disp=0); b, se = float(cl.params[0]), float(cl.bse[0]); fitted = (b, se, float(cl.pvalues[0]), list(Xc.columns)); break
            except Exception as e: err = str(e)[:200]
        if fitted:
            b, se, pv, used = fitted; or_, lo, hi = np.exp(b), np.exp(b-1.96*se), np.exp(b+1.96*se); rd = float(base_rate * (or_ - 1) / (1 + base_rate * (or_ - 1)) * (1 - base_rate))
            res["within_patient_fe"] = {"odds_ratio": round(or_, 3), "ci_95": [round(lo, 3), round(hi, 3)], "p": round(pv, 4), "approx_risk_difference": round(rd, 4), "n_dps": int(len(inf)), "n_patients": int(inf.person_id.nunique()), "covariates_used": used, "mde_or_80pct_power": round(float(np.exp(2.802*se)), 2)}
        else: res["within_patient_fe"] = {"error": err}
    else: res["within_patient_fe"] = {"note": "insufficient informative switchers", "informative_patients": int(inf.person_id.nunique())}
    # MSM (stabilized, truncated weights) on the equipoise sample, patient-clustered
    pa = float(sub.A.mean()); w = np.where(sub.A == 1, pa/sub.ps, (1-pa)/(1-sub.ps)); w = np.clip(w, np.percentile(w, 1), np.percentile(w, 99))
    Xm = sm.add_constant(pd.concat([sub[["A"]].astype(float), pd.get_dummies(sub.tstrat, drop_first=True, dtype=float)], axis=1))
    msm = sm.GLM(sub.y.values, Xm, family=sm.families.Binomial(), freq_weights=w).fit(cov_type="cluster", cov_kwds={"groups": sub.person_id.values})
    b, se = float(msm.params["A"]), float(msm.bse["A"]); res["msm_iptw"] = {"odds_ratio": round(np.exp(b), 3), "ci_95": [round(np.exp(b-1.96*se), 3), round(np.exp(b+1.96*se), 3)], "p": round(float(msm.pvalues["A"]), 4),
        "risk_difference": round(float(np.average(sub.y[sub.A==1], weights=w[sub.A==1]) - np.average(sub.y[sub.A==0], weights=w[sub.A==0])), 4)}
    # Gate 2 agreement (odds ratios within each other's CIs)
    fe = res["within_patient_fe"]; res["gate2_agreement"] = bool("odds_ratio" in fe and fe["ci_95"][0] <= res["msm_iptw"]["odds_ratio"] <= fe["ci_95"][1] and res["msm_iptw"]["ci_95"][0] <= fe["odds_ratio"] <= res["msm_iptw"]["ci_95"][1])
    # marginal sensitivity bounds (Tan/Kallus-Zhou style) on the IPW risk difference for Gamma in {1.25, 1.5, 2}
    bounds = {}
    for G in [1.25, 1.5, 2.0]:
        def bound(sign):
            lo_w, hi_w = w/G, w*G; ww = np.where(((sub.y == 1) == (sign > 0)), hi_w, lo_w); ww0 = np.where(((sub.y == 1) == (sign > 0)), lo_w, hi_w)
            return float(np.average(sub.y[sub.A==1], weights=ww[sub.A==1]) - np.average(sub.y[sub.A==0], weights=ww0[sub.A==0]))
        bounds[f"gamma_{G}"] = [round(bound(-1), 4), round(bound(+1), 4)]
    res["sensitivity_bounds_risk_difference"] = bounds
    # timing stratification (MSM RD by stratum)
    res["by_timing"] = {}
    for s_ in ["0_30","31_90","91_365"]:
        mk = (sub.tstrat == s_).values
        if mk.sum() > 300 and sub.A[mk].nunique() == 2: res["by_timing"][s_] = {"n": int(mk.sum()), "iptw_risk_difference": round(float(np.average(sub.y[mk][sub.A[mk]==1], weights=w[mk][sub.A[mk]==1]) - np.average(sub.y[mk][sub.A[mk]==0], weights=w[mk][sub.A[mk]==0])), 4)}
    out["exposures"][L] = res; print(L, json.dumps({k: res[k] for k in ["prevalence","equipoise_share","informative_patients","gate1_pretrend","within_patient_fe","msm_iptw","gate2_agreement","sensitivity_bounds_risk_difference"]}, default=str)[:1500], flush=True)
json.dump(out, open(R/"levers.json", "w"), indent=1, default=str); print("saved results/levers.json")
