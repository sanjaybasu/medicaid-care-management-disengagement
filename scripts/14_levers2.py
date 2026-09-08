"""Step 14 (amendment v3.1 Section D): additional levers. L5 time to second contact (<=7 days vs longer) after a member's first contact;
L6 same-person continuity (next contact by the same care team member vs different). Outcome: no completed contact in days 31-120 after the
second/next contact and no graduation by day 120, >=60 enrolled days in that window, second/next contact dated <= 2026-05-09.
L6 uses the within-member conditional logit with pre-trend gate and MSM as in 06; L5 has one index per member and uses the MSM with equipoise trimming
and sensitivity bounds only. Writes results/levers2.json."""
import json, pathlib, warnings, numpy as np, pandas as pd, statsmodels.api as sm; warnings.filterwarnings("ignore")
from statsmodels.discrete.conditional_models import ConditionalLogit
from sklearn.ensemble import HistGradientBoostingClassifier
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]; END = pd.Timestamp("2026-05-09"); D1 = np.timedelta64(1, "D")
F = pd.read_parquet(D/"dp_features.parquet"); F["enc_date"] = pd.to_datetime(F.enc_date)
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])
sh = pd.read_parquet(D/"status_history.parquet"); sh["dt"] = pd.to_datetime(sh.updated_at); GRAD = {"GRADUATED","PREGRADUATION"}
sp = pd.read_parquet(D/"enrollment_spans.parquet"); sp["s"] = pd.to_datetime(sp.enrollment_start_date); sp["e"] = pd.to_datetime(sp.enrollment_end_date).fillna(pd.Timestamp("2026-12-31"))
E = {p: (g.enc_date.values, g.created_by_id.values, g.encounter_id.values) for p, g in enc.groupby("person_id")}; S = {p: (g.dt.values, g.new_status.values) for p, g in sh.groupby("person_id")}; SP = {p: (g.s.values, g.e.values) for p, g in sp.groupby("person_id")}
def enrolled(p, a, b):
    s, e = SP.get(p, (np.array([], dtype="datetime64[ns]"),)*2)
    if len(s) == 0: return 0.0
    lo = np.maximum(s, a); hi = np.minimum(e, b); return float(np.clip(((hi-lo)/D1).astype(float)+1, 0, None).sum())
def outcome(p, t2):
    d, _, _ = E[p]; sd, ss = S.get(p, (np.array([], dtype="datetime64[ns]"), np.array([])))
    w = (d > t2 + 30*D1) & (d <= t2 + 120*D1); g = bool(set(ss[(sd > t2) & (sd <= t2 + 120*D1)]) & GRAD); el = enrolled(p, t2 + 31*D1, t2 + 120*D1) >= 60 and pd.Timestamp(t2) <= END
    return int((not w.any()) and not g), el
# ---- build index rows: for each decision point with a next contact within 120 days
rows = []
for r in F.itertuples():
    d, who, ids = E.get(r.person_id, (None, None, None))
    if d is None: continue
    t = r.enc_date.to_datetime64(); i = np.searchsorted(d, t, side="right")
    if i >= len(d): continue
    t2 = d[i]; gap = float((t2 - t)/D1)
    if gap > 120: continue
    y, el = outcome(r.person_id, t2); same = int(who[i] == who[i-1]) if i >= 1 and pd.notna(who[i]) and pd.notna(who[i-1]) else np.nan
    pre = [int(((d >= t - lo*D1) & (d <= t - hi*D1)).sum()) for lo, hi in [(90,61),(60,31),(30,1)]]
    rows.append(dict(decision_id=r.decision_id, person_id=r.person_id, t=r.enc_date, t2=pd.Timestamp(t2), gap=gap, first_contact=int(r.n_prior == 0), same_person=same, y=y, eligible=int(el), pre3=pre[0], pre2=pre[1], pre1=pre[2], training_era=r.training_era, days_from_zd=r.days_from_zd, n_prior_90=r.n_prior_90, days_since_last_contact=r.days_since_last_contact, adt_all_prior_90d=r.adt_all_prior_90d, risk_percentile=r.risk_percentile, goals_open=r.goals_open, gender=r.gender, race=r.race, state=r.state, cur_rg=r.cur_rg, prev_rg=r.prev_rg, encounter_type=r.encounter_type, lex_declined_cur=r.lex_declined_cur, lex_positive_engagement_cur=r.lex_positive_engagement_cur, age=r.age))
X = pd.DataFrame(rows); X = X[X.eligible == 1].reset_index(drop=True)
COV = ["days_from_zd","n_prior_90","days_since_last_contact","adt_all_prior_90d","risk_percentile","goals_open","pre1","pre2","pre3","lex_declined_cur","lex_positive_engagement_cur","age"]
def boot(fn, ids, B=300, seed=20260908):
    rng = np.random.default_rng(seed); up = np.unique(ids); idx = {p: np.where(ids == p)[0] for p in up}; v = []
    for _ in range(B):
        ix = np.concatenate([idx[p] for p in rng.choice(up, len(up))]); z = fn(ix)
        if z is not None and np.isfinite(z): v.append(z)
    v = np.array(v); return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)], round(float(2*min((v >= 0).mean(), (v <= 0).mean())), 4)
def analyze(sub, A, within=True):
    sub = sub.copy(); sub["A"] = A.astype(int); res = {"n": int(len(sub)), "prevalence": round(float(sub.A.mean()), 4), "base_rate": round(float(sub.y.mean()), 4)}
    Xps = pd.concat([sub[COV].astype(float).fillna(0), pd.get_dummies(sub[["gender","race","state","cur_rg","prev_rg","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1)
    tr = (sub.training_era == 1).values; ps = np.clip(HistGradientBoostingClassifier(**HP).fit(Xps.values[tr], sub.A.values[tr]).predict_proba(Xps.values)[:,1], 1e-3, 1-1e-3)
    keep = (ps >= 0.10) & (ps <= 0.90); s2 = sub[keep].copy(); s2["ps"] = ps[keep]; res.update(equipoise_dps=int(keep.sum()), equipoise_share=round(float(keep.mean()), 3))
    if len(s2) < 200 or (s2.A == 1).sum() < 50 or (s2.A == 0).sum() < 50: res["note"] = "insufficient equipoise support; not estimable"; return res
    if within:
        g1, ok = {}, True
        for b in ["pre3","pre2","pre1"]:
            dm = s2[b] - s2.groupby("person_id")[b].transform("mean"); diff = float(dm[s2.A == 1].mean() - dm[s2.A == 0].mean()); ci, p = boot(lambda ix: float(dm.values[ix][s2.A.values[ix]==1].mean() - dm.values[ix][s2.A.values[ix]==0].mean()), s2.person_id.values)
            g1[b] = {"diff_contacts_per_30d": round(diff, 4), "ci_95": ci, "p": p}; ok &= (ci[0] <= 0 <= ci[1]) and abs(diff) <= 0.10*float(s2[b].mean())
        res["gate1_pretrend"] = {"bins": g1, "passed": bool(ok)}
        sw = s2.groupby("person_id").A.transform(lambda s: s.nunique() > 1) & s2.groupby("person_id").y.transform(lambda s: s.nunique() > 1); inf = s2[sw.values]
        res["informative_patients"] = int(inf.person_id.nunique()); res["informative_dps"] = int(len(inf))
        if len(inf) >= 200 and inf.person_id.nunique() >= 50:
            fitted = None
            for cov in (COV, ["days_from_zd","n_prior_90","days_since_last_contact","risk_percentile","goals_open"], []):
                try:
                    Xc = inf[["A"] + cov].astype(float).fillna(0); Xc = Xc.loc[:, Xc.std() > 0]; cl = ConditionalLogit(inf.y.values, Xc.values, groups=inf.person_id.values).fit(disp=0); b, se = float(cl.params[0]), float(cl.bse[0]); fitted = (b, se, float(cl.pvalues[0]), list(Xc.columns)); break
                except Exception as e: err = str(e)[:120]
            if fitted: b, se, pv, used = fitted; res["within_member_fe"] = {"odds_ratio": round(float(np.exp(b)), 3), "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "p": round(pv, 4), "mde_or_80pct_power": round(float(np.exp(2.802*se)), 2), "covariates_used": used}
            else: res["within_member_fe"] = {"error": err}
        else: res["within_member_fe"] = {"note": "insufficient informative switchers"}
    pa = float(s2.A.mean()); w = np.where(s2.A == 1, pa/s2.ps, (1-pa)/(1-s2.ps)); w = np.clip(w, np.percentile(w, 1), np.percentile(w, 99))
    msm = sm.GLM(s2.y.values, sm.add_constant(s2[["A"]].astype(float)), family=sm.families.Binomial(), freq_weights=w).fit(cov_type="cluster", cov_kwds={"groups": s2.person_id.values}); b, se = float(msm.params["A"]), float(msm.bse["A"])
    res["msm_iptw"] = {"odds_ratio": round(float(np.exp(b)), 3), "ci_95": [round(float(np.exp(b-1.96*se)), 3), round(float(np.exp(b+1.96*se)), 3)], "p": round(float(msm.pvalues["A"]), 4), "risk_difference": round(float(np.average(s2.y[s2.A==1], weights=w[s2.A==1]) - np.average(s2.y[s2.A==0], weights=w[s2.A==0])), 4), "mde_or_80pct_power": round(float(np.exp(2.802*se)), 2)}
    if within and "odds_ratio" in res.get("within_member_fe", {}): fe = res["within_member_fe"]; res["gate2_agreement"] = bool(fe["ci_95"][0] <= res["msm_iptw"]["odds_ratio"] <= fe["ci_95"][1] and res["msm_iptw"]["ci_95"][0] <= fe["odds_ratio"] <= res["msm_iptw"]["ci_95"][1])
    bounds = {}
    for G in [1.25, 1.5, 2.0]:
        def bound(sign):
            hi_w, lo_w = w*G, w/G; ww = np.where((s2.y.values == 1) == (sign > 0), hi_w, lo_w); ww0 = np.where((s2.y.values == 1) == (sign > 0), lo_w, hi_w)
            return float(np.average(s2.y[s2.A==1], weights=ww[s2.A==1]) - np.average(s2.y[s2.A==0], weights=ww0[s2.A==0]))
        bounds[f"gamma_{G}"] = [round(bound(-1), 4), round(bound(+1), 4)]
    res["sensitivity_bounds_risk_difference"] = bounds; return res
out = {"index": "decision point with a next completed contact within 120 days; outcome measured from the next contact", "n_index_rows_eligible": int(len(X))}
first = X[X.first_contact == 1]; out["L5_second_contact_within_7_days"] = analyze(first, (first.gap <= 7).values, within=False); out["L5_second_contact_within_7_days"]["design"] = "one index per member (first contact); MSM with equipoise trimming and sensitivity bounds; no within-member estimator or pre-trend gate is possible"
cont = X.dropna(subset=["same_person"]); out["L6_same_person_continuity"] = analyze(cont, cont.same_person.values, within=True)
json.dump(out, open(R/"levers2.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
