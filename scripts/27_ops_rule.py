"""Step 27 (amendment v3.6): operational 'worth attempting' rule. Reads test_predictions, dp_features/outcomes, encounter_times_study (attempts),
simple_score.json. Writes results/ops_rule.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier as GB, GradientBoostingRegressor as GBR
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); D1 = np.timedelta64(1, "D"); rng = np.random.default_rng(20260910)
en = pd.read_parquet(D/"encounter_times_study.parquet"); en["t"] = pd.to_datetime(en.startTime.fillna(en.dateOfEncounter)); en = en.dropna(subset=["t"])
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); F["enc_date"] = pd.to_datetime(F.enc_date); P = pd.read_parquet(D/"test_predictions.parquet")[["decision_id","M2_structured_history","M5_full"]]
d = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); d = d[d.eligible == 1].reset_index(drop=True)
E0 = np.array([], dtype="datetime64[ns]"); fl = lambda v: np.sort(v.astype("datetime64[D]").astype("datetime64[ns]"))
AT = {p: fl(g.t.values) for p, g in en[en.occurred == "NO"].groupby("person_id")}; CT = {p: fl(g.t.values) for p, g in en[en.occurred == "YES"].groupby("person_id")}
def cnt(arr, a, b): return int(np.searchsorted(arr, b, "right") - np.searchsorted(arr, a, "right"))
t = d.enc_date.values.astype("datetime64[ns]"); pid = d.person_id.values
d["c17"] = [cnt(CT.get(p, E0), x, x + 7*D1) for p, x in zip(pid, t)]; d["a17"] = [cnt(AT.get(p, E0), x, x + 7*D1) for p, x in zip(pid, t)]
d["ret_8_30"] = [int(cnt(CT.get(p, E0), x + 7*D1, x + 30*D1) > 0) for p, x in zip(pid, t)]; d["att30"] = [cnt(AT.get(p, E0), x - 31*D1, x - D1) for p, x in zip(pid, t)]
def prior_connect(p, x):
    a = AT.get(p, E0); c = CT.get(p, E0); a = a[a < x]
    if len(a) == 0: return np.nan
    return float(np.mean([cnt(c, ai, ai + 2*D1) > 0 for ai in a[-10:]]))
d["prior_connect"] = [prior_connect(p, x) for p, x in zip(pid, t)]
d["lapsed_unattended"] = ((d.c17 == 0) & (d.a17 == 0)).astype(int); d["lapsed"] = (d.c17 == 0).astype(int); d["complete_30"] = (t + 30*D1 <= np.datetime64("2026-06-07"))
# ---- components (a)+(b) on the test set vs 90-day disengagement
te = d[(d.training_era == 0)].merge(P, on="decision_id"); thr = float(np.quantile(te.M2_structured_history, 0.8)); te["risk_flag"] = (te.M2_structured_history >= thr).astype(int)
def conf(flag, y):
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum()); fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "flagged": tp + fp, "flag_rate": round((tp + fp)/len(y), 4), "sensitivity": round(tp/max(tp+fn, 1), 4), "specificity": round(tn/max(tn+fp, 1), 4), "ppv": round(tp/max(tp+fp, 1), 4), "npv": round(tn/max(tn+fn, 1), 4)}
SS = json.load(open(R/"simple_score.json"))
out = {"design": "amendment v3.6", "n_test": int(len(te)), "test_events": int(te.y_primary.sum()),
       "rule_a_risk_top20": conf(te.risk_flag.values, te.y_primary.values), "rule_b_lapsed_unattended_day7": conf(te.lapsed_unattended.values, te.y_primary.values), "rule_b_lapsed_any_day7": conf(te.lapsed.values, te.y_primary.values),
       "rule_ab_risk_and_lapsed_unattended": conf((te.risk_flag & te.lapsed_unattended).values, te.y_primary.values), "rule_a_or_b": conf((te.risk_flag | te.lapsed_unattended).values, te.y_primary.values),
       "points_score_20pct": {"sensitivity": SS["points_score"]["at_20pct_flagged"]["sensitivity"], "ppv": SS["points_score"]["at_20pct_flagged"]["per_100_flags"]/100, "flag_rate": SS["points_score"]["at_20pct_flagged"]["flag_rate"], "auroc": SS["points_score"]["auroc"]}}
# ---- component (c): uplift of attempting among lapsed episodes (all eras), T-learner, split-sample rank test
L = d[(d.c17 == 0) & (d.complete_30)].copy(); L["A"] = (L.a17 > 0).astype(int); L["Y"] = L.ret_8_30
L["state_va"] = (L.state == "VIRGINIA").astype(int); L["state_wa"] = (L.state == "WASHINGTON").astype(int); L["inp"] = L.contact_type.isin(["HOME_VISIT","OTHER_INPERSON","HOSPITAL","CBO","IN_COMMUNITY","PROVIDER_OFFICE"]).astype(int); L["prior_connect_f"] = L.prior_connect.fillna(-1)
X = L[["age","risk_percentile","days_from_zd","n_prior_90","att30","prior_connect_f","state_va","state_wa","inp","any_bh","adt_all_prior_365d"]].astype(float).fillna(0).values; A = L.A.values; Y = L.Y.values.astype(float); n = len(L); half = rng.permutation(n) % 2
def fit_uplift(idx):
    m1 = GBR(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=1).fit(X[idx][A[idx] == 1], Y[idx][A[idx] == 1]); m0 = GBR(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=1).fit(X[idx][A[idx] == 0], Y[idx][A[idx] == 0]); return m1, m0
def dr(idx):
    e = np.clip(GB(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(X[idx], A[idx]).predict_proba(X[idx])[:,1], 0.02, 0.98); m1, m0 = fit_uplift(idx); p1 = m1.predict(X[idx]); p0 = m0.predict(X[idx])
    return p1 - p0 + A[idx]*(Y[idx] - p1)/e - (1 - A[idx])*(Y[idx] - p0)/(1 - e)
iA, iB = np.where(half == 0)[0], np.where(half == 1)[0]; m1, m0 = fit_uplift(iA); tau = m1.predict(X[iB]) - m0.predict(X[iB]); psi = dr(iB); top = tau >= np.median(tau)
diff = psi[top].mean() - psi[~top].mean(); bs = []
for _ in range(500):
    s_ = rng.integers(0, len(iB), len(iB)); tb = tau[s_] >= np.median(tau[s_]); bs.append(psi[s_][tb].mean() - psi[s_][~tb].mean())
LB = L.iloc[iB].assign(tau=tau, psi=psi, tert=pd.qcut(tau, 3, labels=["low","middle","high"]))
tert = {}
for k, g in LB.groupby("tert"):
    a1, a0 = g[g.A == 1].Y.mean(), g[g.A == 0].Y.mean(); eff = float(g.psi.mean())
    tert[str(k)] = {"n": int(len(g)), "attempted_share": round(float(g.A.mean()), 3), "return_if_attempted": round(float(a1), 3), "return_if_not": round(float(a0), 3), "dr_effect": round(eff, 4), "attempts_per_additional_return": round(1/eff, 1) if eff > 0.005 else None, "mean_predicted_uplift": round(float(g.tau.mean()), 4)}
out["rule_c_uplift"] = {"n_lapsed_episodes": int(n), "attempted_share": round(float(A.mean()), 3), "return_rate": round(float(Y.mean()), 3), "dr_ate_half_B": round(float(psi.mean()), 4), "effect_above_median_uplift": round(float(psi[top].mean()), 4), "effect_below_median_uplift": round(float(psi[~top].mean()), 4), "difference": round(float(diff), 4), "ci_95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "gate_pass": bool(np.percentile(bs, 2.5) > 0), "by_tertile": tert}
# equity of screening out the below-median-uplift patients
LB["screened_out"] = (~top).astype(int); eq = {}
for dim in ["race", "state"]:
    eq[dim] = {str(k): {"n": int(len(g)), "screened_out_share": round(float(g.screened_out.mean()), 3)} for k, g in LB.groupby(dim) if len(g) >= 100}
out["rule_c_equity_screened_out"] = eq
# combined rule on test set: (a) & (b) & (c above median) among test lapsed episodes
teL = te[(te.c17 == 0)].copy(); teL["prior_connect_f"] = teL.prior_connect.fillna(-1); teL["state_va"] = (teL.state == "VIRGINIA").astype(int); teL["state_wa"] = (teL.state == "WASHINGTON").astype(int); teL["inp"] = teL.contact_type.isin(["HOME_VISIT","OTHER_INPERSON","HOSPITAL","CBO","IN_COMMUNITY","PROVIDER_OFFICE"]).astype(int)
Xt = teL[["age","risk_percentile","days_from_zd","n_prior_90","att30","prior_connect_f","state_va","state_wa","inp","any_bh","adt_all_prior_365d"]].astype(float).fillna(0).values
m1f, m0f = fit_uplift(np.arange(n)); teL["tau"] = m1f.predict(Xt) - m0f.predict(Xt); med = float(np.median(tau)); teL["c_flag"] = (teL.tau >= med).astype(int)
te2 = te.merge(teL[["decision_id","c_flag"]], on="decision_id", how="left"); te2["c_flag"] = te2.c_flag.fillna(0).astype(int)
out["rule_abc_on_test"] = conf((te2.risk_flag & te2.lapsed_unattended & te2.c_flag).values, te2.y_primary.values)
out["rule_abc_among_lapsed_unattended_high_risk"] = {"n": int((te2.risk_flag & te2.lapsed_unattended).sum()), "kept_by_c": int((te2.risk_flag & te2.lapsed_unattended & te2.c_flag).sum())}
json.dump(out, open(R/"ops_rule.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str)[:4000])
