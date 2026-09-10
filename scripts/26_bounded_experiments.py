"""Step 26 (amendment v3.5): bounded experiments K1-K5 on outreach effort, attempt timing and channel, and who benefits from an early second contact.
Reads data_cache/encounter_times_study.parquet (attempts with time of day), dp_features, dp_outcomes, encounters, levers2 sample. Writes results/bounded_experiments.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
import statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0, "scripts"); from _boot import boot_ci
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); D1 = np.timedelta64(1, "D"); H1 = np.timedelta64(1, "h"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]; rng = np.random.default_rng(20260910)
en = pd.read_parquet(D/"encounter_times_study.parquet"); en["t"] = pd.to_datetime(en.startTime.fillna(en.dateOfEncounter)); en = en.dropna(subset=["t"])
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); F["enc_date"] = pd.to_datetime(F.enc_date)
S = pd.read_parquet(D/"state_features.parquet")[["encounter_id","zd"]].rename(columns={"encounter_id":"decision_id"}); S["zd"] = pd.to_datetime(S.zd)
d = F.merge(O[["decision_id","eligible","y_primary","lever_eligible","y_lever"]], on="decision_id").merge(S, on="decision_id"); d = d[d.eligible == 1].reset_index(drop=True)
TZ = {"VIRGINIA": "America/New_York", "OHIO": "America/New_York", "WASHINGTON": "America/Los_Angeles"}
att = en[en.occurred == "NO"].sort_values("t"); comp = en[en.occurred == "YES"].sort_values("t")
AT = {p: g.t.values for p, g in att.groupby("person_id")}; CT = {p: g.t.values for p, g in comp.groupby("person_id")}
ATd = {p: np.sort(v.astype("datetime64[D]").astype("datetime64[ns]")) for p, v in AT.items()}; CTd = {p: np.sort(v.astype("datetime64[D]").astype("datetime64[ns]")) for p, v in CT.items()}   # day-floored: windows (t, t+k] exclude the index day
E0 = np.array([], dtype="datetime64[ns]")
def cnt(arr, a, b): return int(np.searchsorted(arr, b, "right") - np.searchsorted(arr, a, "right"))   # count in (a, b]
out = {"source": "attempts = EncounterNote.encounterOccurred = 'NO'; times UTC converted to state time zone"}
# ---------------- K1: attempt features added to M2 and M5
t = d.enc_date.values.astype("datetime64[ns]"); pid = d.person_id.values
a30 = np.array([cnt(ATd.get(p, E0), tt - 31*D1, tt - D1) for p, tt in zip(pid, t)]); a90 = np.array([cnt(ATd.get(p, E0), tt - 91*D1, tt - D1) for p, tt in zip(pid, t)])
def since_last(p, tt):
    c = CTd.get(p, E0); c = c[c < tt]; last = c[-1] if len(c) else tt - 365*D1; a = ATd.get(p, E0); return int(((a > last) & (a < tt)).sum())
def days_last_att(p, tt):
    a = ATd.get(p, E0); a = a[a < tt]; return float((tt - a[-1]) / D1) if len(a) else 365.0
asl = np.array([since_last(p, tt) for p, tt in zip(pid, t)]); dla = np.array([days_last_att(p, tt) for p, tt in zip(pid, t)])
d["att30"], d["att90"], d["att_since_last"], d["days_last_att"] = a30, a90, asl, dla
P = pd.read_parquet(D/"test_predictions.parquet")[["decision_id","M2_structured_history","M5_full"]]
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
ATTF = ["att30","att90","att_since_last","days_last_att"]
X = pd.concat([d[STRUCT + HIST + ATTF].astype(float), pd.get_dummies(d[["gender","race","state","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0)
tr = (d.training_era == 1).values; te = ~tr; y = d.y_primary.values
p2a = HistGradientBoostingClassifier(**HP).fit(X.values[tr], y[tr]).predict_proba(X.values[te])[:,1]
dte = d[te].merge(P, on="decision_id", how="left"); p2 = dte.M2_structured_history.values; p5 = dte.M5_full.values; ids = dte.person_id.values; yte = y[te]
# M5 + attempts: stack attempt features onto M5 predictions via logistic recalibration is not the frozen design; instead refit GBM on M5 features unavailable here -> report M5 + attempt-augmented M2 stack
up = np.unique(ids); idx = {q: np.where(ids == q)[0] for q in up}; g_auroc, g_auprc, g5 = [], [], []
lp = lambda v: np.log(np.clip(v, 1e-6, 1-1e-6) / (1 - np.clip(v, 1e-6, 1-1e-6)))
from sklearn.linear_model import LogisticRegression
stack = LogisticRegression(C=1e6).fit(np.column_stack([lp(p5), lp(p2a)]), yte); p5a = stack.predict_proba(np.column_stack([lp(p5), lp(p2a)]))[:,1]   # in-sample stack, upper bound
for _ in range(500):
    s_ = np.concatenate([idx[q] for q in rng.choice(up, len(up))]); yy = yte[s_]
    if yy.min() == yy.max(): continue
    g_auroc.append(roc_auc_score(yy, p2a[s_]) - roc_auc_score(yy, p2[s_])); g_auprc.append(average_precision_score(yy, p2a[s_]) - average_precision_score(yy, p2[s_])); g5.append(roc_auc_score(yy, p5a[s_]) - roc_auc_score(yy, p5[s_]))
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out["K1_attempt_features"] = {"n_test": int(te.sum()), "M2_auroc": round(float(roc_auc_score(yte, p2)), 4), "M2_plus_attempts_auroc": round(float(roc_auc_score(yte, p2a)), 4), "auroc_gain": round(float(roc_auc_score(yte, p2a) - roc_auc_score(yte, p2)), 4), "auroc_gain_ci_95": ci(g_auroc),
                              "M2_auprc": round(float(average_precision_score(yte, p2)), 4), "M2_plus_attempts_auprc": round(float(average_precision_score(yte, p2a)), 4), "auprc_gain_ci_95": ci(g_auprc),
                              "M5_auroc": round(float(roc_auc_score(yte, p5)), 4), "M5_stacked_with_attempt_model_auroc_in_sample": round(float(roc_auc_score(yte, p5a)), 4), "M5_stack_gain_ci_95": ci(g5),
                              "gate_pass": bool(ci(g_auroc)[0] > 0), "attempt_feature_rates": {"mean_att30": round(float(a30.mean()), 2), "share_any_att30": round(float((a30 > 0).mean()), 3)}}
print("K1", out["K1_attempt_features"], flush=True)
# ---------------- K2: attempts in days 1-7 after a completed contact with no contact in days 1-7 -> completed contact in days 8-30, within patient
rows = []
for r in d.itertuples():
    tt = r.enc_date.to_datetime64(); c = CTd.get(r.person_id, E0); a = ATd.get(r.person_id, E0)
    if cnt(c, tt, tt + 7*D1) > 0 or tt + 30*D1 > np.datetime64("2026-06-07"): continue
    rows.append(dict(person_id=r.person_id, decision_id=r.decision_id, att17=min(cnt(a, tt, tt + 7*D1), 2), y=int(cnt(c, tt + 7*D1, tt + 30*D1) > 0), pre30=cnt(c, tt - 30*D1, tt), att_prior30=cnt(a, tt - 30*D1, tt), days_from_zd=r.days_from_zd, n_prior_90=r.n_prior_90, mq=f"{r.market}|{pd.Timestamp(tt).to_period('Q')}"))
K = pd.DataFrame(rows); K = K[K.groupby("person_id").person_id.transform("size") >= 2]   # patients with >= 2 episodes
K["a1"] = (K.att17 == 1).astype(int); K["a2"] = (K.att17 == 2).astype(int)
fe = smf.ols("y ~ a1 + a2 + days_from_zd + n_prior_90 + att_prior30 + C(mq) + C(person_id)", data=K).fit(cov_type="cluster", cov_kwds={"groups": K.person_id.astype("category").cat.codes.values})
pt = smf.ols("pre30 ~ a1 + a2 + days_from_zd + n_prior_90 + C(mq) + C(person_id)", data=K).fit(cov_type="cluster", cov_kwds={"groups": K.person_id.astype("category").cat.codes.values})
sdp = lambda k: float(pt.params[k] / K.pre30.std())
# marginal structural: IPTW for 3-level exposure with GBM propensities
from sklearn.ensemble import HistGradientBoostingClassifier as GB
Wm = pd.get_dummies(K[["mq"]], dtype=float); Xw = np.column_stack([K[["days_from_zd","n_prior_90","att_prior30","pre30"]].values, Wm.values])
gbm_ = GB(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(Xw, K.att17.values); pr = gbm_.predict_proba(Xw); cls = {c: i for i, c in enumerate(gbm_.classes_)}; marg = K.att17.value_counts(normalize=True).to_dict()
w = np.clip(np.array([marg[a] for a in K.att17.values]) / pr[np.arange(len(K)), [cls[a] for a in K.att17.values]], 0.1, 10)
ms = sm.WLS(K.y.values, sm.add_constant(K[["a1","a2"]].values, has_constant="add"), weights=w).fit(cov_type="cluster", cov_kwds={"groups": K.person_id.astype("category").cat.codes.values})
print("K2 exposure counts", K.att17.value_counts().to_dict(), flush=True)
out["K2_attempts_after_contact"] = {"n_episodes": int(len(K)), "n_patients": int(K.person_id.nunique()), "exposure_distribution": {str(k): int(v) for k, v in K.att17.value_counts().sort_index().items()}, "outcome_rate": round(float(K.y.mean()), 4),
   "within_patient_rd_1_attempt": round(float(fe.params["a1"]), 4), "ci_1": [round(float(v), 4) for v in fe.conf_int().loc["a1"]], "within_patient_rd_2plus": round(float(fe.params["a2"]), 4), "ci_2plus": [round(float(v), 4) for v in fe.conf_int().loc["a2"]],
   "msm_rd_1": round(float(ms.params[1]), 4), "msm_ci_1": [round(float(v), 4) for v in ms.conf_int()[1]], "msm_rd_2plus": round(float(ms.params[2]), 4), "msm_ci_2plus": [round(float(v), 4) for v in ms.conf_int()[2]],
   "gate1_pretrend_std_diff": {"1": round(sdp("a1"), 3), "2plus": round(sdp("a2"), 3)}, "gate1_pass": bool(abs(sdp("a1")) <= 0.10 and abs(sdp("a2")) <= 0.10)}
o2 = out["K2_attempts_after_contact"]; o2["gate2_agree"] = bool(np.sign(o2["within_patient_rd_2plus"]) == np.sign(o2["msm_rd_2plus"]) and o2["ci_2plus"][0] <= o2["msm_ci_2plus"][1] and o2["msm_ci_2plus"][0] <= o2["ci_2plus"][1]); o2["pass"] = bool(o2["gate1_pass"] and o2["gate2_agree"] and (o2["ci_2plus"][0] > 0 or o2["ci_2plus"][1] < 0))
print("K2", o2, flush=True)
# ---------------- K3: attempt-level timing and channel -> completed contact within 48h, within patient
pt_state = F.drop_duplicates("person_id")[["person_id","state","market","age"]]
A = att.merge(pt_state, on="person_id", how="inner"); A = A[A.t <= pd.Timestamp("2026-06-05")]
loc = [pd.Timestamp(x).tz_localize("UTC").tz_convert(TZ.get(s, "America/New_York")) for x, s in zip(A.t, A.state)]
A["hour"] = [x.hour for x in loc]; A["wd"] = [x.weekday() for x in loc]; A["q"] = [f"{x.year}-Q{(x.month-1)//3+1}" for x in loc]
A["block"] = pd.cut(A.hour, [-1, 7, 11, 16, 20, 23], labels=["other","morning_8_12","midday_12_5","evening_5_9","other2"]).astype(str).replace({"other2": "other"}); A["weekend"] = (A.wd >= 5).astype(int)
A["channel"] = A.contact_type.map(lambda c: "call" if str(c).startswith("PHONE") else "text" if c in ("SMS_TEXT","EMAIL","EHR_COMMUNICATION") else "visit" if c in ("HOME_VISIT","OTHER_INPERSON","HOSPITAL","CBO","IN_COMMUNITY","PROVIDER_OFFICE") else "other")
A["y48"] = [int(cnt(CT.get(p, E0), tt, tt + 48*H1) > 0) for p, tt in zip(A.person_id, A.t.values.astype("datetime64[ns]"))]
A["pre30"] = [cnt(CT.get(p, E0), tt - 30*D1, tt) for p, tt in zip(A.person_id, A.t.values.astype("datetime64[ns]"))]
A["role"] = A.created_by_id.map(F.drop_duplicates("person_id").set_index("person_id").get("cur_rg", pd.Series(dtype=str))).fillna("NA") if False else "NA"
A = A[A.groupby("person_id").person_id.transform("size") >= 2]
k3 = smf.ols("y48 ~ C(block, Treatment('midday_12_5')) + weekend + C(channel, Treatment('call')) + C(q) + C(person_id)", data=A).fit(cov_type="cluster", cov_kwds={"groups": A.person_id.astype("category").cat.codes.values})
pl = smf.ols("pre30 ~ C(block, Treatment('midday_12_5')) + weekend + C(channel, Treatment('call')) + C(q) + C(person_id)", data=A).fit(cov_type="cluster", cov_kwds={"groups": A.person_id.astype("category").cat.codes.values})
def term(m, k): return {"rd": round(float(m.params[k]), 4), "ci_95": [round(float(v), 4) for v in m.conf_int().loc[k]]}
keys = {"evening_5_9": "C(block, Treatment('midday_12_5'))[T.evening_5_9]", "morning_8_12": "C(block, Treatment('midday_12_5'))[T.morning_8_12]", "other_hours": "C(block, Treatment('midday_12_5'))[T.other]", "weekend": "weekend", "text_vs_call": "C(channel, Treatment('call'))[T.text]", "visit_vs_call": "C(channel, Treatment('call'))[T.visit]"}
out["K3_attempt_timing_channel"] = {"n_attempts": int(len(A)), "n_patients": int(A.person_id.nunique()), "success_48h_rate": round(float(A.y48.mean()), 4), "success_by_block": {k: round(float(v), 4) for k, v in A.groupby("block").y48.mean().items()}, "success_by_channel": {k: round(float(v), 4) for k, v in A.groupby("channel").y48.mean().items()}, "attempt_share_by_block": {k: round(float(v), 3) for k, v in A.block.value_counts(normalize=True).items()},
   "within_patient": {k: term(k3, v) for k, v in keys.items()}, "placebo_pre30_std_diff": {k: round(float(pl.params[v] / A.pre30.std()), 3) for k, v in keys.items()}}
o3 = out["K3_attempt_timing_channel"]; o3["placebo_pass"] = bool(all(abs(v) <= 0.10 for v in o3["placebo_pre30_std_diff"].values())); o3["effects_excluding_zero"] = [k for k, v in o3["within_patient"].items() if v["ci_95"][0] > 0 or v["ci_95"][1] < 0]; o3["pass"] = bool(o3["placebo_pass"] and o3["effects_excluding_zero"])
het = {}
for lab, mask in [("age_under_45", A.age < 45), ("age_45_plus", A.age >= 45)]:
    g = A[mask]; g = g[g.groupby("person_id").person_id.transform("size") >= 2]
    m = smf.ols("y48 ~ C(block, Treatment('midday_12_5')) + weekend + C(channel, Treatment('call')) + C(q) + C(person_id)", data=g).fit(cov_type="cluster", cov_kwds={"groups": g.person_id.astype("category").cat.codes.values})
    het[lab] = {"n": int(len(g)), "evening": term(m, keys["evening_5_9"]), "weekend": term(m, "weekend"), "text_vs_call": term(m, keys["text_vs_call"])}
o3["heterogeneity"] = het; print("K3", {k: o3[k] for k in ["n_attempts","success_48h_rate","success_by_block","within_patient","placebo_pass","effects_excluding_zero"]}, flush=True)
# ---------------- K4: who benefits from an early second contact (L5), DR-learner with split-sample rank test
L5 = d[d.n_prior == 0].copy() if "n_prior" in d.columns else None
if L5 is not None and len(L5) > 500:
    L5 = L5[L5.lever_eligible == 1]; c_all = L5.person_id.map(lambda p: CTd.get(p, E0)); tt = L5.enc_date.values.astype("datetime64[ns]")
    L5["A"] = [int(cnt(c, x, x + 7*D1) > 0) for c, x in zip(c_all, tt)]; L5["Y"] = L5.y_lever.values
    covs = STRUCT + ["gender_f","race_black","race_hisp"] ; L5["gender_f"] = (L5.gender == "Female").astype(int); L5["race_black"] = (L5.race == "Black or African American").astype(int); L5["race_hisp"] = (L5.race == "Hispanic").astype(int)
    Xc = L5[covs].astype(float).fillna(0).values; Aa = L5.A.values; Yy = L5.Y.values.astype(float); n = len(L5); half = rng.permutation(n) % 2
    from sklearn.ensemble import GradientBoostingRegressor
    def dr_scores(idx):
        Xi, Ai, Yi = Xc[idx], Aa[idx], Yy[idx]; e = np.clip(GB(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(Xi, Ai).predict_proba(Xi)[:,1], 0.05, 0.95)
        m1 = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=1).fit(Xi[Ai == 1], Yi[Ai == 1]).predict(Xi); m0 = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=1).fit(Xi[Ai == 0], Yi[Ai == 0]).predict(Xi)
        return m1 - m0 + Ai*(Yi - m1)/e - (1 - Ai)*(Yi - m0)/(1 - e)
    e_all = np.clip(GB(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=1).fit(Xc, Aa).predict_proba(Xc)[:,1], 0.001, 0.999); keep = (e_all >= 0.10) & (e_all <= 0.90)   # equipoise trimming per amendment v3.1 Section D / v3.5 K4
    n_all = int(len(L5)); Xc, Aa, Yy, L5 = Xc[keep], Aa[keep], Yy[keep], L5[keep]; n = len(L5); half = rng.permutation(n) % 2
    idxA, idxB = np.where(half == 0)[0], np.where(half == 1)[0]; psiA = dr_scores(idxA)
    cate = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=1).fit(Xc[idxA], psiA)   # DR-learner second stage
    tauB = cate.predict(Xc[idxB]); psiB = dr_scores(idxB); top = tauB <= np.median(tauB)   # most negative predicted effect = most benefit (outcome is disengagement)
    diff = psiB[top].mean() - psiB[~top].mean(); bs = []
    for _ in range(500):
        s_ = rng.integers(0, len(idxB), len(idxB)); tb = tauB[s_] <= np.median(tauB[s_]); bs.append(psiB[s_][tb].mean() - psiB[s_][~tb].mean())
    prof = L5.iloc[idxB].assign(top=top).groupby("top")[["age","risk_percentile","any_bh","adt_all_prior_365d","gender_f","race_black","race_hisp"]].mean().round(3)
    out["K4_who_benefits_early_second_contact"] = {"n_before_trimming": n_all, "n_equipoise": int(n), "treated_share": round(float(Aa.mean()), 3), "ate_dr_half_B": round(float(psiB.mean()), 4), "effect_top_half_by_predicted_benefit": round(float(psiB[top].mean()), 4), "effect_bottom_half": round(float(psiB[~top].mean()), 4), "difference_top_minus_bottom": round(float(diff), 4), "ci_95": ci(bs), "pass": bool(ci(bs)[1] < 0), "profile_top_vs_bottom": {str(k): v for k, v in prof.T.to_dict().items()}}
    print("K4", {k: v for k, v in out["K4_who_benefits_early_second_contact"].items() if k != "profile_top_vs_bottom"}, flush=True)
# ---------------- K5: joint contacts x attempts in first 30 days -> disengagement in days 31-120 (patient level, from K-style construction on first decision points)
first = d.sort_values("enc_date").groupby("person_id").head(1)
rows = []
for r in first.itertuples():
    z = r.zd.to_datetime64() if pd.notna(r.zd) else r.enc_date.to_datetime64(); c = CTd.get(r.person_id, E0); a = ATd.get(r.person_id, E0)
    if z + 120*D1 > np.datetime64("2026-06-07"): continue
    rows.append(dict(contacts30=min(cnt(c, z - D1, z + 30*D1), 4), attempts30=min(cnt(a, z - D1, z + 30*D1), 4), y=int(cnt(c, z + 30*D1, z + 120*D1) == 0)))
K5 = pd.DataFrame(rows); tab = K5.groupby(["contacts30","attempts30"]).y.agg(["mean","size"])
out["K5_contacts_x_attempts_first30"] = {f"contacts={c}|attempts={a}": {"n": int(v["size"]), "disengagement_31_120": round(float(v["mean"]), 3)} for (c, a), v in tab.iterrows() if v["size"] >= 50}
json.dump(out, open(R/"bounded_experiments.json", "w"), indent=1, default=str); print("saved")
