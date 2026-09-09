"""Step 22: 95% member-bootstrap intervals for every row of Table 3 (all test-era subgroups and the leave-one-state-out
validation) and for the secondary outcomes (Supplementary Table S18). Subgroup intervals read data_cache/test_predictions.parquet;
leave-one-state-out and secondary-outcome models are refit here with the frozen hyperparameters and the same feature
construction as scripts 04 and 05, and their held-out predictions are bootstrapped by member (500 resamples).
Writes results/table_cis.json."""
import re, json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]; SEED = 20260907; B = 500
rng = np.random.default_rng(20260908)
def gbm(): return HistGradientBoostingClassifier(**HP)
def nb20(y, p):
    thr = np.quantile(p, 0.8); f = p >= thr; n = len(y); tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); return float(tp/n - fp/n*(thr/(1-thr))), float(tp/max(f.sum(), 1)), float(tp/max((y == 1).sum(), 1))
def boot(y, p, ids, fns):
    up = np.unique(ids); ix = {q: np.where(ids == q)[0] for q in up}; pt = {k: f(y, p) for k, f in fns.items()}; bs = {k: [] for k in fns}
    for _ in range(B):
        s = np.concatenate([ix[q] for q in rng.choice(up, len(up))])
        if y[s].min() == y[s].max(): continue
        for k, f in fns.items(): bs[k].append(f(y[s], p[s]))
    return {k: {"estimate": round(float(v), 4), "ci_95": [round(float(np.percentile(bs[k], 2.5)), 4), round(float(np.percentile(bs[k], 97.5)), 4)]} for k, v in pt.items()}
DISC = {"auroc": roc_auc_score, "auprc": average_precision_score}
FULL = {"auroc": roc_auc_score, "auprc": average_precision_score, "net_benefit_20": lambda y, p: nb20(y, p)[0], "ppv_20": lambda y, p: nb20(y, p)[1], "sensitivity_20": lambda y, p: nb20(y, p)[2]}
out = {"bootstrap": f"{B} resamples of members within each row; percentile intervals", "subgroups": {}, "leave_one_state_out": {}, "secondary_outcomes": {}}
# ---- Table 3 subgroups from stored test-era predictions
P = pd.read_parquet(D/"test_predictions.parquet"); yP = P.y.values.astype(int)
groups = [("by_state", "State", [(s, (P.state == s).values) for s in ["VIRGINIA","WASHINGTON","OHIO"]]),
          ("by_race", "Race and ethnicity", [(r, (P.race == r).values) for r in ["Black or African American","White","Hispanic","Asian"]]),
          ("by_days_since_activation", "Days since activation", [(lab, ((P.days_from_zd > lo) & (P.days_from_zd <= hi)).values) for lab, lo, hi in [("0_30",-1,30),("31_90",30,90),("91_180",90,180),("181_365",180,366)]]),
          ("by_activating_discipline", "Discipline of contact", [(g, (P.cur_rg == g).values) for g in ["CHW","CC","CPhT","PharmD","Therapist"]])]
for key, nice, levels in groups:
    out["subgroups"][key] = {}
    for lev, mk in levels:
        if mk.sum() < 200 or len(np.unique(yP[mk])) < 2: continue
        rec = {"n": int(mk.sum()), "events": int(yP[mk].sum())}
        for m in ["M2_structured_history", "M5_full"]: rec[m] = boot(yP[mk], P[m].values[mk], P.person_id.values[mk], DISC)
        rec["auroc_gain"] = boot(yP[mk], np.column_stack([P.M5_full.values[mk], P.M2_structured_history.values[mk]]), P.person_id.values[mk], {"gain": lambda y, q: roc_auc_score(y, q[:,0]) - roc_auc_score(y, q[:,1])})["gain"]
        out["subgroups"][key][lev] = rec; print(key, lev, rec["M5_full"]["auroc"], flush=True)

# ---- equalized odds at the single overall 20% threshold (true and false positive rates by group; ratio of smallest to largest group value)
out["equalized_odds"] = {}
for m in ["M2_structured_history", "M5_full"]:
    thr = float(np.quantile(P[m].values, 0.8)); fl = (P[m].values >= thr)
    for dim, levels in [("race", ["Black or African American","White","Hispanic","Asian"]), ("state", ["VIRGINIA","WASHINGTON","OHIO"])]:
        def rates(mask, f=fl, y=yP):
            return {lev: (float(f[mask[lev] & (y == 1)].mean()), float(f[mask[lev] & (y == 0)].mean())) for lev in levels}
        masks = {lev: (P[dim].values == lev) for lev in levels}; pt = rates(masks)
        eo = lambda r: min(min(v[0] for v in r.values())/max(v[0] for v in r.values()), min(v[1] for v in r.values())/max(v[1] for v in r.values()))
        up = np.unique(P.person_id.values); ix = {q: np.where(P.person_id.values == q)[0] for q in up}; bs = []; bt = {lev: [] for lev in levels}; bf = {lev: [] for lev in levels}
        for _ in range(B):
            s_ = np.concatenate([ix[q] for q in rng.choice(up, len(up))]); mk = {lev: (P[dim].values[s_] == lev) for lev in levels}; r = rates(mk, fl[s_], yP[s_])
            if any(np.isnan(v[0]) or np.isnan(v[1]) for v in r.values()): continue
            bs.append(eo(r)); [bt[lev].append(r[lev][0]) for lev in levels]; [bf[lev].append(r[lev][1]) for lev in levels]
        ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
        out["equalized_odds"][f"{m}:{dim}"] = {"threshold": round(thr, 4), "groups": {lev: {"n": int(masks[lev].sum()), "tpr": round(pt[lev][0], 3), "tpr_ci_95": ci(bt[lev]), "fpr": round(pt[lev][1], 3), "fpr_ci_95": ci(bf[lev])} for lev in levels}, "equalized_odds_ratio": round(eo(pt), 3), "equalized_odds_ratio_ci_95": ci(bs)}
        print("EO", m, dim, out["equalized_odds"][f"{m}:{dim}"]["equalized_odds_ratio"], flush=True)
# ---- shared feature construction (scripts 04 and 05)
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
df = F.merge(O[["decision_id","eligible","y_primary","y_explicit","y_silent","y_30"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + [c for c in ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"] if c in df.columns]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
# ---- leave-one-state-out (script 05 construction)
X = pd.concat([df[STRUCT + HIST + LEX + TAGS].astype(float), pd.get_dummies(df[["gender","race","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0)
Xh = X[[c for c in X.columns if not (c.startswith("lex_") or c.startswith("tag_"))]]; y = df.y_primary.values; txt = df.note_text.fillna("").values
for st in ["VIRGINIA","WASHINGTON","OHIO"]:
    tr = (df.state != st).values; te = ~tr
    p2 = gbm().fit(Xh.values[tr], y[tr]).predict_proba(Xh.values[te])[:,1]
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(txt[tr]); Xte = vec.transform(txt[te]); oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], df.person_id[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    pte = LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]).predict_proba(Xte)[:,1]
    pf = gbm().fit(np.column_stack([X.values[tr], oof]), y[tr]).predict_proba(np.column_stack([X.values[te], pte]))[:,1]
    ids = df.person_id.values[te]
    out["leave_one_state_out"][f"test_{st}"] = {"n": int(te.sum()), "events": int(y[te].sum()), "event_rate": round(float(y[te].mean()), 4), "structured_history": boot(y[te], p2, ids, DISC), "full_text": boot(y[te], pf, ids, DISC)}
    print("LOSO", st, out["leave_one_state_out"][f"test_{st}"]["full_text"]["auroc"], flush=True)
# ---- secondary outcomes (script 04 construction, temporal split)
tr = (df.training_era == 1).values; te = ~tr
cat_struct = pd.get_dummies(df[["gender","race","state"]].fillna("u"), drop_first=True, dtype=float); cat_hist = pd.get_dummies(df[["cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)
X_struct = pd.concat([df[STRUCT].astype(float), cat_struct], axis=1).fillna(0); X_hist = pd.concat([X_struct, df[HIST].astype(float), cat_hist], axis=1).fillna(0)
X_lt = pd.concat([X_hist, df[LEX+TAGS].astype(float)], axis=1).fillna(0).values
enc = pd.read_parquet(D/"encounters.parquet")[["person_id","enc_date","note_text"]]; enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])
E = {p: (g.enc_date.values, g.note_text.fillna("").values) for p, g in enc.groupby("person_id")}; prior_txt = []
for r in df.itertuples():
    d, t_ = E.get(r.person_id, (np.array([], dtype="datetime64[ns]"), np.array([]))); t = r.enc_date.to_datetime64(); m = (d < t) & (d >= t - np.timedelta64(90, "D")); prior_txt.append(" ".join(t_[m])[-8000:])
df["prior_text"] = prior_txt; df["cur_text"] = df.note_text.fillna("")
def text_stack(texts, yy):
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te]); oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, yy[tr], df.person_id[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], yy[tr][a]).predict_proba(Xt[b])[:,1]
    return oof, LogisticRegression(C=0.5, max_iter=3000).fit(Xt, yy[tr]).predict_proba(Xte)[:,1]
emb = pd.read_parquet(D/"note_emb_bge.parquet"); ecols = [c for c in emb.columns if re.fullmatch(r"e\d{3}", c)]
pca = PCA(64, random_state=SEED).fit(emb.loc[emb.encounter_id.isin(df.decision_id[tr]), ecols].values); Z = pca.transform(emb[ecols].values); zc = [f"z{i:02d}" for i in range(64)]
emb = pd.concat([emb[["encounter_id","person_id","enc_date"]], pd.DataFrame(Z, columns=zc)], axis=1); emb["enc_date"] = pd.to_datetime(emb.enc_date)
cur = df[["decision_id"]].merge(emb.drop(columns=["person_id","enc_date"]), left_on="decision_id", right_on="encounter_id", how="left")[zc].fillna(0).values
G = {p: (g.enc_date.values, g[zc].values) for p, g in emb.groupby("person_id")}; prior = np.zeros((len(df), 64), dtype=np.float32)
for i, r in enumerate(df.itertuples()):
    d, z = G.get(r.person_id, (np.array([], dtype="datetime64[ns]"), None)); t = r.enc_date.to_datetime64()
    if z is not None:
        m = (d < t) & (d >= t - np.timedelta64(90, "D"))
        if m.any(): prior[i] = z[m].mean(0)
Zemb = np.column_stack([cur, prior]); ids = df.person_id.values[te]
for yc, lab in [("y_explicit", "Explicit exit status within 90 days"), ("y_silent", "Silent loss within 90 days"), ("y_30", "No contact within 30 days")]:
    yy = df[yc].values.astype(int); block = {"label": lab, "n": int(te.sum()), "events": int(yy[te].sum()), "test_event_rate": round(float(yy[te].mean()), 4)}
    block["M2_structured_history"] = boot(yy[te], gbm().fit(X_hist.values[tr], yy[tr]).predict_proba(X_hist.values[te])[:,1], ids, FULL)
    o_c, p_c = text_stack(df.cur_text.values, yy); o_p, p_p = text_stack(df.prior_text.values, yy)
    p5 = gbm().fit(np.column_stack([X_lt[tr], Zemb[tr], o_c, o_p]), yy[tr]).predict_proba(np.column_stack([X_lt[te], Zemb[te], p_c, p_p]))[:,1]
    block["M5_full"] = boot(yy[te], p5, ids, FULL); out["secondary_outcomes"][yc] = block; print(yc, block["M5_full"]["auroc"], flush=True)
json.dump(out, open(R/"table_cis.json", "w"), indent=1); print("saved results/table_cis.json")
