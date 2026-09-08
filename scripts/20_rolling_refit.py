"""Step 20: deployment emulation with quarterly refitting versus a single frozen model. At each quarter-end Q (2025-03-31 to 2026-03-31) the full model
without embeddings is trained on eligible decision points dated <= Q - 90 days and scores decision points in (Q, Q + 3 months]; the concatenated series is
the 'refit quarterly' deployment. The comparison scores the same decision points with the model frozen at 2025-03-31. Member-bootstrap intervals (500)
for pooled metrics; also bootstrap intervals for the headline frozen landmark (2025-11-30). Writes results/rolling_refit.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
enc = pd.read_parquet(D/"encounters.parquet")[["person_id","enc_date","note_text"]]; enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])
E = {p: (g.enc_date.values, g.note_text.fillna("").values) for p, g in enc.groupby("person_id")}; prior = []
for r in df.itertuples():
    d, n = E.get(r.person_id, (np.array([], dtype="datetime64[ns]"), np.array([]))); t = r.enc_date.to_datetime64(); m = (d < t) & (d >= t - np.timedelta64(90, "D")); prior.append(" ".join(n[m])[-8000:])
df["prior_text"] = prior; df["cur_text"] = df.note_text.fillna("")
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
X = pd.concat([df[STRUCT + HIST + LEX + TAGS].astype(float), pd.get_dummies(df[["gender","race","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0).values; y = df.y_primary.values
def stack(texts, tr, te):
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te]); oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], df.person_id.values[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    return oof, LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]).predict_proba(Xte)[:,1]
def fit_score(L, te):
    tr = (df.enc_date <= L - pd.Timedelta(days=90)).values; oc, pc = stack(df.cur_text.values, tr, te); op, pp = stack(df.prior_text.values, tr, te)
    m = HistGradientBoostingClassifier(**HP).fit(np.column_stack([X[tr], oc, op]), y[tr]); ptr = m.predict_proba(np.column_stack([X[tr], oc, op]))[:,1]; return m.predict_proba(np.column_stack([X[te], pc, pp]))[:,1], float(np.quantile(ptr, 0.8)), int(tr.sum())
def calib(yy, p):
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=500).fit(lp.reshape(-1,1), yy); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def metrics(yy, p, flag):
    sl, ic = calib(yy, p); return {"n": int(len(yy)), "events": int(yy.sum()), "auroc": roc_auc_score(yy, p), "auprc": average_precision_score(yy, p), "calibration_slope": sl, "calibration_intercept": ic, "flag_rate": flag.mean(), "per_100_flags": 100*yy[flag].mean() if flag.sum() else np.nan, "sensitivity": yy[flag].sum()/max(yy.sum(),1)}
def with_ci(yy, p, flag, ids, B=500, seed=20260908):
    pt = metrics(yy, p, flag); rng = np.random.default_rng(seed); up = np.unique(ids); idx = {q: np.where(ids == q)[0] for q in up}; bs = {k: [] for k in pt}
    for _ in range(B):
        ix = np.concatenate([idx[q] for q in rng.choice(up, len(up))]); r = metrics(yy[ix], p[ix], flag[ix]); [bs[k].append(v) for k, v in r.items()]
    return {k: {"estimate": round(float(v), 4), "ci_95": [round(float(np.nanpercentile(bs[k], 2.5)), 4), round(float(np.nanpercentile(bs[k], 97.5)), 4)]} if k not in ("n","events") else int(v) for k, v in pt.items()}
Q = pd.to_datetime(["2025-03-31","2025-06-30","2025-09-30","2025-12-31","2026-03-31"]); END = df.enc_date.max()
pool = (df.enc_date > Q[0]).values; p_refit = np.full(len(df), np.nan); f_refit = np.zeros(len(df), dtype=bool); byq = {}
for i, q in enumerate(Q):
    hi = Q[i+1] if i+1 < len(Q) else END + pd.Timedelta(days=1); te = ((df.enc_date > q) & (df.enc_date <= hi)).values
    if te.sum() == 0: continue
    p, thr, ntr = fit_score(q, te); p_refit[te] = p; f_refit[te] = p >= thr; byq[str(q.date())] = {"train_n": ntr, **with_ci(y[te], p, p >= thr, df.person_id.values[te], B=300)}; print("refit", q.date(), byq[str(q.date())]["auroc"], flush=True)
p_frozen, thr_f, ntr_f = fit_score(Q[0], pool); f_frozen = p_frozen >= thr_f
out = {"design": "quarterly refit: model retrained at each quarter-end on decision points dated at least 90 days earlier and applied to the following quarter; frozen: model trained at 2025-03-31 applied to every later decision point; both evaluated on the same pooled decision points after 2025-03-31", "pooled_n": int(pool.sum()), "refit_quarterly": with_ci(y[pool], p_refit[pool], f_refit[pool], df.person_id.values[pool]), "frozen_2025_03_31": with_ci(y[pool], p_frozen, f_frozen, df.person_id.values[pool]), "refit_by_quarter": byq}
# frozen headline landmark with intervals
L = pd.Timestamp("2025-11-30"); te = (df.enc_date > L).values; p, thr, ntr = fit_score(L, te); out["frozen_2025_11_30_with_ci"] = {"train_n": ntr, **with_ci(y[te], p, p >= thr, df.person_id.values[te])}
# frozen by quarter (same quarters as refit) for a side-by-side
out["frozen_by_quarter"] = {}
for i, q in enumerate(Q):
    hi = Q[i+1] if i+1 < len(Q) else END + pd.Timedelta(days=1); te_q = ((df.enc_date > q) & (df.enc_date <= hi)).values[pool]
    if te_q.sum() == 0: continue
    out["frozen_by_quarter"][str(q.date())] = with_ci(y[pool][te_q], p_frozen[te_q], f_frozen[te_q], df.person_id.values[pool][te_q], B=300)
json.dump(out, open(R/"rolling_refit.json", "w"), indent=1, default=str); print(json.dumps({k: out[k] for k in ["refit_quarterly","frozen_2025_03_31","frozen_2025_11_30_with_ci"]}, indent=1, default=str))
