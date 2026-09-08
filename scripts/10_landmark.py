"""Step 10 (amendment v3.1 Section A): pseudo-prospective, frozen-model, rolling-landmark validation.
At each month-end landmark L, train the full model without embeddings (M3b: structured + history + lexicon + tags + TF-IDF note stack)
on eligible decision points dated <= L - 90 days (outcome observable at L), freeze the 80th-percentile threshold and decile cutpoints
from the training predictions, and score every eligible decision point dated > L. Writes results/landmark.json."""
import json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import sys; sys.path.insert(0, 'scripts'); from _boot import boot_ci, std_fns
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); HP = json.load(open(R/"hyperparameters.json"))["gbm"]
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
enc = pd.read_parquet(D/"encounters.parquet")[["person_id","enc_date","note_text"]]; enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])  # same ordering as 04 and 20
E = {p: (g.enc_date.values, g.note_text.fillna("").values) for p, g in enc.groupby("person_id")}
prior = []
for r in df.itertuples():
    d, n = E.get(r.person_id, (np.array([], dtype="datetime64[ns]"), np.array([]))); t = r.enc_date.to_datetime64(); m = (d < t) & (d >= t - np.timedelta64(90, "D")); prior.append(" ".join(n[m])[-8000:])
df["prior_text"] = prior; df["cur_text"] = df.note_text.fillna("")
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
X = pd.concat([df[STRUCT + HIST + LEX + TAGS].astype(float), pd.get_dummies(df[["gender","race","cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)], axis=1).fillna(0).values
y = df.y_primary.values
def stack(texts, tr, te):
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te]); oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], df.person_id.values[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    return oof, LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]).predict_proba(Xte)[:,1]
def calib(yy, p):
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1,1), yy); return round(float(lr.coef_[0][0]), 3), round(float(lr.intercept_[0]), 3)
def metrics(yy, p, thr):
    if len(yy) < 100 or len(np.unique(yy)) < 2: return {"n": int(len(yy)), "note": "insufficient"}
    flag = p >= thr; sl, ic = calib(yy, p)
    return {"n": int(len(yy)), "events": int(yy.sum()), "event_rate": round(float(yy.mean()), 4), "auroc": round(float(roc_auc_score(yy, p)), 4), "auprc": round(float(average_precision_score(yy, p)), 4), "calibration_slope": sl, "calibration_intercept": ic,
            "flag_rate": round(float(flag.mean()), 4), "per_100_flags": round(float(100*yy[flag].mean()), 1) if flag.sum() > 0 else None, "sensitivity_at_threshold": round(float(yy[flag].sum()/max(yy.sum(),1)), 4)}
PRED = {}
out = {"rule": "training = eligible decision points dated <= L - 90 days; scoring = dated > L; threshold = 80th percentile of training predictions; deciles from training predictions", "landmarks": {}}
for L in pd.date_range("2025-01-31", "2025-12-31", freq="ME"):
    tr = (df.enc_date <= L - pd.Timedelta(days=90)).values; te = (df.enc_date > L).values
    if tr.sum() < 1000 or te.sum() < 500: continue
    oc, pc = stack(df.cur_text.values, tr, te); op, pp = stack(df.prior_text.values, tr, te)
    m = HistGradientBoostingClassifier(**HP).fit(np.column_stack([X[tr], oc, op]), y[tr]); ptr = m.predict_proba(np.column_stack([X[tr], oc, op]))[:,1]; pte = m.predict_proba(np.column_stack([X[te], pc, pp]))[:,1]
    thr = float(np.quantile(ptr, 0.8)); cuts = np.quantile(ptr, np.linspace(0, 1, 11)); cuts[0], cuts[-1] = -np.inf, np.inf
    dec = np.clip(np.digitize(pte, cuts[1:-1]), 0, 9); yte = y[te]
    seen = np.isin(df.person_id.values[te], np.unique(df.person_id.values[tr]))
    months = ((df.enc_date[te] - L).dt.days // 30 + 1).values
    rec = {"landmark": str(L.date()), "train": {"n": int(tr.sum()), "events": int(y[tr].sum())}, "threshold": round(thr, 4), "overall": metrics(yte, pte, thr), "overall_ci": boot_ci(yte, pte, df.person_id.values[te], std_fns(thr), B=300),
           "by_training_decile": {str(k+1): {"n": int((dec==k).sum()), "observed_rate": round(float(yte[dec==k].mean()), 4) if (dec==k).sum() else None} for k in range(10)},
           "by_month_since_landmark": {str(mo): metrics(yte[months==mo], pte[months==mo], thr) for mo in sorted(set(months)) if mo <= 9},
           "members_seen_in_training": metrics(yte[seen], pte[seen], thr), "members_not_seen_in_training": metrics(yte[~seen], pte[~seen], thr)}
    PRED[str(L.date())] = (yte, pte, thr, df.person_id.values[te]); out["landmarks"][str(L.date())] = rec; print(str(L.date()), rec["train"], rec["overall"], flush=True)
elig = [k for k, v in out["landmarks"].items() if k <= "2025-11-30" and v["train"]["events"] >= 2000]
out["headline_landmark"] = max(elig) if elig else None; out["headline_rule"] = "latest month-end landmark on or before 2025-11-30 with at least 2,000 training disengagement events"
if out["headline_landmark"]:
    yte, pte, thr, ids = PRED[out["headline_landmark"]]; rng = np.random.default_rng(20260908); up = np.unique(ids); idx = {q: np.where(ids == q)[0] for q in up}; bs = {}
    for _ in range(500):
        ix = np.concatenate([idx[q] for q in rng.choice(up, len(up))]); r = metrics(yte[ix], pte[ix], thr)
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("n", "events"): bs.setdefault(k, []).append(v)
    out["headline_ci_95"] = {k: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)] for k, v in bs.items()}
json.dump(out, open(R/"landmark.json", "w"), indent=1); print("headline", out["headline_landmark"], out.get("headline_ci_95"))
