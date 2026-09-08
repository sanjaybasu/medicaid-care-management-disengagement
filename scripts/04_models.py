"""Step 4: prediction models M0-M5 (PREREGISTRATION_v3 Section 6). Usage: python 04_models.py [--with-embeddings]
Writes results/models_primary.json (and secondary outcomes), results/hyperparameters.json, data_cache/test_predictions.parquet"""
import re, sys, json, pathlib, warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.decomposition import PCA
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); WITH_EMB = "--with-embeddings" in sys.argv
SEED = 20260907; rng = np.random.default_rng(SEED)
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
df = F.merge(O[["decision_id","eligible","y_primary","y_explicit","y_silent","y_30"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True)
df["enc_date"] = pd.to_datetime(df.enc_date); tr = (df.training_era == 1).values; te = ~tr
print(f"eligible decision points {len(df)}; train {tr.sum()} test {te.sum()}; primary rate test {df.y_primary[te].mean():.3f}", flush=True)
# ---------------- feature sets
STRUCT = ["age","risk_percentile","tier1_flg","days_from_zd","days_since_last_adt"] + [f"adt_all_prior_{w}d" for w in [30,90,180,365]] + [c for c in ["diabetes","htn","chf","copd","sud","any_bh","mdd","asthma","polypharmacy","high_ed_ip","no_pcp_last_10mo"] if c in df.columns]
HIST = ["n_prior","n_prior_90","n_prior_30","days_since_last_contact","chw90","pharm90","cpht90","ther90","cc90","inp90","textmod90","n_disc90","gap_mean","gap_max","cur_inp","cur_textmod","note_len","goals_open","goals_completed_before","goals_social","goals_clinical","goals_bh"]
LEX = [c for c in df.columns if c.startswith("lex_")]; TAGS = [c for c in df.columns if c.startswith("tag_")]
cat_struct = pd.get_dummies(df[["gender","race","state"]].fillna("u"), drop_first=True, dtype=float)
cat_hist = pd.get_dummies(df[["cur_rg","prev_rg","contact_type","encounter_type"]].fillna("u"), drop_first=True, dtype=float)
X_struct = pd.concat([df[STRUCT].astype(float), cat_struct], axis=1).fillna(0)
X_hist = pd.concat([X_struct, df[HIST].astype(float), cat_hist], axis=1).fillna(0)
# prior 90-day note text aggregated (strictly before t)
enc = pd.read_parquet(D/"encounters.parquet")[["person_id","enc_date","note_text"]]; enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])
E = {p: (g.enc_date.values, g.note_text.fillna("").values) for p, g in enc.groupby("person_id")}
prior_txt = []
for r in df.itertuples():
    d, t_ = E.get(r.person_id, (np.array([], dtype="datetime64[ns]"), np.array([]))); t = r.enc_date.to_datetime64()
    m = (d < t) & (d >= t - np.timedelta64(90, "D")); prior_txt.append(" ".join(t_[m])[-8000:])
df["prior_text"] = prior_txt; df["cur_text"] = df.note_text.fillna("")
# ---------------- pilot hyperparameter selection (20% of training era, patient-grouped CV), frozen
pilot_pids = rng.choice(df.person_id[tr].unique(), size=int(0.2*df.person_id[tr].nunique()), replace=False); pil = tr & df.person_id.isin(pilot_pids).values
grid = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2, min_samples_leaf=40, max_iter=400, early_stopping=True, validation_fraction=0.15, random_state=SEED) for lr in (0.03, 0.06) for ml in (15, 31) for l2 in (0.5, 2.0)]
best, best_s = None, -1
for gp in grid:
    s = []
    for a, b in GroupKFold(3).split(X_hist[pil], df.y_primary[pil], df.person_id[pil]):
        m = HistGradientBoostingClassifier(**gp).fit(X_hist[pil].values[a], df.y_primary[pil].values[a]); s.append(average_precision_score(df.y_primary[pil].values[b], m.predict_proba(X_hist[pil].values[b])[:,1]))
    if np.mean(s) > best_s: best_s, best = np.mean(s), gp
HP = {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best.items()}; json.dump({"gbm": HP, "pilot_auprc": round(best_s, 4), "pilot_patients": int(len(pilot_pids)), "seed": SEED, "tfidf": {"min_df": 20, "max_df": 0.5, "ngram": [1,2], "max_features": 50000, "C": 0.5}, "pca_dim": 64}, open(R/"hyperparameters.json","w"), indent=1)
print("frozen GBM hyperparameters:", HP, flush=True)
def gbm(): return HistGradientBoostingClassifier(**HP)
# ---------------- text stack (OOF on training era, grouped by patient)
def text_stack(texts, y):
    vec = TfidfVectorizer(min_df=20, max_df=0.5, ngram_range=(1,2), sublinear_tf=True, max_features=50000); Xt = vec.fit_transform(texts[tr]); Xte = vec.transform(texts[te])
    oof = np.zeros(tr.sum())
    for a, b in GroupKFold(5).split(Xt, y[tr], df.person_id[tr]): oof[b] = LogisticRegression(C=0.5, max_iter=3000).fit(Xt[a], y[tr][a]).predict_proba(Xt[b])[:,1]
    lr = LogisticRegression(C=0.5, max_iter=3000).fit(Xt, y[tr]); pte = lr.predict_proba(Xte)[:,1]
    terms = vec.get_feature_names_out(); top = np.argsort(lr.coef_[0])[::-1][:25]; bot = np.argsort(lr.coef_[0])[:15]
    return oof, pte, {"top_risk_terms": [terms[i] for i in top], "top_protective_terms": [terms[i] for i in bot]}
# ---------------- embeddings (optional)
def emb_features():
    emb = pd.read_parquet(D/"note_emb_bge.parquet"); ecols = [c for c in emb.columns if re.fullmatch(r"e\d{3}", c)]
    pca = PCA(64, random_state=SEED).fit(emb.loc[emb.encounter_id.isin(df.decision_id[tr]), ecols].values)
    Z = pca.transform(emb[ecols].values); zc = [f"z{i:02d}" for i in range(64)]; emb = pd.concat([emb[["encounter_id","person_id","enc_date"]], pd.DataFrame(Z, columns=zc)], axis=1); emb["enc_date"] = pd.to_datetime(emb.enc_date)
    cur = df[["decision_id"]].merge(emb.drop(columns=["person_id","enc_date"]), left_on="decision_id", right_on="encounter_id", how="left")[zc].fillna(0).values
    G = {p: (g.enc_date.values, g[zc].values) for p, g in emb.groupby("person_id")}; prior = np.zeros((len(df), 64), dtype=np.float32)
    for i, r in enumerate(df.itertuples()):
        d, z = G.get(r.person_id, (np.array([], dtype="datetime64[ns]"), None)); t = r.enc_date.to_datetime64()
        if z is not None:
            m = (d < t) & (d >= t - np.timedelta64(90, "D"))
            if m.any(): prior[i] = z[m].mean(0)
    return np.column_stack([cur, prior]), {"pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4)}
# ---------------- metrics
def net_benefit(y, p, flag_frac):
    thr = np.quantile(p, 1 - flag_frac); flag = p >= thr; n = len(y); tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum(); pt = thr
    return float(tp/n - fp/n * (pt/(1-pt))), float(tp/max(flag.sum(),1)), float(tp/max((y==1).sum(),1)), float(flag.mean())
def calib(y, p):
    lp = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1,1), y); return float(lr.coef_[0][0]), float(lr.intercept_[0])
def evaluate(y, p):
    out = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    out["calibration_slope"], out["calibration_intercept"] = calib(y, p)
    for f in (0.10, 0.20, 0.30):
        nb, ppv, sens, fl = net_benefit(y, p, f); out[f"nb_flag{int(f*100)}"] = nb; out[f"ppv_flag{int(f*100)}"] = ppv; out[f"sens_flag{int(f*100)}"] = sens
    out["nb_treat_all"] = float((y.mean()) - (1-y.mean()) * (np.quantile(p, 0.5)/(1-np.quantile(p,0.5))))  # reference at median threshold
    return {k: round(float(v), 4) for k, v in out.items()}
# ---------------- fit models
y = df.y_primary.values; preds = {}; extras = {}
preds["M0_signal_risk"] = gbm().fit(df[["risk_percentile"]].fillna(0).values[tr], y[tr]).predict_proba(df[["risk_percentile"]].fillna(0).values[te])[:,1]
preds["M1_structured"] = gbm().fit(X_struct.values[tr], y[tr]).predict_proba(X_struct.values[te])[:,1]
preds["M2_structured_history"] = gbm().fit(X_hist.values[tr], y[tr]).predict_proba(X_hist.values[te])[:,1]
oof_c, pte_c, terms_c = text_stack(df.cur_text.values, y); oof_p, pte_p, terms_p = text_stack(df.prior_text.values, y); extras["tfidf_terms_current"] = terms_c; extras["tfidf_terms_prior90"] = terms_p
X3_tr = np.column_stack([X_hist.values[tr], oof_c, oof_p]); X3_te = np.column_stack([X_hist.values[te], pte_c, pte_p])
preds["M3_plus_tfidf_text"] = gbm().fit(X3_tr, y[tr]).predict_proba(X3_te)[:,1]
preds["text_only_tfidf"] = 0.5*(pte_c + pte_p)
X_lt = pd.concat([X_hist, df[LEX+TAGS].astype(float)], axis=1).fillna(0).values
preds["M3b_plus_lexicon_tags"] = gbm().fit(np.column_stack([X_lt[tr], oof_c, oof_p]), y[tr]).predict_proba(np.column_stack([X_lt[te], pte_c, pte_p]))[:,1]
if WITH_EMB:
    Zemb, emb_meta = emb_features(); extras["embedding"] = emb_meta
    preds["M4_plus_embeddings"] = gbm().fit(np.column_stack([X_hist.values[tr], Zemb[tr]]), y[tr]).predict_proba(np.column_stack([X_hist.values[te], Zemb[te]]))[:,1]
    X5_tr = np.column_stack([X_lt[tr], Zemb[tr], oof_c, oof_p]); X5_te = np.column_stack([X_lt[te], Zemb[te], pte_c, pte_p])
    m5 = gbm().fit(X5_tr, y[tr]); preds["M5_full"] = m5.predict_proba(X5_te)[:,1]
    preds["embeddings_only"] = gbm().fit(Zemb[tr], y[tr]).predict_proba(Zemb[te])[:,1]
    np.save(D/"X5_te.npy", X5_te); np.save(D/"X5_tr.npy", X5_tr)
# ---------------- evaluate + patient-cluster bootstrap for pre-specified contrasts
res = {"n_test": int(te.sum()), "n_test_patients": int(df.person_id[te].nunique()), "test_event_rate": round(float(y[te].mean()), 4), "models": {k: evaluate(y[te], v) for k, v in preds.items()}}
pids = df.person_id.values[te]; up = np.unique(pids); idx = {p: np.where(pids == p)[0] for p in up}; B = 1000; rngb = np.random.default_rng(SEED)
boots = [np.concatenate([idx[p] for p in rngb.choice(up, len(up))]) for _ in range(B)]
full = "M5_full" if WITH_EMB else "M3b_plus_lexicon_tags"
contrasts = {"full_vs_structured_history": (full, "M2_structured_history"), "structured_history_vs_signal": ("M2_structured_history", "M0_signal_risk"), "text_vs_structured_history": ("M3_plus_tfidf_text", "M2_structured_history"), "full_vs_structured": (full, "M1_structured")}
res["contrasts"] = {}
for name, (a, b) in contrasts.items():
    d = {}
    for metric, fn in [("auprc", average_precision_score), ("auroc", roc_auc_score), ("nb_flag20", lambda yy, pp: net_benefit(yy, pp, 0.20)[0]), ("ppv_flag20", lambda yy, pp: net_benefit(yy, pp, 0.20)[1])]:
        est = fn(y[te], preds[a]) - fn(y[te], preds[b]); bs = np.array([fn(y[te][ix], preds[a][ix]) - fn(y[te][ix], preds[b][ix]) for ix in boots])
        d[metric] = {"diff": round(float(est), 4), "ci_95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], "p": round(float(2*min((bs >= 0).mean(), (bs <= 0).mean())), 4)}
    res["contrasts"][name] = d
# secondary outcomes for M2 and full
res["secondary_outcomes"] = {}
for yc in ["y_explicit", "y_silent", "y_30"]:
    yy = df[yc].values; block = {"test_event_rate": round(float(yy[te].mean()), 4)}
    block["M2_structured_history"] = evaluate(yy[te], gbm().fit(X_hist.values[tr], yy[tr]).predict_proba(X_hist.values[te])[:,1])
    o_c, p_c, _ = text_stack(df.cur_text.values, yy); o_p, p_p, _ = text_stack(df.prior_text.values, yy)
    Xa_tr = np.column_stack([X_lt[tr], o_c, o_p] + ([Zemb[tr]] if WITH_EMB else [])); Xa_te = np.column_stack([X_lt[te], p_c, p_p] + ([Zemb[te]] if WITH_EMB else []))
    block[full] = evaluate(yy[te], gbm().fit(Xa_tr, yy[tr]).predict_proba(Xa_te)[:,1]); res["secondary_outcomes"][yc] = block
res["extras"] = extras; res["with_embeddings"] = WITH_EMB
json.dump(res, open(R/"models_primary.json", "w"), indent=1)
pd.DataFrame({"decision_id": df.decision_id.values[te], "person_id": pids, "state": df.state.values[te], "race": df.race.values[te], "days_from_zd": df.days_from_zd.values[te], "cur_rg": df.cur_rg.values[te], "y": y[te], **{k: v for k, v in preds.items()}}).to_parquet(D/"test_predictions.parquet")
for k, v in res["models"].items(): print(f"{k:28s} AUROC {v['auroc']:.3f} AUPRC {v['auprc']:.3f} slope {v['calibration_slope']:.2f} NB@20 {v['nb_flag20']:.4f} PPV@20 {v['ppv_flag20']:.3f}")
for k, v in res["contrasts"].items(): print(k, {m: (d['diff'], d['ci_95']) for m, d in v.items()})
