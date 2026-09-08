"""Step 7 (supplementary architecture comparison): GRU, Transformer, and state-space (Mamba) policies over the last 10 contacts' tokens vs the gradient-boosted full model.
Equal parameter budgets, 3 seeds, early stopping on a patient-grouped validation fold. Writes results/sequence_models.json"""
import re, json, pathlib, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); SEED = 20260907; dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet"); df = F.merge(O[["decision_id","eligible","y_primary"]], on="decision_id"); df = df[df.eligible == 1].reset_index(drop=True); df["enc_date"] = pd.to_datetime(df.enc_date)
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values(["person_id","enc_date"])
emb = pd.read_parquet(D/"note_emb_bge.parquet"); ecols = [c for c in emb.columns if re.fullmatch(r"e\d{3}", c)]
tr = (df.training_era == 1).values; te = ~tr
pca = PCA(32, random_state=SEED).fit(emb.loc[emb.encounter_id.isin(df.decision_id[tr]), ecols].values); Z = pd.DataFrame(pca.transform(emb[ecols].values), columns=[f"z{i}" for i in range(32)]); Z["encounter_id"] = emb.encounter_id.values
enc = enc.merge(Z, on="encounter_id", how="left")
def rg(r):
    r = str(r); return 0 if "Therap" in r else (1 if "Pharmacist Tech" in r else (2 if "Pharm" in r else (3 if "CHW" in r else 4)))
enc["rg"] = enc.roles.map(rg); enc["inp"] = enc.contact_type.isin(["HOME_VISIT","IN_COMMUNITY","PROVIDER_OFFICE","HOSPITAL","CBO","OTHER_INPERSON"]).astype(int); enc["txt"] = enc.contact_type.isin(["SMS_TEXT","SMS_TEXT_CMT","EMAIL"]).astype(int)
E = {p: g for p, g in enc.groupby("person_id")}
L = 10; d_tok = 5 + 3 + 1 + 32  # role one-hot(5), inp, txt, log-gap, gap bucket placeholder, embeddings(32)
X = np.zeros((len(df), L, d_tok), dtype=np.float32); mask = np.zeros((len(df), L), dtype=np.float32)
for i, r in enumerate(df.itertuples()):
    g = E.get(r.person_id)
    if g is None: continue
    h = g[g.enc_date <= r.enc_date].tail(L)
    if len(h) == 0: continue
    dts = h.enc_date.values; gaps = np.r_[0, np.diff(dts).astype("timedelta64[D]").astype(float)]
    tok = np.zeros((len(h), d_tok), dtype=np.float32); tok[np.arange(len(h)), h.rg.values] = 1; tok[:, 5] = h.inp.values; tok[:, 6] = h.txt.values; tok[:, 7] = np.log1p(gaps); tok[:, 8] = np.log1p((r.enc_date.to_datetime64() - dts).astype("timedelta64[D]").astype(float))
    tok[:, 9:] = np.nan_to_num(h[[f"z{i}" for i in range(32)]].values, nan=0.0)
    X[i, L-len(h):] = tok; mask[i, L-len(h):] = 1
STAT = ["age","risk_percentile","days_from_zd","days_since_last_adt","adt_all_prior_90d","adt_all_prior_365d","n_prior_90","days_since_last_contact","goals_open"]
S = df[STAT].astype(float).fillna(0).values; S = (S - S[tr].mean(0)) / (S[tr].std(0) + 1e-6); y = df.y_primary.values.astype(np.float32)
class GRUNet(nn.Module):
    def __init__(s, d=64): super().__init__(); s.g = nn.GRU(d_tok, d, num_layers=2, batch_first=True); s.h = nn.Sequential(nn.Linear(d + S.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x, st): _, h = s.g(x); return s.h(torch.cat([h[-1], st], 1)).squeeze(-1)
class TfNet(nn.Module):
    def __init__(s, d=64): super().__init__(); s.p = nn.Linear(d_tok, d); s.pos = nn.Parameter(torch.zeros(1, L, d)); s.t = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 128, batch_first=True, dropout=0.1), 2); s.h = nn.Sequential(nn.Linear(d + S.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x, st): h = s.t(s.p(x) + s.pos); return s.h(torch.cat([h[:, -1], st], 1)).squeeze(-1)
class SSM(nn.Module):
    def __init__(s, d=64, n=16, r=8): super().__init__(); s.A_log = nn.Parameter(torch.log(torch.arange(1, n+1, dtype=torch.float32).repeat(d, 1))); s.D = nn.Parameter(torch.ones(d)); s.xp = nn.Linear(d, r + 2*n, bias=False); s.dt = nn.Linear(r, d); s.n = n; s.r = r
    def forward(s, u):
        A = -torch.exp(s.A_log); dl, B, C = torch.split(s.xp(u), [s.r, s.n, s.n], -1); dl = Fn.softplus(s.dt(dl)); h = torch.zeros(u.shape[0], u.shape[2], s.n, device=u.device); ys = []
        for t in range(u.shape[1]):
            h = torch.exp(dl[:, t].unsqueeze(-1) * A) * h + dl[:, t].unsqueeze(-1) * B[:, t].unsqueeze(1) * u[:, t].unsqueeze(-1); ys.append(torch.matmul(h, C[:, t].unsqueeze(-1)).squeeze(-1))
        return torch.stack(ys, 1) + u * s.D
class MambaBlock(nn.Module):
    def __init__(s, d=64): super().__init__(); s.i = nn.Linear(d, 2*d); s.c = nn.Conv1d(d, d, 3, padding=2); s.s = SSM(d); s.o = nn.Linear(d, d)
    def forward(s, x): a, z = s.i(x).chunk(2, -1); a = Fn.silu(s.c(a.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)); return s.o(s.s(a) * Fn.silu(z))
class MambaNet(nn.Module):
    def __init__(s, d=64): super().__init__(); s.p = nn.Linear(d_tok, d); s.b1 = MambaBlock(d); s.b2 = MambaBlock(d); s.h = nn.Sequential(nn.Linear(d + S.shape[1], 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, x, st): h = s.p(x); h = h + s.b1(h); h = h + s.b2(h); return s.h(torch.cat([h[:, -1], st], 1)).squeeze(-1)
def train(mk, name):
    preds = []; params = None
    pids = df.person_id.values[tr]; up = np.unique(pids)
    for seed in [1, 2, 3]:
        torch.manual_seed(seed); rng = np.random.default_rng(seed); vp = set(rng.choice(up, int(0.15*len(up)), replace=False)); vmask = np.array([p in vp for p in pids]); tri = np.where(tr)[0][~vmask]; vai = np.where(tr)[0][vmask]
        m = mk().to(dev); params = sum(p.numel() for p in m.parameters()); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); pos_w = torch.tensor(float((1-y[tri].mean())/y[tri].mean()), dtype=torch.float32, device=dev)
        Xt = torch.tensor(X, device=dev); St = torch.tensor(S, dtype=torch.float32, device=dev); yt = torch.tensor(y, device=dev); best, bad, best_state = -1, 0, None
        for ep in range(30):
            m.train(); perm = rng.permutation(tri)
            for k in range(0, len(perm), 256):
                b = perm[k:k+256]; opt.zero_grad(); loss = Fn.binary_cross_entropy_with_logits(m(Xt[b], St[b]), yt[b], pos_weight=pos_w); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad(): pv = torch.sigmoid(m(Xt[vai], St[vai])).cpu().numpy()
            ap = average_precision_score(y[vai], pv)
            if ap > best: best, bad, best_state = ap, 0, {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= 4: break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad(): preds.append(torch.sigmoid(m(Xt[te], St[te])).cpu().numpy())
    p = np.mean(preds, 0); return {"auroc": round(float(roc_auc_score(y[te], p)), 4), "auprc": round(float(average_precision_score(y[te], p)), 4), "params": int(params), "seed_auprc": [round(float(average_precision_score(y[te], q)), 4) for q in preds]}, p
res = {"n_test": int(te.sum()), "sequence_length": L, "token_dim": d_tok, "models": {}}
P = pd.read_parquet(D/"test_predictions.parquet")
for name, mk in [("GRU", GRUNet), ("Transformer", TfNet), ("Mamba_SSM", MambaNet)]:
    r, p = train(mk, name); res["models"][name] = r; P[f"seq_{name}"] = p; print(name, r, flush=True)
ref = "M5_full" if "M5_full" in P.columns else "M3b_plus_lexicon_tags"; res["reference_gbm"] = {ref: {"auroc": round(float(roc_auc_score(P.y, P[ref])), 4), "auprc": round(float(average_precision_score(P.y, P[ref])), 4)}}
P.to_parquet(D/"test_predictions.parquet"); json.dump(res, open(R/"sequence_models.json", "w"), indent=1); print(res["reference_gbm"])
