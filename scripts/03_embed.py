"""Embed all encounter notes locally with BAAI/bge-base-en-v1.5 (no text leaves the machine). Output: data_cache/note_emb_bge.parquet (encounter_id, person_id, enc_date, 768-d)."""
import pandas as pd, numpy as np, torch, time, pathlib
from sentence_transformers import SentenceTransformer
D = pathlib.Path("data_cache"); out = D/"note_emb_bge.parquet"
enc = pd.read_parquet(D/"encounters.parquet")[["encounter_id","person_id","enc_date","note_text"]]
enc["note_text"] = enc.note_text.fillna("").str.slice(0, 4000)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev); m.max_seq_length = 384
t0 = time.time(); uniq, inv = np.unique(enc.note_text.values, return_inverse=True)
print(f"{len(enc)} notes, {len(uniq)} unique; device {dev}", flush=True)
E = m.encode(list(uniq), batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
emb = E[inv].astype(np.float32)
df = pd.DataFrame(emb, columns=[f"e{i:03d}" for i in range(emb.shape[1])]); df.insert(0,"encounter_id",enc.encounter_id.values); df.insert(1,"person_id",enc.person_id.values); df.insert(2,"enc_date",enc.enc_date.values)
df.to_parquet(out); print(f"saved {out} {df.shape} in {time.time()-t0:.0f}s", flush=True)
