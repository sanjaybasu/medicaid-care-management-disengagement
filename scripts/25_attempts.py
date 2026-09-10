"""Step 25: unsuccessful outreach attempts (EncounterNote.encounterOccurred = 'NO') around decision points, and time of day of contacts.
Reads data_cache/encounter_times_study.parquet (pulled 2026-09-10 from lighthouse), dp_features, dp_outcomes. Writes results/attempts.json."""
import json, pathlib, numpy as np, pandas as pd
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); D1 = np.timedelta64(1, "D")
en = pd.read_parquet(D/"encounter_times_study.parquet"); F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
d = F[["decision_id","person_id","enc_date","state","training_era"]].merge(O[["decision_id","eligible","y_primary","y_silent","y_explicit"]], on="decision_id"); d = d[d.eligible == 1].copy(); d["enc_date"] = pd.to_datetime(d.enc_date)
TZ = {"VIRGINIA": "America/New_York", "OHIO": "America/New_York", "WASHINGTON": "America/Los_Angeles"}
en["t"] = pd.to_datetime(en.startTime.fillna(en.dateOfEncounter)); en = en.dropna(subset=["t"])
att = en[en.occurred == "NO"]; A = {p: g.t.values for p, g in att.groupby("person_id")}
comp = en[en.occurred == "YES"]
def count(p, a, b):
    v = A.get(p); 
    return 0 if v is None else int(((v > a) & (v <= b)).sum())
d["att_after_90"] = [count(r.person_id, r.enc_date.to_datetime64(), r.enc_date.to_datetime64() + 90*D1) for r in d.itertuples()]
d["att_prior_30"] = [count(r.person_id, r.enc_date.to_datetime64() - 30*D1, r.enc_date.to_datetime64()) for r in d.itertuples()]
dis = d[d.y_primary == 1]; ret = d[d.y_primary == 0]
out = {"source": "lighthouse EncounterNote, encounterOccurred = 'NO' (attempt without two-way contact), pulled 2026-09-10; study patients only",
       "attempt_notes_study_patients": int(len(att)), "completed_notes_study_patients": int(len(comp)),
       "attempts_by_contact_type": att.contact_type.value_counts().head(6).to_dict(),
       "disengagements_with_any_attempt_in_90d_pct": round(100*float((dis.att_after_90 > 0).mean()), 1), "disengagements_with_3plus_attempts_pct": round(100*float((dis.att_after_90 >= 3).mean()), 1),
       "disengagements_median_attempts_90d": float(dis.att_after_90.median()), "disengagements_mean_attempts_90d": round(float(dis.att_after_90.mean()), 2),
       "silent_loss_with_any_attempt_pct": round(100*float((d[d.y_silent == 1].att_after_90 > 0).mean()), 1), "explicit_exit_with_any_attempt_pct": round(100*float((d[d.y_explicit == 1].att_after_90 > 0).mean()), 1),
       "retained_mean_attempts_90d": round(float(ret.att_after_90.mean()), 2),
       "disengagement_rate_by_prior30_attempts": {k: {"n": int(len(g)), "rate": round(float(g.y_primary.mean()), 4)} for k, g in d.groupby(pd.cut(d.att_prior_30, [-1, 0, 1, 2, 4, 1000], labels=["0","1","2","3-4","5+"]))},
       "test_era_disengagements_with_any_attempt_pct": round(100*float((dis[dis.training_era == 0].att_after_90 > 0).mean()), 1)}
# time of day of completed contacts and attempts (local time by state)
en2 = en.merge(F.drop_duplicates("person_id")[["person_id","state"]], on="person_id", how="inner")
loc = pd.Series([pd.Timestamp(t).tz_localize("UTC").tz_convert(TZ.get(s, "America/New_York")) for t, s in zip(en2.t, en2.state)])
en2["hour"] = [x.hour for x in loc]; en2["wd"] = [x.weekday() for x in loc]; en2["evening"] = ((en2.hour >= 17) & (en2.wd < 5)).astype(int); en2["weekend"] = (en2.wd >= 5).astype(int)
for lab, m in [("completed", en2.occurred == "YES"), ("attempts", en2.occurred == "NO")]:
    g = en2[m]; out[f"{lab}_share_weekday_evening_after_5pm_local"] = round(100*float(g.evening.mean()), 1); out[f"{lab}_share_weekend_local"] = round(100*float(g.weekend.mean()), 1); out[f"{lab}_hour_distribution_local"] = {int(k): round(float(v), 3) for k, v in g.hour.value_counts(normalize=True).sort_index().items()}
json.dump(out, open(R/"attempts.json", "w"), indent=1, default=str); print(json.dumps({k: v for k, v in out.items() if "distribution" not in k}, indent=1, default=str))
