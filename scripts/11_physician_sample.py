"""Step 11 (amendment v3.1 Section B): build the blinded physician review sample. All packet files contain PHI and stay in data_cache/ (never versioned).
Phase 1: 200 test-era decision points stratified by full-model risk stratum x realized outcome, one per member; packets show only information up to the index contact.
Phase 2: disengaged cases with post-index notes and status changes (days 0-120). Phase 3: 20 narrative patterns with three snippets each.
Writes data_cache/physician_review/{phase1,phase2,phase3}/..., ANSWER_KEY_do_not_share.csv, assignment.csv, and results/physician_sample_meta.json (counts only)."""
import json, pathlib, re, html, numpy as np, pandas as pd
def clean(t):
    t = html.unescape(re.sub(r"<[^>]+>", " ", str(t or ""))); return re.sub(r"[ \t]+", " ", re.sub(r"\s*\n\s*", "\n", t)).strip()
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); PR = D/"physician_review"; SEED = 20260908; rng = np.random.default_rng(SEED)
P = pd.read_parquet(D/"test_predictions.parquet"); F = pd.read_parquet(D/"dp_features.parquet"); O = pd.read_parquet(D/"dp_outcomes.parquet")
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"]).sort_values("enc_date")
sh = pd.read_parquet(D/"status_history.parquet"); sh["dt"] = pd.to_datetime(sh.updated_at)
df = P.merge(F.drop(columns=["person_id","state","race","days_from_zd","cur_rg"]), on="decision_id").merge(O[["decision_id","grad_90","dis_status_90","admin_90","n_contacts_90"]], on="decision_id"); df["enc_date"] = pd.to_datetime(df.enc_date)
df["decile"] = pd.qcut(df.M5_full, 10, labels=False) + 1
df["stratum"] = np.where(df.decile == 10, "top_decile", np.where(df.decile >= 4, "deciles_4_7", "bottom_3_deciles")); df = df[df.decile != 8]; df = df[df.decile != 9]  # deciles 8-9 unsampled by design
cells = [(s_, y_) for s_ in ["top_decile","deciles_4_7","bottom_3_deciles"] for y_ in (1, 0)]
picked = []; used = set()
for s_, y_ in cells:
    pool = df[(df.stratum == s_) & (df.y == y_) & (~df.person_id.isin(used))].sample(frac=1, random_state=int(rng.integers(1e9)))
    pool = pool.drop_duplicates("person_id").head(34 if len(picked) < 4*34 else 32); picked.append(pool); used |= set(pool.person_id)
S = pd.concat(picked).reset_index(drop=True).head(200); S["case_id"] = [f"C{k:04d}" for k in rng.permutation(np.arange(1000, 1000+len(S)))]
S = S.sort_values("case_id").reset_index(drop=True)
# pair assignment: reviewers A, B, C; pairs AB, BC, AC balanced; core 30 by all three
pairs = np.array(["AB","BC","AC"])[np.arange(len(S)) % 3]; rng.shuffle(pairs); S["reviewers"] = pairs
core = pd.concat([g.sample(5, random_state=SEED) for _, g in S.groupby(["stratum","y"])]).case_id; S.loc[S.case_id.isin(core), "reviewers"] = "ABC"
def band(a): return "under 18" if a < 18 else "18 to 44" if a < 45 else "45 to 64" if a < 65 else "65 and over"
COND = {"diabetes":"diabetes","htn":"hypertension","chf":"heart failure","copd":"COPD","sud":"substance use disorder","any_bh":"behavioral health condition","mdd":"depression","asthma":"asthma","polypharmacy":"polypharmacy","high_ed_ip":"high prior ED/inpatient use","no_pcp_last_10mo":"no primary care visit in prior 10 months"}
def packet1(r):
    t = r.enc_date; g = enc[(enc.person_id == r.person_id) & (enc.enc_date < t) & (enc.enc_date >= t - pd.Timedelta(days=90))]
    conds = ", ".join(v for k, v in COND.items() if getattr(r, k) == 1) or "none flagged"
    lines = [f"# Case {r.case_id}", "", f"**Structured summary (as of the index contact).** Age band {band(r.age)}; gender {r.gender}; state {r.state.title()}; race/ethnicity {r.race}; days since activation {int(r.days_from_zd)}; contacts in the prior 90 days {int(r.n_prior_90)}; days since last contact {int(r.days_since_last_contact) if pd.notna(r.days_since_last_contact) else 'none recorded'}; open goals {int(r.goals_open) if pd.notna(r.goals_open) else 0}; condition flags: {conds}.", "", f"**Index contact.** Discipline {r.cur_rg}; channel {r.contact_type}; encounter type {r.encounter_type}.", "", "## Prior 90 days of notes (oldest first)", ""]
    if len(g) == 0: lines.append("_No contacts in the prior 90 days._")
    for e in g.itertuples(): lines += [f"**Day -{(t - e.enc_date).days}** ({e.roles}; {e.contact_type}; {e.encounter_type})", "", clean(e.note_text) or "_(empty note)_", ""]
    lines += ["## Index note", "", clean(r.note_text) or "_(empty note)_", ""]
    return "\n".join(lines)
for r in S.itertuples(): (PR/"phase1"/f"case_{r.case_id}.md").write_text(packet1(r))
for rev in "ABC":
    sub = S[S.reviewers.str.contains(rev)]; (PR/"phase1"/f"reviewer_{rev}_all_cases.md").write_text("\n\n---\n\n".join(packet1(r) for r in sub.itertuples()))
    pd.DataFrame({"case_id": sub.case_id, "reviewer": rev, "q1_engagement_status": "", "q2_prognosis_1to5": "", "q3_open_need_medication": "", "q3_open_need_referral_or_appointment": "", "q3_open_need_uncontrolled_condition": "", "q3_open_need_social_crisis": "", "q3_open_need_behavioral_health": "", "q3_no_open_need": "", "q4_harm_if_lost": "", "q5_stated_intent": "", "comments": ""}).to_csv(PR/"phase1"/f"reviewer_{rev}_form.csv", index=False)
# phase 2: disengaged cases, post-index notes and status changes days 0-120, single reviewer each, 20% double-coded
S2 = S[S.y == 1].copy(); S2["reviewer2"] = np.array(list("ABC"))[np.arange(len(S2)) % 3]; dbl = S2.sample(frac=0.2, random_state=SEED).case_id; S2["double_coder"] = np.where(S2.case_id.isin(dbl), S2.reviewer2.map({"A":"B","B":"C","C":"A"}), "")
def packet2(r):
    t = r.enc_date; g = enc[(enc.person_id == r.person_id) & (enc.enc_date > t) & (enc.enc_date <= t + pd.Timedelta(days=120))]; st = sh[(sh.person_id == r.person_id) & (sh.dt >= t) & (sh.dt <= t + pd.Timedelta(days=120))]
    lines = [f"# Case {r.case_id} (phase 2)", "", f"See the phase 1 packet for this case. Outcome: no completed contact in the 90 days after the index contact. Graduation status in window: {'yes' if r.grad_90 else 'no'}.", "", "## Status changes, days 0 to 120", ""]
    lines += [f"- Day {(x.dt - t).days}: {x.old_status} to {x.new_status}" for x in st.itertuples()] or ["_No status change recorded._"]
    lines += ["", "## Any documented encounters, days 91 to 120 (none occur in days 1 to 90 by definition)", ""]
    lines += [f"**Day +{(e.enc_date - t).days}** ({e.roles}; {e.contact_type}; {e.encounter_type})\n\n{clean(e.note_text)}" for e in g.itertuples()] or ["_None._"]
    return "\n".join(lines)
for r in S2.itertuples(): (PR/"phase2"/f"case_{r.case_id}_phase2.md").write_text(packet2(r))
for rev in "ABC":
    sub = S2[(S2.reviewer2 == rev) | (S2.double_coder == rev)]; (PR/"phase2"/f"reviewer_{rev}_phase2_cases.md").write_text("\n\n---\n\n".join(packet2(r) for r in sub.itertuples()))
    pd.DataFrame({"case_id": sub.case_id, "reviewer": rev, "reason": "", "confidence_1to5": "", "comments": ""}).to_csv(PR/"phase2"/f"reviewer_{rev}_phase2_form.csv", index=False)
# phase 3: 20 narrative patterns with three snippets each from test-era notes
lex = json.load(open(R/"features_meta.json"))["lexicon_patterns"]; terms = json.load(open(R/"models_primary.json"))["extras"]["tfidf_terms_current"]["top_risk_terms"]
pats = [(k.replace("lex_",""), v) for k, v in lex.items()] + [(t, re.escape(t)) for t in terms if not re.search(r"\d", t)][:20 - len(lex)]
notes = [clean(n) for n in df.note_text.fillna("").values]; rows = []; lines = ["# Phase 3: narrative patterns", ""]
for name, pat in pats:
    hits = [n for n in notes if re.search(pat, n.lower())]; rng.shuffle(hits)
    lines += [f"## Pattern: {name}", "", f"Pattern definition (regular expression on lower-cased note text): `{pat}`", ""]
    for n in hits[:3]:
        m = re.search(pat, n.lower()); a, b = max(0, m.start()-200), min(len(n), m.end()+200); lines += ["> ..." + n[a:b].replace("\n", " ") + "...", ""]
    rows.append({"pattern": name, "clinical_meaningfulness_1to5": "", "what_it_indicates": "", "comments": ""})
(PR/"phase3"/"patterns_with_snippets.md").write_text("\n".join(lines)); pd.DataFrame(rows).to_csv(PR/"phase3"/"phase3_form.csv", index=False)
S[["case_id","decision_id","person_id","M5_full","decile","stratum","y","reviewers"]].to_csv(PR/"ANSWER_KEY_do_not_share.csv", index=False)
S[["case_id","reviewers"]].merge(S2[["case_id","reviewer2","double_coder"]], on="case_id", how="left").to_csv(PR/"assignment.csv", index=False)
meta = {"phase1_cases": int(len(S)), "cells": {f"{a}_y{b}": int(v) for (a, b), v in S.groupby(["stratum","y"]).size().items()}, "cases_per_reviewer_phase1": {rev: int(S.reviewers.str.contains(rev).sum()) for rev in "ABC"}, "core_all_three": int((S.reviewers == "ABC").sum()), "phase2_cases": int(len(S2)), "phase2_double_coded": int((S2.double_coder != "").sum()), "phase3_patterns": len(pats), "seed": SEED}
json.dump({str(k): v for k, v in meta.items()}, open(R/"physician_sample_meta.json", "w"), indent=1, default=str); print(json.dumps(meta, indent=1, default=str))
