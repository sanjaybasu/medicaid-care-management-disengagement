"""Step 1: outcomes and eligibility per PREREGISTRATION_v3 Section 4 and Section 7.
Inputs (data_cache/): state_features.parquet (decision points), encounters.parquet, status_history.parquet, enrollment_spans.parquet.
Outputs: data_cache/dp_outcomes.parquet; results/flow.json"""
import pandas as pd, numpy as np, json, pathlib
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
sf = pd.read_parquet(D/"state_features.parquet"); sf["enc_date"] = pd.to_datetime(sf.enc_date); sf = sf.reset_index(drop=True)
enc = pd.read_parquet(D/"encounters.parquet"); enc["enc_date"] = pd.to_datetime(enc.enc_date); enc = enc.dropna(subset=["enc_date"])
sh = pd.read_parquet(D/"status_history.parquet"); sh["dt"] = pd.to_datetime(sh.updated_at)
sp = pd.read_parquet(D/"enrollment_spans.parquet"); sp["s"] = pd.to_datetime(sp.enrollment_start_date); sp["e"] = pd.to_datetime(sp.enrollment_end_date).fillna(pd.Timestamp("2026-12-31"))
E = {p: g.enc_date.values for p, g in enc.groupby("person_id")}
S = {p: (g.dt.values, g.new_status.values) for p, g in sh.groupby("person_id")}
SP = {p: (g.s.values, g.e.values) for p, g in sp.groupby("person_id")}
DIS = {"DROPPED_OUT_OF_CONTACT","WITHDRAWN_PATIENT","REFUSED_NO","REFUSED_MAYBE"}; GRAD = {"GRADUATED","PREGRADUATION"}
ADMIN = {"NOT_ELIGIBLE","TARGETED","ASSIGNED","OUTREACH","IN_CONTACT"}; WM = {"WITHDRAWN_WAYMARK"}
empty = np.array([], dtype="datetime64[ns]"); D1 = np.timedelta64(1, "D")
def enrolled_days(p, a, b):
    s, e = SP.get(p, (empty, empty)); 
    if len(s) == 0: return 0
    lo = np.maximum(s, a); hi = np.minimum(e, b); d = ((hi - lo) / D1).astype(float) + 1
    return float(np.clip(d, 0, None).sum())
rows = []
for r in sf.itertuples():
    t = r.enc_date.to_datetime64(); d = E.get(r.person_id, empty); sd, ss = S.get(r.person_id, (empty, np.array([])))
    w90 = (d > t) & (d <= t + 90*D1); w30 = (d > t) & (d <= t + 30*D1); w31_120 = (d > t + 30*D1) & (d <= t + 120*D1)
    st90 = set(ss[(sd > t) & (sd <= t + 90*D1)]); st120 = set(ss[(sd > t) & (sd <= t + 120*D1)])
    en90 = enrolled_days(r.person_id, t + D1, t + 90*D1); en120 = enrolled_days(r.person_id, t + 31*D1, t + 120*D1)
    rows.append(dict(decision_id=r.encounter_id, person_id=r.person_id, enc_date=r.enc_date, training_era=r.training_era, state=r.state,
        n_contacts_90=int(w90.sum()), no_contact_90=int(not w90.any()), no_contact_30=int(not w30.any()), grad_90=int(bool(st90 & GRAD)),
        dis_status_90=int(bool(st90 & DIS)), admin_90=int(bool(st90 & ADMIN)), wm_90=int(bool(st90 & WM)), enrolled_days_90=en90,
        no_contact_31_120=int(not w31_120.any()), grad_120=int(bool(st120 & GRAD)), enrolled_days_31_120=en120))
o = pd.DataFrame(rows)
CUT = pd.Timestamp("2025-07-01")  # test era begins; training-era patients' contacts on or after this date are excluded so training and test do not overlap in calendar time
o["eligible_landmark"] = (o.enrolled_days_90 >= 90).astype(int)  # enrollment criterion alone (used by the rolling-landmark and quarterly-refit analyses, which split by date)
o["training_overlap_excluded"] = ((o.enrolled_days_90 >= 90) & (o.training_era == 1) & (o.enc_date >= CUT)).astype(int)
o["eligible"] = ((o.enrolled_days_90 >= 90) & ~((o.training_era == 1) & (o.enc_date >= CUT))).astype(int)
o["y_primary"] = ((o.no_contact_90 == 1) & (o.grad_90 == 0)).astype(int)          # disengagement within 90 days
o["y_explicit"] = o.dis_status_90                                                     # secondary a
o["y_silent"] = ((o.y_primary == 1) & (o.admin_90 == 0)).astype(int)               # secondary b
o["y_30"] = o.no_contact_30                                                           # secondary c
o["lever_eligible"] = ((o.enrolled_days_31_120 >= 90) & (o.enc_date <= pd.Timestamp("2026-05-09"))).astype(int)  # 120-day window complete at the 2026-09-06 pull
o["y_lever"] = ((o.no_contact_31_120 == 1) & (o.grad_120 == 0)).astype(int)         # lever outcome, days 31-120
o.to_parquet(D/"dp_outcomes.parquet")
el = o[o.eligible == 1]; te = el[el.training_era == 0]
flow = {"decision_points": len(o), "patients": int(o.person_id.nunique()), "eligible_90_enrolled_days": int(o.eligible_landmark.sum()), "eligible_pct": round(100*o.eligible_landmark.mean(), 1), "excluded_training_overlap": int(o.training_overlap_excluded.sum()), "eligible_analysis": int(o.eligible.sum()), "test_window": [str(te.enc_date.min().date()), str(te.enc_date.max().date())], "training_window": [str(el[el.training_era == 1].enc_date.min().date()), str(el[el.training_era == 1].enc_date.max().date())],
        "eligible_train": int((el.training_era == 1).sum()), "eligible_test": int(len(te)), "eligible_test_patients": int(te.person_id.nunique()),
        "primary_rate_all": round(el.y_primary.mean(), 4), "primary_rate_test": round(te.y_primary.mean(), 4), "primary_events_test": int(te.y_primary.sum()),
        "explicit_rate": round(el.y_explicit.mean(), 4), "silent_rate": round(el.y_silent.mean(), 4), "no_contact_30_rate": round(el.y_30.mean(), 4),
        "graduation_90_rate": round(el.grad_90.mean(), 4), "withdrawn_by_waymark_rate": round(el.wm_90.mean(), 4),
        "primary_rate_by_state": el.groupby("state").y_primary.mean().round(4).to_dict(),
        "primary_event_composition": {"explicit_status": round(el.loc[el.y_primary==1, "dis_status_90"].mean(), 3), "administrative_status": round(el.loc[el.y_primary==1, "admin_90"].mean(), 3)},
        "lever_eligible": int(o.lever_eligible.sum()), "lever_outcome_rate": round(o.loc[o.lever_eligible==1, "y_lever"].mean(), 4)}
json.dump(flow, open(R/"flow.json", "w"), indent=1); print(json.dumps(flow, indent=1))
