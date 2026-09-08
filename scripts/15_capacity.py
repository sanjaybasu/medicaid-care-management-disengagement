"""Step 15 (amendment v3.1 Section E): weekly re-engagement capacity arithmetic on the test era. Writes results/capacity.json."""
import json, pathlib, numpy as np, pandas as pd
D = pathlib.Path("data_cache"); R = pathlib.Path("results")
P = pd.read_parquet(D/"test_predictions.parquet"); F = pd.read_parquet(D/"dp_features.parquet")[["decision_id","market","enc_date"]]; df = P.merge(F, on="decision_id"); df["week"] = pd.to_datetime(df.enc_date).dt.to_period("W").astype(str)
rng = np.random.default_rng(20260908); df["random"] = rng.random(len(df)); df["first30"] = (df.days_from_zd <= 30).astype(float) + rng.random(len(df))*1e-3
rules = {"full_model": "M5_full", "structured_history": "M2_structured_history", "deployed_risk_score": "M0_signal_risk", "first_30_days": "first30", "random": "random"}
out = {"unit": "true disengagements reached per care team per week, averaged over team-weeks in the test era; teams defined by market", "team_weeks": int(df.groupby(["market","week"]).ngroups), "mean_contacts_per_team_week": round(float(df.groupby(["market","week"]).size().mean()), 1), "mean_disengagements_per_team_week": round(float(df.groupby(["market","week"]).y.sum().mean()), 2), "capacity": {}}
for cap in (5, 10, 20, 40):
    out["capacity"][str(cap)] = {}
    for name, col in rules.items():
        reached = df.sort_values(col, ascending=False).groupby(["market","week"]).head(cap).groupby(["market","week"]).y.sum(); out["capacity"][str(cap)][name] = round(float(reached.mean()), 2)
json.dump(out, open(R/"capacity.json", "w"), indent=1); print(json.dumps(out, indent=1))
