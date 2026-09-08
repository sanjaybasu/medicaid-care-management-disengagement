"""Step 21: scope of the program for the analytic cohort (generalizability descriptors): distinct assigned and attributed primary care
clinicians (NPIs), distinct practices (tax identification numbers), distinct payer-attributed practices, and distinct health-system partner
entities, computed for the eligible members. Reads dbt.im__attributed_pcp and dbt.financial_pmpm__stg_provider_attribution from coredb
(read-only) and data_cache/monthly_panel.parquet. Writes results/program_scope.json (aggregate counts only)."""
import sys, json, pathlib, pandas as pd; sys.path.insert(0, "/Users/sanjaybasu/.claude/skills/waymark-data-access/scripts"); import wm_conn as w
D = pathlib.Path("data_cache"); R = pathlib.Path("results"); eng = w.coredb("prod")
O = pd.read_parquet(D/"dp_outcomes.parquet"); ids = set(O[O.eligible == 1].person_id)
a = w.query(eng, "select person_id, assigned_pcp_npi, assigned_pcp_specialty, assigned_tin, assigned_tin_name, attributed_pcp_npi, attributed_tin from dbt.im__attributed_pcp"); a = a[a.person_id.isin(ids)]
pa = w.query(eng, "select person_id, payer_attributed_provider, payer_attributed_provider_practice, payer_attributed_provider_organization from dbt.financial_pmpm__stg_provider_attribution"); pa = pa[pa.person_id.isin(ids)]
mp = pd.read_parquet(D/"monthly_panel.parquet", columns=["person_id","entity","market","state"]); mp = mp[mp.person_id.isin(ids)]
out = {"eligible_members": len(ids), "members_in_attribution_table": int(a.person_id.nunique()),
       "members_with_assigned_pcp": int(a.dropna(subset=["assigned_pcp_npi"]).person_id.nunique()), "distinct_assigned_pcp_npi": int(a.assigned_pcp_npi.dropna().nunique()), "distinct_attributed_pcp_npi": int(a.attributed_pcp_npi.dropna().nunique()),
       "distinct_pcp_npi_assigned_or_attributed": int(pd.concat([a.assigned_pcp_npi, a.attributed_pcp_npi]).dropna().nunique()),
       "distinct_assigned_tin": int(a.assigned_tin.dropna().nunique()), "distinct_assigned_tin_name": int(a.assigned_tin_name.dropna().nunique()), "distinct_attributed_tin": int(a.attributed_tin.dropna().nunique()),
       "members_in_payer_attribution": int(pa.person_id.nunique()), "distinct_payer_attributed_provider": int(pa.payer_attributed_provider.dropna().nunique()), "distinct_payer_attributed_practice": int(pa.payer_attributed_provider_practice.dropna().nunique()), "distinct_payer_attributed_organization": int(pa.payer_attributed_provider_organization.dropna().nunique()),
       "distinct_partner_entities": int(mp.entity.dropna().nunique()), "distinct_markets": int(mp.market.dropna().nunique()), "partner_entities": sorted(mp.entity.dropna().unique().tolist()),
       "assigned_pcp_specialty_top": a.assigned_pcp_specialty.value_counts().head(8).to_dict()}
json.dump(out, open(R/"program_scope.json", "w"), indent=1); print(json.dumps(out, indent=1))
