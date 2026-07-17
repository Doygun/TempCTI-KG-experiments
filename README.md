# TempCTI-KG Experiments

Code for the paper *"TempCTI-KG: Temporal Validity and Confidence Decay Modeling for Cyber Threat Intelligence Knowledge Graphs"*.

## Overview

TempCTI-KG models the temporal validity of cyber threat intelligence facts in a knowledge graph. Facts are represented as quintuples with observation time, validity interval, and a confidence value that decays over time according to relation-type-specific survival models (exponential and Weibull) fitted on real CTI lifetime data.

Data sources:

- MITRE ATT&CK Enterprise (40 releases, v1.0 - v19.1)
- abuse.ch URLhaus (full database dump)
- abuse.ch ThreatFox
- abuse.ch MalwareBazaar

## Scripts

| Script | Purpose |
|---|---|
| `00_download_attck.py` | Download all ATT&CK Enterprise STIX releases |
| `01_parse_attck_lifetimes.py` | Extract fact lifetimes across ATT&CK versions |
| `02_fetch_abusech.py` | Fetch URLhaus / ThreatFox / MalwareBazaar feeds |
| `02b_rebuild_merged.py` | Merge sources with full-coverage window filtering |
| `02c_parse_urlhaus_full.py` | Parse the full URLhaus database dump |
| `03_survival_analysis.py` | Kaplan-Meier + exponential/Weibull fits per relation type |
| `04_validity_prediction.py` | Task 1: temporal validity prediction (AUC, Brier, bootstrap CIs) |
| `05_downstream_fp.py` | Task 3: downstream stale-retrieval simulation |
| `05_run_all.py` | Run the full pipeline |
| `06_ablation.py` | Ablation study |
| `07_versioning_scalability.py` | Bi-temporal versioning and scalability benchmarks |
| `08_replay_eval.py` | Task 4: real-feed replay evaluation (URLhaus) |
| `09_case_study_kg.py` | Amadey case-study knowledge graph figure |
| `10_per_source_auc.py` | Per-source AUC breakdown |

## Requirements

```
pip install -r requirements.txt
```

Python 3.11+ with `numpy`, `pandas`, `scipy`, `scikit-learn`, `lifelines`, `matplotlib`, `networkx`, `requests`, `stix2`, `mitreattack-python`.

## Reproducing the results

```
python scripts/00_download_attck.py
python scripts/01_parse_attck_lifetimes.py
python scripts/02_fetch_abusech.py
python scripts/02c_parse_urlhaus_full.py
python scripts/02b_rebuild_merged.py
python scripts/05_run_all.py
```

Results (JSON/CSV) are written to `results/`, figures to `results/figures/`.

Note: abuse.ch feeds are live; re-fetching will produce a newer snapshot than the one used in the paper. The processed lifetime tables used for the published results are included under `data/`.

## Citation

If you use this code, please cite the paper (citation will be added upon acceptance).

## Contact

Doygun Demirol — <ddemirol@bingol.edu.tr>
