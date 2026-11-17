"""After d2_app_bayes_export_uncorrected.m (thesis code, Tese/D2) has written the four
Bayesian files (sim_m1_bayes, sim_m2_bayes, age_nocorr_bayes, age_bayes .json), update the
two files that depend on Model 3b's posterior:
  age_draws.json   500 draws evenly spaced over the 5000 exported draws (scenario band);
                   the covariates NT and mix are kept
  comparison.json  the "Model 3b (Bayesian, posterior mean)" column

    python scripts/update_bayes_uncorrected.py path/to/Tese/D2/app_export/age_bayes_draws.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "public" / "data"
SERIES = ["cases", "ward", "icu", "deaths"]

draws = pd.read_csv(sys.argv[1])
idx = np.linspace(0, len(draws) - 1, 500).round().astype(int)
d = draws.iloc[idx]
ad = json.loads((DATA / "age_draws.json").read_text(encoding="utf-8"))
ad["draws"] = {c: [float(v) for v in d[c]] for c in d.columns}
(DATA / "age_draws.json").write_text(json.dumps(ad, separators=(",", ":")), encoding="utf-8")

ab = json.loads((DATA / "age_bayes.json").read_text(encoding="utf-8"))
comp = json.loads((DATA / "comparison.json").read_text(encoding="utf-8"))
col = comp["models"]["Model 3b (Bayesian, posterior mean)"]
pm = {p["name"].lower(): p["post_mean"] for p in ab["params"]}
col["params"] = {k: (float(pm[k]) if k in pm else None) for k in comp["order"]}
col["J"] = {k: float(ab["fit"]["J_post_mean_params"][k]) for k in SERIES + ["total"]}
(DATA / "comparison.json").write_text(json.dumps(comp, separators=(",", ":")), encoding="utf-8")
print("age_draws.json and comparison.json updated; Model 3b Bayesian J =", col["J"]["total"])
