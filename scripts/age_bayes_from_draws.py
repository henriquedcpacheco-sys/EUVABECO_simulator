"""Make Model 3b's Bayesian summary match its own draws.

The exported age_bayes_params.csv / age_bayes_fit.csv came from a different
MCMC run than age_bayes_draws.csv and age_bayes_bands.csv (the curve, bands
and histograms shown in the app). This rewrites the two summary files from
the draws, so every Model 3b number in the app describes the same run:
  params: posterior mean and 95% interval of the 5000 draws (the
          least-squares estimate is kept);
  fit:    Error J of the posterior-mean curve (point_est in the bands file)
          and over 500 draws evenly spaced in the draws file.

    python scripts/age_bayes_from_draws.py path/to/data
then run build_data.py on the same folder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import AGE_MULT, AGE_NTNORM, SERIES, fitted_schedule, simulate  # noqa: E402

SRC = Path(sys.argv[1])
draws = pd.read_csv(SRC / "age_bayes_draws.csv")
bands = pd.read_csv(SRC / "age_bayes_bands.csv")
cov = pd.read_csv(SRC / "age_covariates.csv").sort_values("t")
obs = {s: bands[bands["series"] == s].sort_values("t")["obs"].to_numpy() for s in SERIES}

# ---- parameters
params = pd.read_csv(SRC / "age_bayes_params.csv")
for i, name in enumerate(params["name"]):
    params.loc[i, "post_mean"] = draws[name].mean()
    params.loc[i, "ci_low"], params.loc[i, "ci_high"] = draws[name].quantile([0.025, 0.975])
params.to_csv(SRC / "age_bayes_params.csv", index=False, float_format="%.6f")


def error_j(sim, s):
    return np.sum((np.log(np.maximum(sim, 0) + 1) - np.log(obs[s] + 1)[:, None]) ** 2, axis=0)


# ---- Error J of the posterior-mean curve
point = {s: bands[bands["series"] == s].sort_values("t")["point_est"].to_numpy()[:, None] for s in SERIES}
j_mean = {s: float(error_j(point[s], s)[0]) for s in SERIES}

# ---- Error J over 500 draws
d = draws.iloc[np.round(np.linspace(0, len(draws) - 1, 500)).astype(int)]
mix = cov[["p_0_49", "p_50_59", "p_60_69", "p_70plus"]].to_numpy() @ AGE_MULT
psi = np.exp(-np.outer(cov["NT"], d["F_test"]) / AGE_NTNORM) * d["psi_base"].to_numpy() * mix[:, None]
inp = dict(theta=d["theta"].to_numpy(), phi_h=d["phi_h"].to_numpy(), phi_c=(d["r_c"] * d["phi_h"]).to_numpy(),
           E0=d["E0"].to_numpy(), I0=1.0, psi=psi)
tr = simulate(inp, fitted_schedule([d[f"beta{k}"].to_numpy() for k in range(1, 6)]), n=len(d))
j_draws = {s: error_j(tr[s], s) for s in SERIES}
j_draws["total"] = sum(j_draws[s] for s in SERIES)
j_mean["total"] = sum(j_mean[s] for s in SERIES)

cols = [*SERIES, "total"]
fit = pd.DataFrame([
    ["J_post_mean_params", *[j_mean[c] for c in cols]],
    ["J_draws_mean", *[j_draws[c].mean() for c in cols]],
    ["J_draws_lo", *[np.quantile(j_draws[c], 0.025) for c in cols]],
    ["J_draws_hi", *[np.quantile(j_draws[c], 0.975) for c in cols]],
], columns=["quantity", *cols])
fit.to_csv(SRC / "age_bayes_fit.csv", index=False, float_format="%.4f")
print(params.to_string(index=False))
print(fit.to_string(index=False))
