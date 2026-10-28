"""Convert the precomputed fit results (the data/ folder of the EUVABECO_sim
Streamlit repo) into the compact JSON files the web app reads from
public/data/.

    python scripts/build_data.py path/to/EUVABECO_sim/data

Nothing is fitted here: the numbers are exactly those of the MATLAB pipeline
(least-squares fits) and the mcmcstat DRAM runs (Bayesian bands), only
reshaped. Run it again whenever those exports change.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../EUVABECO_sim/data")
OUT = Path(__file__).resolve().parent.parent / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

SERIES = ["cases", "ward", "icu", "deaths"]
POINT = ["sim_m1", "sim_m2", "age_nocorr", "age_point"]
BAYES = {"sim_m1": "sim_m1_bayes", "sim_m2": "sim_m2_bayes",
         "age_nocorr": "age_nocorr_bayes", "age_point": "age_bayes"}
N_SCENARIO_DRAWS = 500


def r(x, nd=4):
    """Round to nd significant digits, keeping JSON small."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(x == 0, 0, np.floor(np.log10(np.abs(x))))
    dec = np.clip(nd - 1 - mag, 0, 10).astype(int)
    return [float(round(v, int(d))) for v, d in zip(x, dec)]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    print(f"  {name:28s} {(OUT / name).stat().st_size / 1024:7.1f} kB")


def rt_for(key):
    rt = pd.read_csv(SRC / "rt.csv")
    rt = rt[rt["model"] == key].sort_values("t")
    return r(rt["Rt"], 5)


def by_series(df, col, nd=5):
    return {s: r(df[df["series"] == s].sort_values("t")[col], nd) for s in SERIES}


summary = pd.read_csv(SRC / "summary.csv").set_index("model")

# Observed series (identical in every model file) and the day-0 date.
obs_df = pd.read_csv(SRC / "seq_m1.csv")
first = obs_df[obs_df["series"] == "cases"].sort_values("t")
dump("observed.json", {
    "start": first["date"].iloc[0],          # date of t = 0 in the trajectory files
    "n": int(len(first)),
    **by_series(obs_df, "obs", 6),
})

for key in POINT:
    traj = pd.read_csv(SRC / f"{key}.csv")
    params = pd.read_csv(SRC / f"{key}_params.csv")
    s = summary.loc[key]
    dump(f"{key}.json", {
        "sim": by_series(traj, "sim"),
        "params": [{"name": n, "value": float(v)} for n, v in zip(params["name"], params["value"])],
        "J": {k: float(s[f"J_{k}"]) for k in SERIES + ["total"]},
        "rt": rt_for(key),
    })

for point_key, key in BAYES.items():
    bands = pd.read_csv(SRC / f"{key}_bands.csv")
    params = pd.read_csv(SRC / f"{key}_params.csv")
    fit = pd.read_csv(SRC / f"{key}_fit.csv").set_index("quantity")
    sigma2 = pd.read_csv(SRC / f"{key}_sigma2.csv").iloc[0]
    draws = pd.read_csv(SRC / f"{key}_draws.csv")
    hist = {}
    for c in draws.columns:
        if c == "sigma2":
            continue
        counts, edges = np.histogram(draws[c].to_numpy(), bins=40)
        hist[c] = {"edges": r(edges, 6), "counts": counts.tolist()}
    dump(f"{key}.json", {
        "bands": {col: by_series(bands, col) for col in
                  ["point_est", "pred_lo", "pred_hi", "obs_lo", "obs_hi"]},
        "params": params.to_dict(orient="records"),
        "fit": {q: {c: float(fit.loc[q, c]) for c in fit.columns} for q in fit.index},
        "sigma2": {k: float(v) for k, v in sigma2.items()},
        "nDraws": int(len(draws)),
        "hist": hist,
        "rt": rt_for(key),
    })

# Inputs of the runtime forward simulator (scenarios).
sp = pd.read_csv(SRC / "sim_params.csv")
psi = pd.read_csv(SRC / "sim_psi.csv")
sim_inputs = {}
for key in POINT:
    p = sp[sp["model"] == key].set_index("name")["value"]
    sim_inputs[key] = {
        "betas": [float(p[f"b{k}"]) for k in range(1, 6)],
        "theta": float(p["theta"]), "phi_h": float(p["phi_h"]), "phi_c": float(p["phi_c"]),
        "E0": float(p["E0"]), "I0": float(p["I0"]),
        "psi": [float(v) for v in psi[psi["model"] == key].sort_values("t")["psi"]],
    }
dump("sim_inputs.json", sim_inputs)

# Model 3b posterior draws for the scenario predictive band: 500 evenly
# spaced over the 5000 saved draws, plus the daily covariates of psi(t).
d = pd.read_csv(SRC / "age_bayes_draws.csv")
idx = np.linspace(0, len(d) - 1, N_SCENARIO_DRAWS).round().astype(int)
d = d.iloc[idx]
cov = pd.read_csv(SRC / "age_covariates.csv").sort_values("t")
dump("age_draws.json", {
    "draws": {c: [float(v) for v in d[c]] for c in d.columns},
    "NT": [float(v) for v in cov["NT"]],
    "mix": [[float(a), float(b), float(c_), float(e)] for a, b, c_, e in
            cov[["p_0_49", "p_50_59", "p_60_69", "p_70plus"]].to_numpy()],
})

# All-models comparison table (same rows/columns as the Streamlit download).
order = ([f"beta{k}" for k in range(1, 6)] + ["psi"] + [f"psi{k}" for k in range(1, 6)]
         + ["psi_base", "f_test", "theta", "phi_h", "r_c", "e0"])
cols = [("Model 1", "sim_m1"), ("Model 2", "sim_m2"), ("Model 3a", "age_nocorr"),
        ("Model 3b (least squares)", "age_point"),
        ("Model 3b (Bayesian, posterior mean)", "age_bayes")]


def canon(n):
    n = n.lower()
    return "beta" + n[1:] if n in {f"b{k}" for k in range(1, 6)} else n


table = {}
for label, key in cols:
    if key == "age_bayes":
        pr = pd.read_csv(SRC / "age_bayes_params.csv")
        est = dict(zip(pr["name"].map(canon), pr["post_mean"]))
        f = pd.read_csv(SRC / "age_bayes_fit.csv").set_index("quantity").loc["J_post_mean_params"]
        J = {k: float(f[k]) for k in SERIES + ["total"]}
    else:
        pr = pd.read_csv(SRC / f"{key}_params.csv")
        est = dict(zip(pr["name"].map(canon), pr["value"]))
        J = {k: float(summary.loc[key, f"J_{k}"]) for k in SERIES + ["total"]}
    vals = {p: (float(est[p]) if p in est else None) for p in order}
    table[label] = {"params": vals, "nEst": sum(v is not None for v in vals.values()), "J": J}
dump("comparison.json", {"order": order, "models": table})

# Model reference pages and PDF.
for f in list(SRC.glob("model_reference*")):
    shutil.copy(f, OUT / f.name)
print("done ->", OUT)
