# COVID-19 Model Simulator — Portugal

Interactive simulator of the SEIQHICURD model fitted to Portugal's 2020 epidemic
(EUVABECO project, CEMAT). Henrique Pacheco and Erida Gjini.

**Live:** https://henriquedcpacheco-sys.github.io/EUVABECO_simulator/

The app is a [marimo](https://marimo.io) notebook written in Python (`app.py`).
GitHub Pages hosts it as a static site: marimo exports it to WebAssembly, so the
same Python code runs in the visitor's browser (Pyodide), with no server. Every
push to `main` rebuilds and redeploys it (`.github/workflows/deploy.yml`).

## Run it locally

```bash
pip install -r requirements.txt
marimo run app.py          # the app, as on the website
marimo edit app.py         # the notebook, to change the code
```

## Modes

1. **Estimate early dynamics**: the fitted models (1, 2, 3a, 3b) over the whole
   of 2020. A run shows the least-squares fit; a toggle in the results then
   switches to the Bayesian MCMC posterior (credible and predictive bands).
2. **Estimate partial early dynamics**: the same, for a chosen period.
3. **Transmission scenarios**: an extra lockdown (Rₜ reduction on top of the
   fitted transmission, outside the two real lockdowns) or a later school
   reopening, simulated with the ODE model; save runs and compare them. For the
   Model 3b baseline, a toggle adds the 95% band from 500 posterior draws.
4. **Model with vaccination**: not available yet.

## How it works

- Nothing is fitted in the app. The fits come from the MATLAB pipeline (least
  squares) and the mcmcstat DRAM runs (Bayesian bands), exported to
  `public/data/*.json` by `scripts/build_data.py`.
- The scenarios integrate the ODE model in `simulate()` (classical Runge-Kutta 4,
  20 steps per day), the same equations as `d2_rhs.m`.
- Charts are Plotly figures built in Python; the controls are marimo's own
  widgets (`mo.ui.radio`, `mo.ui.slider`, `mo.ui.tabs`, ...).

## Files

| Path | What it holds |
|---|---|
| `app.py` | The whole app: constants, data loading, the ODE simulator, the Plotly charts, and one cell per part of the page |
| `custom.css` | Page styles (colours, layout, cards, tables) |
| `public/data/` | Precomputed results (JSON), model reference PNG/PDF |
| `scripts/build_data.py` | Rebuilds `public/data/` from the exports of the fitting code |

After new MATLAB/MCMC exports, refresh the data:

```bash
python scripts/build_data.py path/to/exports
```

Data: [DSSG Portugal](https://github.com/dssg-pt/covid19pt-data),
[Our World in Data](https://github.com/owid/covid-19-data). Model based on
[Dings et al. (2022)](https://doi.org/10.3390/v14102114).
