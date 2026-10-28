# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "plotly",
# ]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="full",
    app_title="COVID-19 Model Simulator — Portugal",
    css_file="custom.css",
)

with app.setup:
    # COVID-19 model simulator for Portugal 2020 (EUVABECO, CEMAT).
    # A marimo app: run it locally with `marimo run app.py`, or open the
    # GitHub Pages build, where the same Python runs in the browser (Pyodide).
    #
    # Reading order: constants -> data -> simulator -> charts -> HTML helpers
    # -> page cells (top bar, hero, modes, fit modes 1/2, scenarios, footer).
    import io
    import json
    import zipfile
    from datetime import date, datetime, timedelta, timezone
    from functools import cache
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    # ------------------------------------------------------------------
    # Files in public/ (data and icons). Locally they are read from disk;
    # on GitHub Pages they are fetched from the site.
    # ------------------------------------------------------------------
    def read_public(rel: str) -> str:
        loc = str(mo.notebook_location() / "public" / rel)
        if loc.startswith("http"):
            from pyodide.http import open_url

            return open_url(loc).read()
        return Path(loc).read_text(encoding="utf-8")

    def public_url(rel: str) -> str:
        """Link to a file in public/, for <img> and download links. Run
        locally, the file is inlined (marimo run does not serve public/)."""
        loc = str(mo.notebook_location() / "public" / rel)
        if loc.startswith("http"):
            return loc
        import base64
        import mimetypes

        kind = mimetypes.guess_type(loc)[0] or "application/octet-stream"
        return f"data:{kind};base64," + base64.b64encode(Path(loc).read_bytes()).decode()

    @cache
    def load(name: str):
        return json.loads(read_public(f"data/{name}.json"))

    ICONS = json.loads(read_public("icons.json"))

    # ------------------------------------------------------------------
    # Series, models, dates and texts
    # ------------------------------------------------------------------
    SERIES = ["cases", "ward", "icu", "deaths"]
    SERIES_LABELS = {"cases": "Daily confirmed cases", "ward": "Ward occupancy",
                     "icu": "ICU occupancy", "deaths": "Daily deaths"}
    SERIES_SHORT = {"cases": "Daily cases", "ward": "General ward", "icu": "ICU",
                    "deaths": "Daily deaths", "Rt": "Rₜ"}
    # The page is in dark mode. Series colours: Okabe-Ito based (readable for
    # colour-blind viewers), lightened to stand out on a dark background;
    # DEEP is the model line drawn over its own bands.
    COLORS = {"cases": "#56B4E9", "ward": "#F0B23A", "icu": "#D98FBA",
              "deaths": "#F07A3A", "Rt": "#2EC497"}
    DEEP = {"cases": "#9AD3F5", "ward": "#F7CF7A", "icu": "#EBB8D5",
            "deaths": "#F7A97E", "Rt": "#7BE0BF"}
    COMPARE_PALETTE = ["#56B4E9", "#F0B23A", "#2EC497", "#D98FBA", "#F07A3A",
                       "#A5B4FC", "#C4A3E0", "#AAB0BA"]
    # chart colours (dark mode)
    CT = dict(text="#e6eaf0", muted="#8b95a6", grid="#222c39", axis="#3a4658",
              rule="#7d889a", obs="#9aa3b2", window="#f07a55", border="#34425a", tooltip="#1b2430")
    FONT = "Inter Variable, system-ui, sans-serif"

    DAY0 = datetime(2020, 3, 1, tzinfo=timezone.utc)   # day 0 of the simulator
    N_DAYS = 306

    # NPI event, policy date, changepoint used, days after the policy
    NPI_EVENTS = [
        ("State of emergency (first national lockdown)", "2020-03-18", "2020-03-30", 12),
        ("Phased deconfinement begins", "2020-05-04", "2020-05-10", 6),
        ("Schools reopen", "2020-09-14", "2020-09-14", 0),
        ("Second state of emergency", "2020-11-04", "2020-11-06", 2),
    ]

    FIXED_PARAMS = [
        ("σ = 1/6 per day", "Incubation rate (latent period 6 days)"),
        ("γ_C = 1/2 per day", "Confirmation rate (2 days infectious before confirmation)"),
        ("γ_I = 1/5 per day", "Exit rate of the confirmed infectious stage"),
        ("α = 1/7 per day", "Exit rate of the non-hospitalised compartment"),
        ("γ_H = 1/9 per day", "Ward discharge rate"),
        ("γ_ICU = 1/20 per day", "ICU discharge rate"),
        ("φ_q = 0.002", "Fatality probability outside hospital"),
        ("N = 10 000 000", "Population"),
        ("I₀ = 1", "Initial infectious individuals"),
    ]
    ESTIMATED_PARAMS = [
        ("β (β₁…β₅)", "Transmission rate, one value per period between NPI dates"),
        ("ψ", "Probability that a confirmed case is hospitalised"),
        ("θ", "Probability that a ward patient is admitted to ICU"),
        ("φ_h", "Fatality probability in the ward"),
        ("r_c", "ICU/ward mortality ratio, so the ICU fatality is φ_c = r_c · φ_h"),
        ("E₀", "Initial exposed individuals"),
    ]

    MODEL_OPTIONS = {
        "sim_m1": ("Model 1 — constant", "Model 1", "One hospitalisation probability ψ for the whole year"),
        "sim_m2": ("Model 2 — piecewise ψ", "Model 2", "One value per NPI segment"),
        "age_nocorr": ("Model 3a — age-stratified ψ", "Model 3a", "Without testing-volume coupling"),
        "age_3b": ("Model 3b — age-stratified ψ", "Model 3b", "With testing-volume coupling"),
    }
    def model_label(name: str, caption: str) -> str:
        """A model choice: its name with its caption underneath (inline styles:
        the list is drawn by marimo, out of reach of custom.css)."""
        return (f'<div style="text-align:left;white-space:normal;line-height:1.3;padding:3px 2px;width:285px">'
                f'<div style="font-size:13.5px;font-weight:500">{name}</div>'
                f'<div style="font-size:12.5px;opacity:.65">{caption}</div></div>')

    # model picker label -> model key, as used by the fit modes
    MODEL_BY_LABEL = {model_label(v[0], v[2]): k for k, v in MODEL_OPTIONS.items()}
    POINT_KEY = {"sim_m1": "sim_m1", "sim_m2": "sim_m2", "age_nocorr": "age_nocorr", "age_3b": "age_point"}
    BAYES_KEY = {"sim_m1": "sim_m1_bayes", "sim_m2": "sim_m2_bayes",
                 "age_nocorr": "age_nocorr_bayes", "age_3b": "age_bayes"}
    MODEL_LABELS = {
        "sim_m1": "Model 1 — constant",
        "sim_m2": "Model 2 — piecewise ψ, one value per NPI segment",
        "age_nocorr": "Model 3a — age-stratified ψ, without testing-volume coupling",
        "age_point": "Model 3b — age-stratified ψ, with testing-volume coupling",
        "sim_m1_bayes": "Model 1 — constant · Bayesian posterior",
        "sim_m2_bayes": "Model 2 — piecewise ψ · Bayesian posterior",
        "age_nocorr_bayes": "Model 3a — age-stratified ψ, without coupling · Bayesian posterior",
        "age_bayes": "Model 3b — age-stratified ψ, with coupling · Bayesian posterior",
    }
    AGE_PARAM_NOTES = {
        "age_nocorr": "This variant adds one kind of parameter beyond Models 1/2: <strong>ψ_base</strong>, the "
        "hospitalisation probability of the 0-49 age group (the 50-59, 60-69 and 70+ groups use "
        "2x, 4x and 8x ψ_base, mixed each day by that day's case age composition). There is no "
        "testing-volume coupling here, so no F_test parameter.",
        "age_3b": "This variant adds two kinds of parameter beyond Models 1/2: <strong>ψ_base</strong>, the "
        "hospitalisation probability of the 0-49 age group (the 50-59, 60-69 and 70+ groups use "
        "2x, 4x and 8x ψ_base, mixed each day by that day's case age composition); and "
        "<strong>F_test</strong>, how fast the testing-volume coupling Ch(t) pulls ψ down as testing increases.",
    }

    MODE_COLORS = {"early": "#4ea8e3", "partial": "#34c3ad", "scenarios": "#f2865f", "vaccination": "#b591d6"}
    ICON_OF_MODE = {"early": "activity", "partial": "waypoints", "scenarios": "flask-conical", "vaccination": "syringe"}
    MODES = [
        dict(value="early", n=1, title="Estimate early dynamics", caption="Whole of 2020, fixed time period", icon=ICONS["activity"]),
        dict(value="partial", n=2, title="Estimate partial early dynamics", caption="Choose the time period", icon=ICONS["waypoints"]),
        dict(value="scenarios", n=3, title="Transmission scenarios", caption="Extra lockdown or later school reopening", icon=ICONS["flask-conical"]),
        dict(value="vaccination", n=4, title="Model with vaccination", caption="Not available yet", icon=ICONS["syringe"], soon=True),
    ]
    ZOOM_HINT = "Drag to move along the time axis; the toolbar at the top right of a plot zooms, resets, or saves it as PNG. Double-click to reset."
    # Plotly toolbar shown when hovering a chart
    PLOT_CONFIG = {"displaylogo": False, "scrollZoom": False, "doubleClick": "reset",
                   "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
                   "toImageButtonOptions": {"format": "png", "scale": 2}}

    # ------------------------------------------------------------------
    # The ODE model (same as Tese/D2/d2_rhs.m and the Streamlit simulator.py)
    # ------------------------------------------------------------------
    SIG, GC, GI, ALPHA, GH, GICU, PHIQ = 1 / 6, 1 / 2, 1 / 5, 1 / 7, 1 / 9, 1 / 20, 0.002
    N_POP = 1e7
    TOTINF = 1 / GC + 1 / GI          # = 7 days, so R0 = 7 * beta
    FITTED_CP = [29, 70, 197, 250]    # day of each fitted beta switch
    CPS = [0, *FITTED_CP, N_DAYS - 1]
    STEPS_PER_DAY = 20
    AGE_NTNORM = 12632                # Ch(t) = exp(-F_test * NT(t) / AGE_NTNORM)
    AGE_MULT = np.array([1.0, 2.0, 4.0, 8.0])
    N_BAND_DRAWS = 500

    # Scenarios: the two real lockdowns stay as fitted, so an extra lockdown
    # goes before the first or between the two. Schools reopened on day 197;
    # that switch can move later, but not past the Nov 2020 tightening.
    FREE_ZONES = [("Before the spring lockdown", 0, FITTED_CP[0]),
                  ("Between the two lockdowns", FITTED_CP[1], FITTED_CP[3])]
    SCHOOL_CP, NEXT_CP = FITTED_CP[2], FITTED_CP[3]
    MAX_SCHOOL_DELAY = NEXT_CP - SCHOOL_CP - 1


# =========================================================================
# Small helpers: dates and number formats
# =========================================================================
@app.function
def day_to_time(d: int) -> int:
    """Day index -> milliseconds since 1970 (UTC)."""
    return int((DAY0 + timedelta(days=int(d))).timestamp() * 1000)


@app.function
def iso(t_ms: int) -> str:
    return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


@app.function
def fmt_date(t_ms: int) -> str:
    return datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%d/%m/%Y")


@app.function
def iso_to_day(s: str) -> int:
    return (datetime.fromisoformat(s).replace(tzinfo=timezone.utc) - DAY0).days


@app.function
def fmt_num(v, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.{digits}f}"


@app.function
def fmt_int(v) -> str:
    return f"{round(v):,}"


@app.function
def fmt_sig(v, sig: int = 4) -> str:
    """Parameter values with 4 significant digits."""
    if v is None:
        return "—"
    if v == 0:
        return "0"
    if abs(v) >= 1000:
        return fmt_int(v)
    return f"{float(f'{v:.{sig}g}'):g}"


@app.function
def param_symbol(name: str) -> str:
    n = name.lower()
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    if n in {f"b{k}" for k in range(1, 6)} or n in {f"beta{k}" for k in range(1, 6)}:
        return "β" + n[-1].translate(sub)
    if n == "psi0":
        return "ψ₀"
    if n.startswith("psi") and n[3:].isdigit():
        return "ψ" + n[3:].translate(sub)
    return {"psi": "ψ", "psi_base": "ψ_base", "theta": "θ", "phi_h": "φ_h", "r_c": "r_c",
            "f_test": "F_test", "e0": "E₀", "i0": "I₀"}.get(n, name)


@app.function
def param_meaning(name: str) -> str:
    n = name.lower()
    if n in {f"b{k}" for k in range(1, 6)} or n in {f"beta{k}" for k in range(1, 6)}:
        return f"Transmission rate, period {n[-1]}"
    if n == "psi0":
        return "Baseline hospitalisation probability"
    if n.startswith("psi") and n[3:].isdigit():
        return f"Hospitalisation probability, period {n[3:]}"
    return {
        "psi": "Hospitalisation probability",
        "psi_base": "Hospitalisation probability of the 0-49 age group (other groups: x2, x4, x8)",
        "theta": "ICU admission probability (from the ward)",
        "phi_h": "Ward fatality probability",
        "r_c": "ICU/ward mortality ratio (φ_c = r_c · φ_h)",
        "f_test": "Testing decay: how fast the hospitalisation probability falls as testing grows",
        "e0": "Initial exposed individuals",
        "i0": "Initial infectious individuals",
    }.get(n, "")


@app.function
def to_csv(header: list, rows: list) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v)
        return '"' + s.replace('"', '""') + '"' if any(c in s for c in '",\n') else s

    return "\n".join(",".join(cell(v) for v in r) for r in [header, *rows]) + "\n"


# =========================================================================
# Simulator: integrate the model with classical Runge-Kutta 4 on a fixed
# step (20 per day), one constant beta per segment of the schedule.
# =========================================================================
@app.function
def fitted_schedule(betas) -> list:
    """[(start_day, end_day, beta), ...] at the fitted changepoints."""
    return [(CPS[k], CPS[k + 1], betas[k]) for k in range(5)]


@app.function
def lockdown_schedule(betas, start: int, duration: int, reduction: float) -> list:
    """Extra Rt reduction on top of the fitted transmission: from `start`, for
    `duration` days, beta(t) is multiplied by (1 - reduction)."""
    end = min(start + duration, N_DAYS - 1)
    bounds = sorted(set(CPS + [start, end]))
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        k = max(j for j in range(5) if CPS[j] <= a)
        out.append((a, b, betas[k] * ((1 - reduction) if start <= a < end else 1.0)))
    return out


@app.function
def school_schedule(betas, delay: int) -> list:
    """Later school reopening: the 14/09/2020 switch moves `delay` days later."""
    c = list(CPS)
    c[3] = min(c[3] + delay, c[4] - 1)
    return [(c[k], c[k + 1], betas[k]) for k in range(5)]


@app.function
def simulate(inp: dict, schedule: list, n: int = 1) -> dict:
    """Integrate days 0..305. With n > 1 every input (betas, theta, ..., psi)
    holds one value per parameter set, and all n sets run at once.
    Returns daily cases, ward, icu, deaths and Rt, each of shape (N_DAYS, n)."""
    theta, phi_h, phi_c = (np.broadcast_to(np.asarray(inp[k], float), (n,)) for k in ("theta", "phi_h", "phi_c"))
    psi = np.asarray(inp["psi"], float).reshape(N_DAYS, -1)
    f0 = (1 / GC) / (1 / GC + 1 / GI)
    E0 = np.broadcast_to(np.asarray(inp["E0"], float), (n,))
    I0 = np.broadcast_to(np.asarray(inp["I0"], float), (n,))
    y = np.zeros((10, n))
    y[0], y[1], y[2], y[3] = N_POP - E0 - I0, E0, I0 * f0, I0 * (1 - f0)
    out = np.zeros((N_DAYS, 10, n))
    out[0] = y
    beta_day = np.zeros((N_DAYS, n))
    hstep = 1 / STEPS_PER_DAY

    def rhs(y, beta, p):
        S, E, Iuc, Ic, Ti, H, ICU, R, D, C = y
        lam = beta * (Iuc + Ic) / (S + E + Iuc + Ic + Ti + H + ICU + R + D)
        return np.array([
            -lam * S,
            lam * S - SIG * E,
            SIG * E - GC * Iuc,
            GC * Iuc - GI * Ic,
            (1 - p) * GI * Ic - ALPHA * Ti,
            p * GI * Ic - GH * H,
            GH * theta * H - GICU * ICU,
            (1 - PHIQ) * ALPHA * Ti + (1 - phi_h) * GH * (1 - theta) * H + (1 - phi_c) * GICU * ICU,
            PHIQ * ALPHA * Ti + phi_h * GH * (1 - theta) * H + phi_c * GICU * ICU,
            GC * Iuc,
        ])

    for a, b, beta in schedule:
        beta = np.broadcast_to(np.asarray(beta, float), (n,))
        for day in range(a, b):
            p = psi[min(max(day, 0), N_DAYS - 1)]   # psi of the day being integrated
            for _ in range(STEPS_PER_DAY):
                k1 = rhs(y, beta, p)
                k2 = rhs(y + 0.5 * hstep * k1, beta, p)
                k3 = rhs(y + 0.5 * hstep * k2, beta, p)
                k4 = rhs(y + hstep * k3, beta, p)
                y = y + hstep / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out[day + 1] = y
        beta_day[a:b + 1] = beta
    C, D = out[:, 9], out[:, 8]
    return {
        "cases": np.maximum(np.diff(C, axis=0, prepend=0.0), 0),
        "ward": out[:, 5],
        "icu": out[:, 6],
        "deaths": np.maximum(np.diff(D, axis=0, prepend=0.0), 0),
        "Rt": beta_day * TOTINF * out[:, 0] / N_POP,
    }


@app.function
@cache
def run_scenario(model: str, scn: tuple | None) -> dict:
    """One scenario run, as plain lists. scn: None (baseline),
    ("lockdown", start, duration, reduction %) or ("school", delay)."""
    inp = load("sim_inputs")[model]
    if scn is None:
        sched = fitted_schedule(inp["betas"])
    elif scn[0] == "lockdown":
        sched = lockdown_schedule(inp["betas"], scn[1], scn[2], scn[3] / 100)
    else:
        sched = school_schedule(inp["betas"], scn[1])
    tr = simulate(inp, sched)
    return {k: v[:, 0].tolist() for k, v in tr.items()}


@app.function
@cache
def baseline_bands() -> dict:
    """95% posterior-predictive band of the fitted Model 3b baseline: 500
    posterior draws are integrated together, each with its own psi(t)
    (rebuilt from the age mix and the testing volume), then multiplied
    each day by exp(eps), eps ~ N(0, sigma2 of that draw)."""
    ad = load("age_draws")
    d = {k: np.asarray(v, float) for k, v in ad["draws"].items()}
    m_t = np.asarray(ad["mix"]) @ AGE_MULT                          # (N_DAYS,)
    nt = np.asarray(ad["NT"])
    psi = np.exp(-np.outer(nt, d["F_test"]) / AGE_NTNORM) * d["psi_base"] * m_t[:, None]
    betas = [d[f"beta{k}"] for k in range(1, 6)]
    n = len(d["theta"])
    inp = dict(theta=d["theta"], phi_h=d["phi_h"], phi_c=d["r_c"] * d["phi_h"], E0=d["E0"], I0=1.0, psi=psi)
    tr = simulate(inp, fitted_schedule(betas), n=n)
    rng = np.random.default_rng(20231024)
    sd = np.sqrt(d["sigma2"])
    bands = {}
    for s in SERIES:
        noisy = np.exp(np.log(np.maximum(tr[s], 0) + 1) + rng.standard_normal(tr[s].shape) * sd) - 1
        lo, hi = np.quantile(noisy, [0.025, 0.975], axis=1)
        bands[s] = {"lo": lo.tolist(), "hi": hi.tolist()}
    return bands


# =========================================================================
# Charts (Plotly). Every time chart looks the same: light grid, the fixed
# NPI changepoints as dashed lines, a tooltip with the values of the day.
# =========================================================================
@app.function
def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


@app.function
def base_layout(title: str | None = None, top: int | None = None) -> dict:
    return dict(
        font=dict(family=FONT, size=11, color=CT["muted"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=14, t=top if top is not None else (38 if title else 16), b=6),
        title=dict(text=title, x=0, xref="paper", xanchor="left", y=0.995, yref="container", yanchor="top",
                   font=dict(size=13, color=CT["text"], weight=600)) if title else None,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=CT["tooltip"], bordercolor=CT["border"], font=dict(family=FONT, size=12, color=CT["text"])),
        dragmode="pan",
        modebar=dict(bgcolor="rgba(19,26,35,0.85)", color=CT["muted"], activecolor=CT["text"]),
    )


@app.function
def time_axis(start: int, end: int) -> dict:
    return dict(type="date", range=[iso(start), iso(end)], showgrid=False, zeroline=False,
                linecolor=CT["axis"], ticks="", tickformat="%b", dtick="M1", hoverformat="%d/%m/%Y",
                tickfont=dict(size=11, color=CT["muted"]), automargin=True)


@app.function
def value_axis(digits: int | None = None, title: str | None = None) -> dict:
    return dict(showgrid=True, gridcolor=CT["grid"], zeroline=False, showline=False, ticks="",
                tickformat=f".{digits}f" if digits is not None else "~s", nticks=5, rangemode="tozero",
                tickfont=dict(size=11, color=CT["muted"]), automargin=True, fixedrange=True,
                title=dict(text=title, font=dict(size=11, color=CT["muted"])) if title else None)


@app.function
def marks(start: int, end: int, window: tuple | None = None, rt_line: bool = False, changepoints: bool = True) -> list:
    """Dashed changepoint lines, the Rt = 1 line and the scenario window."""
    shapes = []
    if changepoints:
        for ev in NPI_EVENTS:
            t = day_to_time(iso_to_day(ev[2]))
            if start <= t <= end:
                shapes.append(dict(type="line", xref="x", yref="paper", x0=ev[2], x1=ev[2], y0=0, y1=1,
                                   line=dict(color=CT["rule"], width=1.1, dash="5px,4px")))
    if rt_line:
        shapes.append(dict(type="line", xref="paper", yref="y", x0=0, x1=1, y0=1, y1=1,
                           line=dict(color=CT["rule"], width=1, dash="2px,3px")))
    if window:
        a, b = max(window[0], start), min(window[1], end)
        if a < b:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=iso(a), x1=iso(b), y0=0, y1=1,
                               fillcolor=CT["window"], opacity=0.12, line=dict(width=0), layer="below"))
    return shapes


@app.function
def time_panel(times: list, layers: list, start: int, end: int, title: str | None = None,
               window=None, rt_line=False, y_digits=None, y_title=None) -> go.Figure:
    """Layers: ("dots", name, values, color), ("line", name, values, color, width),
    ("band", name, lo, hi, color, opacity)."""
    idx = [i for i, t in enumerate(times) if start <= t <= end]
    x = [iso(times[i]) for i in idx]
    pick = lambda v: [v[i] for i in idx]
    hover_digits = y_digits if y_digits is not None else 0
    fig = go.Figure()
    for L in layers:
        kind, name = L[0], L[1]
        if kind == "dots":
            fig.add_scatter(x=x, y=pick(L[2]), mode="markers", name=name,
                            marker=dict(size=4.5, color=L[3], opacity=0.75),
                            hovertemplate=f"%{{y:,.{hover_digits}f}}")
        elif kind == "line":
            fig.add_scatter(x=x, y=pick(L[2]), mode="lines", name=name,
                            line=dict(color=L[3], width=L[4] if len(L) > 4 else 2.2),
                            hovertemplate=f"%{{y:,.{1 if y_digits is None else 3}f}}")
        else:  # band: the low edge, then the high edge filled down to it
            lo = [max(v, 0) for v in pick(L[2])]
            hi = [max(v, 0) for v in pick(L[3])]
            fig.add_scatter(x=x, y=lo, mode="lines", name=f"{name}, low", line=dict(width=0),
                            hovertemplate="%{y:,.1f}")
            fig.add_scatter(x=x, y=hi, mode="lines", name=f"{name}, high", line=dict(width=0),
                            fill="tonexty", fillcolor=rgba(L[4], L[5]), hovertemplate="%{y:,.1f}")
    fig.update_layout(**base_layout(title), xaxis=time_axis(start, end),
                      yaxis=value_axis(y_digits, y_title),
                      shapes=marks(start, end, window, rt_line))
    return fig


@app.function
def overlay_chart(start: int, end: int, left: dict, right: dict, window=None) -> dict:
    """Two series on one time axis, each with its own y-axis."""
    fig = go.Figure()
    for side, yaxis in ((left, "y"), (right, "y2")):
        idx = [i for i, t in enumerate(side["times"]) if start <= t <= end]
        fig.add_scatter(x=[iso(side["times"][i]) for i in idx], y=[side["values"][i] for i in idx],
                        mode="lines", name=side["name"], yaxis=yaxis,
                        line=dict(color=side["color"], width=2.3),
                        hovertemplate=f"%{{y:,.{3 if side['is_rt'] else 1}f}}")

    def axis(s, pos):
        ax = value_axis(1 if s["is_rt"] else None)
        ax.update(tickfont=dict(size=11, color=s["color"]), side=pos, showgrid=pos == "left")
        if pos == "right":
            ax.update(overlaying="y")
        return ax

    shapes = marks(start, end, window)
    for s, yref in ((left, "y"), (right, "y2")):
        if s["is_rt"]:
            shapes.append(dict(type="line", xref="paper", yref=yref, x0=0, x1=1, y0=1, y1=1,
                               line=dict(color=s["color"], width=1, dash="2px,3px")))
    fig.update_layout(**base_layout(top=62), xaxis=time_axis(start, end),
                      yaxis=axis(left, "left"), yaxis2=axis(right, "right"), shapes=shapes,
                      annotations=[
                          dict(text=f"<b>{left['name']}</b>", xref="paper", yref="paper", x=0, y=1.03,
                               xanchor="left", yanchor="bottom", showarrow=False, font=dict(size=12, color=left["color"])),
                          dict(text=f"<b>{right['name']}</b>", xref="paper", yref="paper", x=1, y=1.03,
                               xanchor="right", yanchor="bottom", showarrow=False, font=dict(size=12, color=right["color"])),
                      ])
    return fig


@app.function
def residual_chart(weeks: list, stacks: list, start: int, end: int) -> dict:
    """Stacked weekly squared log-residuals, one bar per week."""
    fig = go.Figure()
    week_ms = 7 * 86_400_000
    for s in stacks:
        fig.add_bar(x=[iso(w + week_ms // 2) for w in weeks], y=s["values"], name=s["name"],
                    marker=dict(color=s["color"], line=dict(width=0)), width=week_ms * 0.86,
                    hovertemplate="%{y:.2f}")
    layout = base_layout(top=36)
    layout.update(showlegend=True, barmode="stack",
                  legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom",
                              font=dict(size=12, color=CT["text"])))
    fig.update_layout(**layout, xaxis=time_axis(start, end), yaxis=value_axis(0), shapes=marks(start, end))
    fig.update_xaxes(hoverformat="Week of %d/%m/%Y")
    return fig


@app.function
def histogram_chart(title: str, edges: list, counts: list, mean: float, ls: float, lo: float, hi: float) -> dict:
    """Posterior histogram with its 95% CI, posterior mean and least-squares estimate."""
    centers = [(a + b) / 2 for a, b in zip(edges[:-1], edges[1:])]
    width = (edges[-1] - edges[0]) / max(len(counts), 1)
    fig = go.Figure()
    fig.add_bar(x=centers, y=counts, width=width * 0.97, marker=dict(color=CT["obs"], opacity=0.8, line=dict(width=0)),
                customdata=[[a, b] for a, b in zip(edges[:-1], edges[1:])],
                hovertemplate="%{customdata[0]:.4g} – %{customdata[1]:.4g}<br><b>%{y}</b> draws<extra></extra>")
    xmin, xmax = min(edges[0], ls, lo), max(edges[-1], ls, hi)
    layout = base_layout(title, top=32)
    layout.update(hovermode="closest", margin=dict(l=6, r=10, t=32, b=4))
    fig.update_layout(
        **layout,
        xaxis=dict(range=[xmin, xmax], showgrid=False, zeroline=False, linecolor=CT["axis"], ticks="",
                   tickformat="~s" if abs(xmax) >= 1000 else "~r", nticks=4, tickfont=dict(size=10, color=CT["muted"]), automargin=True),
        yaxis=dict(visible=False, fixedrange=True),
        shapes=[
            dict(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1, fillcolor=CT["window"], opacity=0.1,
                 line=dict(width=0), layer="below"),
            dict(type="line", xref="x", yref="paper", x0=mean, x1=mean, y0=0, y1=1, line=dict(color=CT["window"], width=2)),
            dict(type="line", xref="x", yref="paper", x0=ls, x1=ls, y0=0, y1=1,
                 line=dict(color=CT["text"], width=1.4, dash="4px,3px"), opacity=0.7),
        ],
    )
    return fig


@app.function
def compare_chart(items: list, y_title: str, is_rt: bool) -> dict:
    """One line per saved simulation."""
    start = min(s["times"][0] for s in items)
    end = max(s["times"][-1] for s in items)
    fig = go.Figure()
    for k, s in enumerate(items):
        color = COMPARE_PALETTE[k % len(COMPARE_PALETTE)]
        fig.add_scatter(x=[iso(t) for t in s["times"]], y=s["values"], mode="lines", name=s["name"],
                        line=dict(color=color, width=2.2), hovertemplate=f"%{{y:,.{3 if is_rt else 1}f}}")
    layout = base_layout(top=24)
    layout.update(showlegend=True, margin=dict(l=8, r=14, t=24, b=14 + 20 * len(items)),
                  legend=dict(orientation="v", x=0, y=-0.08, xanchor="left", yanchor="top",
                              font=dict(size=12, color=CT["text"])))
    fig.update_layout(**layout, xaxis=time_axis(start, end), yaxis=value_axis(1 if is_rt else None, y_title),
                      shapes=marks(start, end, rt_line=is_rt))
    return fig


@app.function
def chart(fig, height: int = 260) -> str:
    """HTML of an interactive Plotly chart, to place inside the page HTML."""
    fig = go.Figure(fig).update_layout(height=height)
    return mo.ui.plotly(fig, config=PLOT_CONFIG).text


# =========================================================================
# HTML helpers: the cards, tables and legends of the page
# =========================================================================
@app.function
def icon(name: str, size: int = 16, cls: str = "icon") -> str:
    return (f'<svg class="{cls}" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')


@app.function
def card(body: str, title: str = "", subtitle: str = "", ic: str | None = None, accent: bool = False, pad: bool = True) -> str:
    head = ""
    if title:
        badge = f'<span class="ic">{icon(ic)}</span>' if ic else ""
        sub = f"<p>{subtitle}</p>" if subtitle else ""
        head = f'<header class="card-head {"accent" if accent else ""}">{badge}<div><h3>{title}</h3>{sub}</div></header>'
    bar = '<div class="accent-bar"></div>' if accent else ""
    return f'<section class="card">{bar}{head}<div class="{"card-body" if pad else ""}">{body}</div></section>'


@app.function
def field(content: str, label: str = "", hint: str = "") -> str:
    lab = f'<div class="field-label">{label}</div>' if label else ""
    hin = f'<div class="field-hint">{hint}</div>' if hint else ""
    return f"<div>{lab}{content}{hin}</div>"


@app.function
def table(columns: list, rows: list, dense: bool = False, highlight=None) -> str:
    """columns: (key, label, css class); rows: dicts of already formatted HTML."""
    head = "".join(f'<th class="{c[2] if len(c) > 2 else ""}">{c[1]}</th>' for c in columns)
    body = ""
    for r in rows:
        hl = ' class="hl"' if highlight and highlight(r) else ""
        body += f"<tr{hl}>" + "".join(f'<td class="{c[2] if len(c) > 2 else ""}">{r.get(c[0], "—")}</td>' for c in columns) + "</tr>"
    return f'<div class="table-wrap"><table class="data {"dense" if dense else ""}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


@app.function
def legend(items: list) -> str:
    sw = {
        "dot": lambda c: f'<span class="sw-dot" style="background:{c}"></span>',
        "line": lambda c: f'<span class="sw-line" style="border-color:{c}"></span>',
        "dash": lambda c: f'<span class="sw-dash" style="border-color:{c}"></span>',
        "band": lambda c: f'<span class="sw-band" style="background:{c};opacity:.45"></span>',
        "band-light": lambda c: f'<span class="sw-band" style="background:{c};opacity:.2"></span>',
        "window": lambda c: f'<span class="sw-band" style="background:{c};opacity:.2"></span>',
        "rule": lambda c: f'<span class="sw-rule" style="border-color:{c}"></span>',
    }
    return '<div class="legend">' + "".join(f'<span class="it">{sw[k](c)}{label}</span>' for k, c, label in items) + "</div>"


@app.function
def empty_state(ic: str, title: str, text: str) -> str:
    return f'<div class="empty"><div class="ic">{icon(ic, 24)}</div><h3>{title}</h3><div class="t">{text}</div></div>'


@app.function
def callout(ic: str, html: str) -> str:
    return f'<div class="callout"><span class="ic">{icon(ic)}</span><div>{html}</div></div>'


@app.function
def chart_card(chart_el, color: str | None = None) -> str:
    return f'<div class="chart-card"><div class="top" style="background:{color or "var(--eu-primary)"}"></div>{chart_el}</div>'


@app.function
def panel_grid(cards: list, cols: int = 2) -> str:
    return (f'<div class="stack" style="--g:0"><div class="{"grid4" if cols == 4 else "grid2"}">{"".join(cards)}</div>'
            f'<p class="small" style="margin-top:8px">{ZOOM_HINT}</p></div>')


@app.function
def metric(label: str, value: str, color: str = "var(--eu-primary)", ic: str | None = None, sub: str = "") -> str:
    badge = (f'<span class="ic" style="background:color-mix(in oklab, {color} 18%, transparent);color:{color}">{icon(ic, 14)}</span>'
             if ic else "")
    s = f'<div class="sub">{sub}</div>' if sub else ""
    return (f'<div class="metric" style="background-image:linear-gradient(135deg, color-mix(in oklab, {color} 12%, transparent), transparent 60%)">'
            f'<div class="side" style="background:{color}"></div><div class="lab">{badge}{label}</div>'
            f'<div class="val">{value}</div>{s}</div>')


@app.function
def page(mode: str, html: str, place: str = "main") -> mo.Html:
    """Wrap a cell's output: `place` puts it in the left column ("side"), the
    right column ("main") or across the page ("wide"); `mode` sets the accent."""
    return mo.Html(f'<div class="{place}-anchor"></div><div data-mode="{mode}">{html}</div>')


# =========================================================================
# Epidemic metrics and the overlay picker, shared by the fit and scenario modes
# =========================================================================
@app.function
def compute_metrics(times: list, values: dict, rt_times: list, rt: list, start: int, end: int) -> dict:
    idx = [i for i, t in enumerate(times) if start <= t <= end]
    peaks = {}
    for s in SERIES:
        best = max(idx, key=lambda i: values[s][i])
        peaks[s] = (values[s][best], times[best])
    rt_idx = [i for i, t in enumerate(rt_times) if start <= t <= end]
    k = max(rt_idx, key=lambda i: rt[i])
    return dict(days=len(idx), cases=sum(values["cases"][i] for i in idx),
                deaths=sum(values["deaths"][i] for i in idx), rt_max=rt[k], rt_day=rt_times[k], peaks=peaks)


@app.function
def metrics_csv_rows(m: dict) -> list:
    return [["Days simulated", m["days"]], ["Total confirmed cases", round(m["cases"])],
            ["Total deaths", round(m["deaths"])], [f"Highest Rt ({fmt_date(m['rt_day'])})", round(m["rt_max"], 3)],
            *[[f"Peak {SERIES_LABELS[s]} ({fmt_date(m['peaks'][s][1])})", round(m["peaks"][s][0])] for s in SERIES]]


@app.function
def metrics_view(m: dict, intro: str = "") -> str:
    cards = "".join([
        metric("Days simulated", str(m["days"]), ic="calendar-days"),
        metric("Total confirmed cases", fmt_int(m["cases"]), COLORS["cases"], "users"),
        metric("Total deaths", fmt_int(m["deaths"]), COLORS["deaths"], "heart-pulse"),
        metric("Highest Rₜ", fmt_num(m["rt_max"], 2), COLORS["Rt"], "trending-up", f"on {fmt_date(m['rt_day'])}"),
    ])
    rows = [dict(series=f'<span style="display:inline-flex;align-items:center;gap:8px;font-weight:500">'
                        f'<span class="sw-dot" style="background:{COLORS[s]}"></span>{SERIES_LABELS[s]}</span>',
                 day=fmt_date(m["peaks"][s][1]), value=fmt_int(m["peaks"][s][0])) for s in SERIES]
    intro_html = f'<p class="lede" style="font-size:14px">{intro}</p>' if intro else ""
    return (f'<div class="stack">{intro_html}<div class="grid-metrics">{cards}</div>'
            f'<div><h4 class="sec">Peak of each series</h4>'
            f'{table([("series", "Series"), ("day", "Peak day", "r"), ("value", "Peak value", "r")], rows)}</div></div>')



# =========================================================================
# Overlay picker and downloads, shared by the fit and scenario modes
# =========================================================================
@app.function
def overlay_widgets():
    """The overlay picker: a preset, or any two series ("Custom")."""
    presets = ["Rₜ vs Daily cases", "Daily cases vs General ward", "Daily cases vs Daily deaths", "Custom"]
    names = ["Rₜ", "Daily cases", "General ward", "ICU", "Daily deaths"]
    preset = mo.ui.radio(presets, value=presets[0], inline=True)
    left = mo.ui.dropdown(names, value="Daily cases", label="Left y-axis")
    right = mo.ui.dropdown(names, value="Daily deaths", label="Right y-axis")
    return preset, left, right


@app.function
def overlay_pair(preset, left, right) -> tuple:
    if preset.value == "Custom":
        return left.value, right.value
    a, b = preset.value.split(" vs ")
    return a, b


@app.function
def overlay_picker_html(preset, left, right) -> str:
    custom = ""
    if preset.value == "Custom":
        custom = f'<div class="grid2" style="max-width:36rem;margin-top:12px"><div>{left}</div><div>{right}</div></div>'
    return f"<div>{preset}{custom}</div>"


@app.function
def figures_html(title: str, figures: list) -> str:
    """A stand-alone HTML page with interactive charts, for the downloads."""
    parts = [go.Figure(f).update_layout(height=360).to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
             for i, f in enumerate(figures)]
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{title}</title></head>'
            f'<body style="font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;background:#0c1118;color:#e6eaf0"><h2>{title}</h2>'
            + "".join(parts) + "</body></html>")


@app.function
def zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


# =========================================================================
# Page: top bar, hero, mode selector
# =========================================================================
@app.cell(hide_code=True)
def _():
    # Top bar. "About" and "Model scheme" are <details> elements: a click
    # opens the dialog, a click outside it closes it (HTML and CSS only).
    _about = (
        "<p>We provide a tool in which it is possible to explore how COVID-19 spreads in a particular setting, and how "
        "different factors influence the course of an outbreak, through our interactive online simulator. We have "
        "adjusted the model to epidemiological data from Portugal (<a href='https://github.com/dssg-pt/covid19pt-data'>DSSG Portugal</a>; "
        "<a href='https://github.com/owid/covid-19-data'>Our World in Data</a>), especially the early phase of the "
        "SARS-CoV-2 pandemic. We allow several model versions, based on "
        "<a href='https://doi.org/10.3390/v14102114'>Dings et al. (2022)</a>, of increasing complexity to be employed, "
        "where several parameters such as transmission rates, hospitalisation probabilities and others are estimated "
        "to adjust model predictions to the observed time series.</p>"
        "<p>In addition to model calibration based on early dynamics, it is possible to investigate hypothetical "
        "counterfactual scenarios and observe the impact of different parameters on infection patterns over time. "
        "We also implement models with vaccination for later stages of the pandemic. Although we only model "
        "aggregated nation-level data for Portugal, the platform can be flexibly extended to model specific regional "
        "data of the same type and resolution.</p>"
    )
    _pdf = public_url("data/model_reference.pdf")
    _scheme = "".join(
        f'<img src="{public_url(f"data/model_reference-{i}.png")}" alt="Model reference, page {i}" loading="lazy">'
        for i in (1, 2)
    ) + (f'<a href="{_pdf}" download="model_reference.pdf" style="display:inline-flex;align-items:center;gap:8px;'
         f'font-size:13.5px;font-weight:500">{icon("file-down")}Download the equations (PDF)</a>')

    def _dialog(button, title, body, wide=False):
        return (f'<details class="dlg"><summary class="top-btn">{button}</summary>'
                f'<div class="modal-back"><div class="modal {"wide" if wide else ""}" role="dialog" aria-label="{title}">'
                f'<div class="modal-head"><h2>{title}</h2><span class="close-hint">Click outside to close</span></div>'
                f'<div class="modal-body">{body}</div></div></div></details>')

    mo.Html(f"""<div class="topbar-anchor"></div>
    <header class="topbar"><div class="hero-line"></div><div class="topbar-in">
      <img src="{public_url('favicon.svg')}" alt="" class="logo">
      <div class="brand"><div class="name">COVID-19 Model Simulator</div><div class="sub">EUVABECO · CEMAT · Portugal 2020</div></div>
      <nav>
        {_dialog(f'{icon("info")}<span>About</span>', "Welcome to the Online COVID-19 Simulator", _about)}
        {_dialog(f'{icon("sigma")}<span>Model scheme</span>', "Model scheme and equations", _scheme, wide=True)}
        <a class="top-btn pdf" href="{_pdf}" download="model_reference.pdf">{icon("file-down")}<span>Equations (PDF)</span></a>
      </nav></div></header>""")
    return


@app.cell(hide_code=True)
def _():
    # Hero, with decorative epidemic curves in the series colours.
    _curves = """
    <svg class="curves" aria-hidden="true" viewBox="0 0 800 260" preserveAspectRatio="none">
      <path d="M0 230 C 120 230 150 150 200 150 S 260 225 330 225 S 470 225 540 160 S 640 40 690 60 S 760 140 800 150" fill="none" stroke="#56B4E9" stroke-width="3" stroke-linecap="round" opacity=".9"/>
      <path d="M0 240 C 140 240 170 185 220 185 S 290 235 360 235 S 500 232 570 190 S 660 95 710 100 S 770 160 800 165" fill="none" stroke="#F0B23A" stroke-width="2.5" stroke-linecap="round" opacity=".8"/>
      <path d="M0 248 C 150 248 185 205 235 205 S 300 244 380 244 S 520 240 590 215 S 680 140 730 140 S 780 185 800 190" fill="none" stroke="#D98FBA" stroke-width="2.2" stroke-linecap="round" opacity=".75"/>
      <path d="M0 254 C 160 254 200 228 250 228 S 320 252 400 252 S 540 250 610 236 S 700 190 745 186 S 785 210 800 214" fill="none" stroke="#5be3c4" stroke-width="2" stroke-linecap="round" opacity=".7"/>
      """ + "".join(f'<line x1="{x}" x2="{x}" y1="20" y2="260" stroke="white" stroke-opacity=".18" stroke-dasharray="5 6"/>'
                    for x in (200, 330, 540, 690)) + "</svg>"
    mo.Html(f"""
    <section class="hero">{_curves}
      <div class="in">
        <div class="pill"><i></i>Online COVID-19 simulator</div>
        <h1>COVID-19 Model Simulator — Portugal</h1>
        <p class="lead">Explore how a compartmental model fitted to Portugal's 2020 epidemic reproduces confirmed cases,
        ward and ICU occupancy and deaths, compare model versions of increasing complexity, and test
        counterfactual extra lockdowns and later school reopenings.</p>
        <p class="by"><b>Henrique Pacheco</b>, CEMAT · <b>Erida Gjini</b>, CEMAT</p>
      </div>
    </section>""")
    return


@app.cell(hide_code=True)
def _():
    # Mode selector, overlapping the bottom of the hero.
    # Each tab label is a small card: the mode's icon on its own colour.
    # (Inline styles: the tab bar is drawn by marimo, out of reach of custom.css.)
    def _mode_label(m):
        color = MODE_COLORS[m["value"]]
        soon = " · soon" if m.get("soon") else ""
        return (f'<div title="{m["caption"]}" style="display:flex;align-items:center;gap:10px;text-align:left;padding:2px 6px;'
                f'width:calc((min(100vw, 1440px) - 150px) / 4 / 1.15)">'
                f'<span style="display:grid;place-items:center;width:30px;height:30px;flex-shrink:0;border-radius:8px;'
                f'background:{color};color:#06121d">{icon(ICON_OF_MODE[m["value"]], 17)}</span>'
                f'<span style="line-height:1.15"><span style="display:block;font-size:11px;font-weight:600;color:{color}">'
                f'Mode {m["n"]}{soon}</span><span style="display:block;font-size:13.5px;font-weight:600">{m["title"]}</span></span></div>')

    MODE_TABS = {_mode_label(m): m["value"] for m in MODES}
    mode_picker = mo.ui.tabs({k: "" for k in MODE_TABS})
    return MODE_TABS, mode_picker


@app.cell(hide_code=True)
def _(mode_picker):
    mo.Html(f'<div class="modes-wrap">{mode_picker}</div>')
    return


@app.cell(hide_code=True)
def _(MODE_TABS, mode_picker):
    mode = MODE_TABS.get(mode_picker.value, "early")
    return (mode,)


# =========================================================================
# Modes 1 and 2: fitted models
# =========================================================================
@app.cell(hide_code=True)
def _(mode):
    # Controls of the fit modes. They are made again when the mode changes,
    # so switching between mode 1 and mode 2 starts from a clean state.
    _ = mode
    observed = load("observed")
    obs_times = [day_to_time(iso_to_day(observed["start"]) + i) for i in range(observed["n"])]
    get_fit_ran, set_fit_ran = mo.state(None)                       # model key of the last run
    get_fit_tab, set_fit_tab = mo.state("Data and fixed dates")     # open tab
    get_fit_bands, set_fit_bands = mo.state(False)                  # Bayesian bands on/off

    fit_model = mo.ui.tabs({k: "" for k in MODEL_BY_LABEL}, orientation="vertical")
    _first, _last = (datetime.fromtimestamp(t / 1000, tz=timezone.utc).date() for t in (obs_times[0], obs_times[-1]))
    fit_period = mo.ui.date_range(start=_first, stop=_last, value=(_first, _last), full_width=True)

    def _run(clicks):
        # Always the least-squares fit first, bands off, Fitted model tab open.
        if fit_model.value:
            set_fit_ran(POINT_KEY[MODEL_BY_LABEL[fit_model.value]])
            set_fit_bands(False)
            set_fit_tab("Fitted model")
        return clicks + 1

    fit_run = mo.ui.button(label="▶  Run model fitting", value=0, on_click=_run, kind="success", full_width=True)
    fit_view = mo.ui.radio(["Fit", "Fit quality", "Parameters", "Rₜ", "Epidemic metrics", "Overlay"], value="Fit", inline=True)
    fit_ov_preset, fit_ov_left, fit_ov_right = overlay_widgets()
    return (fit_model, fit_ov_left, fit_ov_preset, fit_ov_right, fit_period, fit_run, fit_view, get_fit_bands,
            get_fit_ran, get_fit_tab, obs_times, observed, set_fit_bands, set_fit_tab)


@app.cell(hide_code=True)
def _(get_fit_bands, get_fit_tab, set_fit_bands, set_fit_tab):
    # Tab bar and bands switch follow the state, so a run can reset them.
    fit_tabs = mo.ui.tabs({"Data and fixed dates": "", "Parameters": "", "Fitted model": ""},
                          value=get_fit_tab(), on_change=set_fit_tab)
    fit_bands = mo.ui.switch(value=get_fit_bands(), on_change=set_fit_bands, label="**Show 95% Bayesian bands (MCMC)**")
    return fit_bands, fit_tabs


@app.cell(hide_code=True)
def _(fit_model, fit_period, fit_run, mode):
    # Left column of the fit modes
    _out = None
    if mode in ("early", "partial"):
        _period = ""
        if mode == "partial":
            _period = field(f"{fit_period}", "Period to display",
                            "All models were fitted on the whole of 2020. This selects the window shown in every plot.")
        _hint = "The models differ in how the hospitalisation probability ψ is constructed."
        _body = f'<div class="stack">{_period}{field(f"{fit_model}", "Model", _hint)}<div class="run-wrap">{fit_run}</div></div>'
        _sub = "Choose the period and a model." if mode == "partial" else "Whole of 2020. Choose a model."
        _out = page(mode, card(_body, "Set up the fit", _sub, "sliders-horizontal", accent=True), "side")
    _out
    return


@app.cell(hide_code=True)
def _(fit_tabs, mode):
    page(mode, f"{fit_tabs}") if mode in ("early", "partial") else None
    return


@app.cell(hide_code=True)
def _(
    fit_bands,
    fit_model,
    fit_ov_left,
    fit_ov_preset,
    fit_ov_right,
    fit_period,
    fit_view,
    get_fit_bands,
    get_fit_ran,
    get_fit_tab,
    mode,
    obs_times,
    observed,
):
    # Right column of the fit modes: the open tab
    def data_tab(start, end):
        panels = []
        for s in SERIES:
            fig = time_panel(obs_times, [("dots", "Observed", observed[s], COLORS[s])], start, end, SERIES_LABELS[s])
            panels.append(chart_card(chart(fig, 250), COLORS[s]))
        npi_rows = [dict(event=e[0], policy=fmt_date(day_to_time(iso_to_day(e[1]))),
                         cp=fmt_date(day_to_time(iso_to_day(e[2]))), days=str(e[3])) for e in NPI_EVENTS]
        return f"""<div class="stack">
          <p class="lede">Four observed daily series for Portugal in 2020. The dashed vertical lines are the <strong>fixed
          NPI dates</strong> (non-pharmaceutical interventions): the transmission rate β is constant between them, and can
          change at each one. For the age-stratified model and the testing-volume coupling, new data is introduced: the
          daily age composition of confirmed cases, and the daily testing volume.</p>
          {legend([*[("dot", COLORS[s], SERIES_LABELS[s].replace("Daily confirmed cases", "Cases")) for s in SERIES],
                   ("rule", CT["rule"], "Fixed NPI date")])}
          {panel_grid(panels)}
          {card(table([("event", "NPI event"), ("policy", "Policy date", "r"), ("cp", "Changepoint used", "r"),
                       ("days", "Days after policy", "r")], npi_rows),
                "Interventions and the changepoints used",
                "Each changepoint was estimated from the case data inside the 21 days that follow its policy date, "
                "giving 5 periods with their own β.")}
          <p class="small" style="font-size:13px"><strong>Data sources:</strong> daily confirmed cases, deaths, ward/ICU
          occupancy and the age breakdown of confirmed cases come from DSSG Portugal's
          <a href="https://github.com/dssg-pt/covid19pt-data">covid19pt-data</a> project. Daily testing volume comes from
          <a href="https://github.com/owid/covid-19-data">Our World in Data</a>'s testing dataset.</p></div>"""

    def params_tab(model):
        cols = [("p", "Parameter", "nw b"), ("m", "Meaning")]
        fixed = table(cols, [dict(p=p, m=m) for p, m in FIXED_PARAMS], dense=True)
        est = table(cols, [dict(p=p, m=m) for p, m in ESTIMATED_PARAMS], dense=True)
        note = callout("info", AGE_PARAM_NOTES[model]) if model in AGE_PARAM_NOTES else ""
        return f"""<div class="stack">
          <p class="lede">The model has fixed parameters taken from the literature, and parameters estimated from the data.
          Each model variant estimates a different set; the exact list for the model you run is in its
          <strong>Parameters</strong> view.</p>
          <div class="cols2">{card(fixed, f'{icon("lock")}Fixed')}{card(est, f'{icon("search-check")}Estimated from the data')}</div>
          {note}</div>"""

    def age_note(model, is_open):
        tb = ("<li><strong>Model 3b only:</strong> it also uses the daily testing volume (Our World in Data), whose 7-day "
              "rolling sum drives the coupling Ch(t) that scales ψ(t) down as testing grows.</li>") if model == "age_3b" else ""
        return f"""<details class="exp" {"open" if is_open else ""}>
          <summary><span class="ic">{icon("users")}</span>Age structure: new data and model structure<span class="chev">{icon("chevron-down")}</span></summary>
          <div class="in prose cols2">
            <div><h4>New data</h4><ul><li>Confirmed cases by age band and sex (DSSG Portugal), summed into four groups:
              0-49, 50-59, 60-69 and 70+.</li>{tb}</ul></div>
            <div><h4>New model structure</h4><ul>
              <li>Each age group has its own hospitalisation probability, doubling from one group to the next (ψ_base,
              2ψ_base, 4ψ_base, 8ψ_base), roughly matching the age-specific hospitalisation ratios in Caetano et al.</li>
              <li>Each day, ψ(t) is the average of the four group values weighted by that day's age composition, so ψ rises
              when older people make up more of the cases.</li></ul></div>
            <p style="grid-column:1/-1">{AGE_PARAM_NOTES[model]}</p></div></details>"""

    def models_at_a_glance():
        comp = load("comparison")["models"]
        max_j = max(m["J"]["total"] for m in comp.values())
        rows = []
        for name, m in comp.items():
            bar = (f'<div class="jbar"><div class="tr"><i style="width:{100 * m["J"]["total"] / max_j:.1f}%"></i></div>'
                   f'<b>{fmt_num(m["J"]["total"])}</b></div>')
            rows.append(dict(name=name, n=str(m["nEst"]), total=bar, **{k: fmt_num(m["J"][k]) for k in SERIES}))
        cols = [("name", "Model", "b"), ("n", "Estimated parameters", "r"),
                *[(k, f"J — {'ICU' if k == 'icu' else k}", "r") for k in SERIES], ("total", "J — total")]
        return card(table(cols, rows), "Models at a glance",
                    "J: sum of squared log-residuals between data and model. The Bayesian column uses the posterior-mean parameters.")

    def bands_explainer():
        return f"""<details class="exp"><summary><span class="ic">{icon("info")}</span>How the posterior was sampled and the two bands obtained<span class="chev">{icon("chevron-down")}</span></summary>
          <div class="in prose">
          <p>A short pilot chain first tunes the sampler; three longer chains (one started at the least-squares estimate, two
          started well away from it) are then run and pooled once they agree. The fitting is done on the effective sample size
          of the time series: consecutive days tend to share the same deviation from the model, so the data carry fewer
          independent observations than the raw number of days.</p>
          <p>500 parameter sets were taken from the chain, spread evenly over the pooled chains, and are used to build both bands.</p>
          <div class="cols2">
            <div><h4>95% credible interval (darker band): uncertainty in the model curve</h4><ul>
              <li>Each of the 500 parameter sets is run through the ODE model, giving 500 model trajectories per series.</li>
              <li>On each day, the band goes from the 2.5% to the 97.5% quantile of those 500 values, so it holds the middle
              95% of the trajectories.</li></ul></div>
            <div><h4>95% posterior predictive interval (lighter band)</h4><ul>
              <li>It uses the same 500 parameter sets, taken evenly spaced from the pooled chains, as the darker band.</li>
              <li>Every MCMC iteration saves, next to the parameter set, the noise variance σ² drawn at that same iteration;
              for parameter set <i>i</i>, its own σ<sub>i</sub>² is used.</li>
              <li>Each trajectory Y<sub>v</sub><sup>(i)</sup>(t) (series <i>v</i>, parameter set <i>i</i>) is run through the
              ODE model, as for the darker band, and then multiplied on each day by exp(ε<sub>i</sub>(t)), with
              ε<sub>i</sub>(t) ~ N(0, σ<sub>i</sub>²) drawn independently each day: the same log-scale noise the model is fit with.</li>
              <li>On each day, the band goes from the 2.5% to the 97.5% quantile of those 500 noisy values.</li>
              <li>It is the actual posterior predictive interval: the range a new observation on that day could plausibly fall in.</li>
            </ul></div></div></div></details>"""

    def weekly_residuals(obs, sim, start, end):
        weeks, vals = [], {s: [] for s in SERIES}
        for i, t in enumerate(obs_times):
            if not start <= t <= end:
                continue
            dt = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
            w = t - dt.weekday() * 86_400_000      # Monday of that week
            if not weeks or weeks[-1] != w:
                weeks.append(w)
                for s in SERIES:
                    vals[s].append(0.0)
            for s in SERIES:
                r = np.log(max(sim[s][i], 0) + 1) - np.log(obs[s][i] + 1)
                vals[s][-1] += float(r * r)
        return weeks, vals

    def result(model, key, bayes, start, end):
        fit = load(key)
        sim = fit["bands"]["point_est"] if bayes else fit["sim"]
        rt = fit["rt"]
        rt_times = [day_to_time(i) for i in range(len(rt))]
        label = MODEL_LABELS[key]
        short = MODEL_OPTIONS[model][1]

        # ---- charts
        fit_figs = []
        for s in SERIES:
            if bayes:
                b = fit["bands"]
                layers = [("band", "95% predictive", b["obs_lo"][s], b["obs_hi"][s], COLORS[s], 0.16),
                          ("band", "95% credible", b["pred_lo"][s], b["pred_hi"][s], COLORS[s], 0.38),
                          ("dots", "Observed", observed[s], CT["obs"]),
                          ("line", "Model (posterior mean)", sim[s], DEEP[s], 2)]
            else:
                layers = [("dots", "Observed", observed[s], CT["obs"]), ("line", "Fitted model", sim[s], COLORS[s])]
            fit_figs.append(time_panel(obs_times, layers, start, end, SERIES_LABELS[s]))
        rt_fig = time_panel(rt_times, [("line", "Rₜ", rt, COLORS["Rt"], 2.4)], start, end, rt_line=True, y_digits=1, y_title="Rₜ")
        weeks, wvals = weekly_residuals(observed, sim, start, end)
        resid_fig = residual_chart(weeks, [dict(name=SERIES_LABELS[s], color=COLORS[s], values=wvals[s]) for s in SERIES], start, end)
        pair = overlay_pair(fit_ov_preset, fit_ov_left, fit_ov_right)
        names = {"Rₜ": "Rt", "Daily cases": "cases", "General ward": "ward", "ICU": "icu", "Daily deaths": "deaths"}

        def side(n):
            k = names[n]
            if k == "Rt":
                return dict(name=n, times=rt_times, values=rt, color=COLORS["Rt"], is_rt=True)
            return dict(name=n, times=obs_times, values=sim[k], color=COLORS[k], is_rt=False)

        ov_fig = overlay_chart(start, end, side(pair[0]), side(pair[1]))
        hist_figs = []
        if bayes:
            for bp in fit["params"]:
                hst = fit["hist"].get(bp["name"])
                if hst:
                    hist_figs.append(histogram_chart(param_symbol(bp["name"]), hst["edges"], hst["counts"],
                                                     bp["post_mean"], bp["point_est"], bp["ci_low"], bp["ci_high"]))
        m = compute_metrics(obs_times, sim, rt_times, rt, start, end)

        # ---- tables
        lbl = {"cases": "Cases", "ward": "Ward", "icu": "ICU", "deaths": "Deaths", "total": "Total"}
        if bayes:
            f = fit["fit"]
            j_rows = [dict(Series=lbl[k], a=f["J_post_mean_params"][k], b=f["J_draws_mean"][k],
                           lo=f["J_draws_lo"][k], hi=f["J_draws_hi"][k]) for k in lbl]
            comp = load("comparison")["models"]
            others = [("Model 1 — constant", "Model 1"), ("Model 2 — piecewise ψ", "Model 2"),
                      ("Model 3a — age-stratified, without coupling", "Model 3a"),
                      ("Model 3b — age-stratified, with coupling (least squares)", "Model 3b (least squares)")]
            comp_rows = [dict(Model=a, J=comp[b]["J"]["total"], own=False) for a, b in others] + [
                dict(Model=f"{short} — Bayesian, at posterior-mean parameters", J=f["J_post_mean_params"]["total"], own=True),
                dict(Model=f"{short} — Bayesian, posterior mean of J", J=f["J_draws_mean"]["total"], own=True)]
            param_rows = [dict(name=param_symbol(p["name"]), ls=fmt_sig(p["point_est"]), mean=fmt_sig(p["post_mean"]),
                               ci=f"[{fmt_sig(p['ci_low'])}, {fmt_sig(p['ci_high'])}]", meaning=param_meaning(p["name"]))
                          for p in fit["params"]]
        else:
            j_rows = [dict(Series=lbl[k], a=fit["J"][k]) for k in lbl]
            comp_rows = []
            param_rows = [dict(name=param_symbol(p["name"]), value=fmt_sig(p["value"]), meaning=param_meaning(p["name"]))
                          for p in fit["params"]]

        # ---- ZIP download: tables as CSV, charts as interactive HTML pages
        def build_zip():
            comparison = load("comparison")
            models = list(comparison["models"])
            comp_csv = [[param_symbol(p), *[comparison["models"][mm]["params"][p] for mm in models]] for p in comparison["order"]]
            comp_csv.append(["Number of estimated parameters", *[comparison["models"][mm]["nEst"] for mm in models]])
            for k, l in [("cases", "J — cases"), ("ward", "J — ward"), ("icu", "J — ICU"), ("deaths", "J — deaths"), ("total", "J — total")]:
                comp_csv.append([l, *[comparison["models"][mm]["J"][k] for mm in models]])
            files = {
                "all_models_comparison.csv": to_csv(["Parameter / error", *models], comp_csv),
                "metrics.csv": to_csv(["Metric", "Simulated"], metrics_csv_rows(m)),
                "objective_function_J.csv": (
                    to_csv(["Series", "J at posterior-mean parameters", "Posterior mean of J", "95% interval of J"],
                           [[r["Series"], round(r["a"], 1), round(r["b"], 1), f"[{fmt_num(r['lo'])}, {fmt_num(r['hi'])}]"] for r in j_rows])
                    if bayes else to_csv(["Series", "J"], [[r["Series"], round(r["a"], 1)] for r in j_rows])),
                "weekly_residual_breakdown.csv": to_csv(["week", *SERIES], [[fmt_date(w), *[round(wvals[s][i], 4) for s in SERIES]]
                                                                           for i, w in enumerate(weeks)]),
                "estimated_parameters.csv": (
                    to_csv(["Parameter", "Least-squares estimate (start of the centred chain)", "Posterior mean", "95% CI", "Meaning"],
                           [[r["name"], r["ls"], r["mean"], r["ci"], r["meaning"]] for r in param_rows])
                    if bayes else to_csv(["Parameter", "Estimate", "Meaning"], [[r["name"], r["value"], r["meaning"]] for r in param_rows])),
                "fitted_model.html": figures_html(label, fit_figs),
                "weekly_residual_breakdown.html": figures_html("Weekly squared log-residuals", [resid_fig]),
                "effective_reproduction_number.html": figures_html("Effective reproduction number Rₜ", [rt_fig]),
                "overlay.html": figures_html(f"{pair[0]} (left) vs {pair[1]} (right) — simulated", [ov_fig]),
            }
            if bayes:
                files["model_comparison_J.csv"] = to_csv(["Model", "J (total)"], [[r["Model"], round(r["J"], 1)] for r in comp_rows])
                files["posterior_histograms.html"] = figures_html("Posterior distributions of the parameters", hist_figs)
            return zip_bytes(files)

        download = mo.download(data=build_zip, filename=f"{key}_results.zip", label="Download results (ZIP)")

        # ---- the open view
        view = fit_view.value
        if view == "Fit":
            leg = (legend([("dot", CT["obs"], "Observed"), ("line", CT["muted"], "Model at posterior-mean parameters"),
                           ("band", CT["muted"], "95% credible interval"), ("band-light", CT["muted"], "95% posterior predictive"),
                           ("rule", CT["rule"], "NPI date")]) if bayes else
                   legend([("dot", CT["obs"], "Observed"), ("line", CT["muted"], "Fitted model"), ("rule", CT["rule"], "NPI date")]))
            body = (leg + panel_grid([chart_card(chart(f, 260), COLORS[s]) for f, s in zip(fit_figs, SERIES)])
                    + '<p class="small">Hover a panel to read the values of each day.</p>'
                    + (bands_explainer() if bayes else ""))
        elif view == "Fit quality":
            intro = ("The residuals are the same log-residuals as in the other models, now taken between the data and the "
                     "<strong>solid line</strong> (the model run with the posterior mean of each parameter):" if bayes else
                     "Parameters are estimated by minimising the sum of squared log-residuals between each observed series "
                     "and its simulated counterpart, summed over the four series:")
            formula = mo.md(r"$$J = \sum_{s \,\in\, \{\text{cases, ward, ICU, deaths}\}} \sum_{t} \Big(\log(y_s(t)+1) - \log(\hat{y}_s(t)+1)\Big)^2$$")
            more = ("<p>Since every parameter set in the posterior gives its own curve, J is also computed for each of the 500 "
                    "parameter sets used for the bands, giving a posterior mean and a 95% interval for J.</p>" if bayes else "")
            if bayes:
                jt = table([("Series", "Series", "b"), ("a", "J at posterior-mean parameters", "r"), ("b", "Posterior mean of J", "r"),
                            ("ci", "95% interval of J", "r")],
                           [dict(Series=r["Series"], a=fmt_num(r["a"]), b=fmt_num(r["b"]), ci=f"[{fmt_num(r['lo'])}, {fmt_num(r['hi'])}]",
                                 tot=r["Series"] == "Total") for r in j_rows], dense=True, highlight=lambda r: r["tot"])
                ct_ = table([("Model", "Model"), ("J", "J (total)", "r")],
                            [dict(Model=r["Model"], J=fmt_num(r["J"]), own=r["own"]) for r in comp_rows],
                            dense=True, highlight=lambda r: r["own"])
                tables = (f'<div class="stack" style="margin-top:16px"><div><h4 class="sec">J per series</h4>{jt}</div>'
                          f'<div><h4 class="sec">Comparison with the other models <span class="muted" style="font-weight:400">'
                          f'(simultaneous fitting)</span></h4>{ct_}</div></div>'
                          '<p class="small" style="margin-top:16px">The least-squares fit is, by construction, the parameter set '
                          'with the lowest J, so the posterior-mean curve has a slightly higher J, and the posterior mean of J, '
                          'which averages over all plausible parameter sets, is higher still.</p>')
            else:
                tables = ('<div style="margin-top:16px;max-width:28rem">' + table(
                    [("Series", "Series", "b"), ("a", "J", "r")],
                    [dict(Series=r["Series"], a=fmt_num(r["a"]), tot=r["Series"] == "Total") for r in j_rows],
                    dense=True, highlight=lambda r: r["tot"]) + "</div>")
            body = (card(f'<div class="lede" style="font-size:14px"><p style="margin:0 0 12px">{intro}</p>'
                         f'<div class="formula">{formula.text}</div>{more}</div>{tables}',
                         "Fit quality of the posterior" if bayes else "Objective function")
                    + card(chart(resid_fig, 320), "Weekly residual breakdown",
                           "Weekly sum of squared log-residuals, stacked by series"
                           + (", between the data and the solid line (model at the posterior-mean parameters)." if bayes else ".")))
        elif view == "Parameters":
            if bayes:
                pt = table([("name", "Parameter", "b nw"), ("ls", "Least-squares estimate (start of the centred chain)", "r"),
                            ("mean", "Posterior mean", "r"), ("ci", "95% CI", "r nw"), ("meaning", "Meaning", "soft")], param_rows)
                hists = panel_grid([chart_card(chart(f, 170)) for f in hist_figs], cols=4)
                body = (pt + f'<p class="small">Least-squares estimate: the {short} least-squares fit, used as the starting point '
                        'of the centred chain (not estimated by the MCMC). Posterior mean and 95% CI: from the pooled MCMC draws.</p>'
                        '<h3 class="sec">Posterior distributions of the parameters</h3>'
                        + legend([("band-light", CT["obs"], f"Histogram of {fit['nDraws']} posterior draws, evenly spaced over the pooled chains"),
                                  ("line", CT["window"], "Posterior mean"), ("dash", CT["text"], "Least-squares estimate"),
                                  ("window", CT["window"], "95% CI")]) + hists)
            else:
                body = table([("name", "Parameter", "b nw"), ("value", "Estimate", "r"), ("meaning", "Meaning", "soft")], param_rows)
        elif view == "Rₜ":
            body = card(chart(rt_fig, 300), "Effective reproduction number",
                        "Rₜ = β(t)(1/γ_C + 1/γ_I) · S(t)/N.  Rₜ > 1: the epidemic grows. Rₜ < 1: the epidemic shrinks.")
        elif view == "Epidemic metrics":
            body = metrics_view(m, f"Summary of the simulated trajectories over the period shown "
                                   f"(<strong>{fmt_date(start)} – {fmt_date(end)}</strong>)"
                                + (", using the model at the posterior-mean parameters (solid line)." if bayes else "."))
        else:
            note = ('<p class="small">Simulated series: the model at the posterior-mean parameters (solid line in the Fit view).</p>'
                    if bayes else "")
            body = card(f'<div class="stack-sm">{overlay_picker_html(fit_ov_preset, fit_ov_left, fit_ov_right)}'
                        f"{chart(ov_fig, 360)}{note}</div>", "Overlay",
                        "Compare two simulated series on the same time axis, each with its own y-axis, so that their different "
                        "scales do not matter.")
        hint = ('<span class="hint-under">Switches from the least-squares point estimates to the Bayesian posterior: credible '
                'and predictive bands, the model at the posterior-mean parameters, and the posterior of every parameter.</span>')
        return f"""<div class="stack">
          <div class="result-head"><div><h2>{label}</h2><p>{fmt_date(start)} – {fmt_date(end)}</p></div>{download}</div>
          <div class="soft-panel">{fit_bands}{hint}</div>
          <div>{fit_view}</div>
          <div class="stack">{body}</div></div>"""

    def fitted_tab(model, ran, start, end):
        out = age_note(model, ran is None or ran != POINT_KEY.get(model)) if model in ("age_nocorr", "age_3b") else ""
        if model is None:
            out += empty_state("arrow-left", "Pick a model",
                               "Choose one of the four models, then press <strong>Run model fitting</strong>. The table below "
                               "compares the fit error J of every model.") + models_at_a_glance()
        elif ran != POINT_KEY[model]:
            out += empty_state("play", "Ready to run", "Press <strong>Run model fitting</strong> to see the fitted result. "
                               "The table below compares the fit error J of every model.") + models_at_a_glance()
        else:
            bayes = bool(get_fit_bands())
            out += result(model, BAYES_KEY[model] if bayes else ran, bayes, start, end)
        return f'<div class="stack">{out}</div>'

    _out = None
    if mode in ("early", "partial"):
        _ = fit_bands
        if mode == "partial":
            _a, _b = fit_period.value
            _start = day_to_time((_a - DAY0.date()).days)
            _end = day_to_time((_b - DAY0.date()).days)
        else:
            _start, _end = obs_times[0], obs_times[-1]
        _tab = get_fit_tab()
        if _tab == "Data and fixed dates":
            _html = data_tab(_start, _end)
        elif _tab == "Parameters":
            _html = params_tab(MODEL_BY_LABEL.get(fit_model.value))
        else:
            _html = fitted_tab(MODEL_BY_LABEL.get(fit_model.value), get_fit_ran(), _start, _end)
        _out = page(mode, f'<div style="margin-top:20px">{_html}</div>')
    _out
    return


# =========================================================================
# Mode 3: transmission scenarios
# =========================================================================
@app.function
def scenario_title(model: str, scn: tuple | None) -> str:
    short = {"sim_m1": "Model 1", "sim_m2": "Model 2", "age_nocorr": "Model 3a", "age_point": "Model 3b"}[model]
    if scn is None:
        return f"{short} · Baseline"
    if scn[0] == "lockdown":
        return f"{short} · Extra lockdown from {fmt_date(day_to_time(scn[1]))}, {scn[2]} days, Rₜ −{scn[3]}%"
    return (f"{short} · School reopening delayed to {fmt_date(day_to_time(SCHOOL_CP + scn[1]))} "
            f"({scn[1]} days later)")


@app.function
def scenario_window(scn: tuple | None):
    """(start ms, end ms, label) of the days where the scenario differs."""
    if scn is None:
        return None
    if scn[0] == "lockdown":
        return (day_to_time(scn[1]), day_to_time(scn[1] + scn[2]), "Extra Rₜ reduction")
    return (day_to_time(SCHOOL_CP), day_to_time(SCHOOL_CP + scn[1]), "School reopening delayed")


@app.function
def day_date(d: int) -> date:
    return (DAY0 + timedelta(days=int(d))).date()


@app.cell(hide_code=True)
def _():
    # Controls of the scenario mode (kept when switching modes).
    get_scn_config, set_scn_config = mo.state(None)   # the last run: (model, scenario, (first day, last day))
    get_saved, set_saved = mo.state([])                # saved simulations, for Compare
    get_scn_tab, set_scn_tab = mo.state("Results")
    get_scn_bands, set_scn_bands = mo.state(False)
    SCN_MODELS = {model_label(MODEL_OPTIONS[c][0], MODEL_OPTIONS[c][2]): v for v, c in
                  [("sim_m1", "sim_m1"), ("sim_m2", "sim_m2"), ("age_nocorr", "age_nocorr"), ("age_point", "age_3b")]}
    scn_model = mo.ui.tabs({k: "" for k in SCN_MODELS}, orientation="vertical")
    scn_type = mo.ui.radio({"Baseline": "baseline", "Extra lockdown": "lockdown", "Later reopening": "school"},
                           value="Extra lockdown", inline=True)
    scn_zone = mo.ui.radio({z[0]: z for z in FREE_ZONES}, value=FREE_ZONES[1][0])
    scn_red = mo.ui.slider(5, 90, step=5, value=40, show_value=True, full_width=True)
    scn_delay = mo.ui.slider(1, MAX_SCHOOL_DELAY, value=21, show_value=True, full_width=True)
    scn_period = mo.ui.date_range(start=day_date(0), stop=day_date(N_DAYS - 1),
                                  value=(day_date(0), day_date(N_DAYS - 1)), full_width=True)
    scn_view = mo.ui.radio(["Plots", "Epidemic metrics", "Overlay"], value="Plots", inline=True)
    scn_ov_preset, scn_ov_left, scn_ov_right = overlay_widgets()
    return (SCN_MODELS, get_saved, get_scn_bands, get_scn_config, get_scn_tab, scn_delay, scn_model, scn_ov_left, scn_ov_preset,
            scn_ov_right, scn_period, scn_red, scn_type, scn_view, scn_zone, set_saved, set_scn_bands, set_scn_config,
            set_scn_tab)


@app.cell(hide_code=True)
def _(scn_zone):
    # Start date of the extra lockdown, limited to the chosen zone.
    zone = scn_zone.value
    scn_start = mo.ui.date(start=day_date(zone[1]), stop=day_date(zone[2] - 1), value=day_date(zone[1]), full_width=True)
    return scn_start, zone


@app.cell(hide_code=True)
def _(scn_start, zone):
    # The duration is capped so the window ends before the next real lockdown.
    start_day = min(max((scn_start.value - DAY0.date()).days, zone[1]), zone[2] - 1)
    max_dur = zone[2] - start_day
    scn_dur = mo.ui.slider(1, max_dur, value=min(14, max_dur), show_value=True, full_width=True) if max_dur > 1 else None
    return max_dur, scn_dur, start_day


@app.cell(hide_code=True)
def _(
    SCN_MODELS,
    scn_delay,
    scn_dur,
    scn_model,
    scn_period,
    scn_red,
    scn_type,
    set_scn_bands,
    set_scn_config,
    set_scn_tab,
    start_day,
):
    # "Run scenario simulation": freeze the current set-up as the run to show;
    # every new run starts without the Bayesian band.
    def _run(clicks):
        kind = scn_type.value
        if kind == "baseline":
            scn = None
        elif kind == "lockdown":
            scn = ("lockdown", start_day, scn_dur.value if scn_dur is not None else 1, scn_red.value)
        else:
            scn = ("school", scn_delay.value)
        a, b = scn_period.value
        set_scn_config((SCN_MODELS[scn_model.value], scn, ((a - DAY0.date()).days, (b - DAY0.date()).days)))
        set_scn_bands(False)
        set_scn_tab("Results")
        return clicks + 1

    scn_run = mo.ui.button(label="▶  Run scenario simulation", value=0, on_click=_run, kind="success", full_width=True)
    return (scn_run,)


@app.cell(hide_code=True)
def _(
    max_dur,
    mode,
    scn_delay,
    scn_dur,
    scn_model,
    scn_period,
    scn_red,
    scn_run,
    scn_start,
    scn_type,
    scn_zone,
    start_day,
    zone,
):
    # Left column of the scenario mode
    _out = None
    if mode == "scenarios":
        _type = scn_type.value
        if _type == "lockdown":
            _dur = scn_dur.value if scn_dur is not None else 1
            _dur_field = (field(f"{scn_dur}", "Duration of the Rₜ reduction (days)",
                                "Capped so the window ends before the next real lockdown. Afterwards it is back to the baseline Rₜ.")
                          if scn_dur is not None else
                          "<p>Only 1 day of room left before the next real lockdown. Pick an earlier start date for a longer window.</p>")
            _extra = f"""<div class="box">
              <p>The two real lockdowns stay as fitted: spring 2020 (<strong>{fmt_date(day_to_time(FITTED_CP[0]))} –
              {fmt_date(day_to_time(FITTED_CP[1]))}</strong>) and the Nov 2020 tightening (from
              <strong>{fmt_date(day_to_time(FITTED_CP[3]))}</strong>). Place the extra lockdown anywhere else, for any duration.</p>
              {field(f"{scn_zone}", "When")}
              {field(f"{scn_start}", "Extra lockdown start date",
                     f"Any date between {fmt_date(day_to_time(zone[1]))} and {fmt_date(day_to_time(zone[2] - 1))}.")}
              {_dur_field}
              {field(f"{scn_red}", "Rₜ reduction (%)", "Reduction relative to the baseline Rₜ, inside the window.")}
              <p class="ink">Rₜ reduced from <strong>{fmt_date(day_to_time(start_day))}</strong> to
              <strong>{fmt_date(day_to_time(min(start_day + _dur, N_DAYS - 1)))}</strong>.</p></div>"""
        elif _type == "school":
            _extra = f"""<div class="box">
              <p>Schools actually reopened on <strong>{fmt_date(day_to_time(SCHOOL_CP))}</strong>, when β steps up from the
              lower summer level to the higher post-reopening level. Push that switch later, as long as it doesn't reach the
              next real changepoint, the Nov 2020 tightening on <strong>{fmt_date(day_to_time(NEXT_CP))}</strong>.</p>
              {field(f"{scn_delay}", "Delay school reopening by (days)")}
              <p class="ink">Reopening moved from <strong>{fmt_date(day_to_time(SCHOOL_CP))}</strong> to
              <strong>{fmt_date(day_to_time(SCHOOL_CP + scn_delay.value))}</strong>.</p></div>"""
        else:
            _extra = '<p class="small">The fitted model as it is, with no counterfactual change.</p>'
        _body = (f'<div class="stack">{field(f"{scn_model}", "Model")}{field(f"{scn_type}", "Scenario")}{_extra}'
                 f'{field(f"{scn_period}", "Period to show")}<div class="run-wrap">{scn_run}</div></div>')
        _out = page(mode, card(_body, "Set up a scenario", "Simulated values only, on top of each model's fitted transmission.",
                               "flask-conical", accent=True), "side")
    _out
    return


@app.cell(hide_code=True)
def _(get_saved, get_scn_tab, mode, set_scn_tab):
    # Tab bar of the scenario mode
    scn_tabs = mo.ui.tabs({"Results": "", "Compare": ""}, value=get_scn_tab(), on_change=set_scn_tab)
    _n = len(get_saved())
    _badge = f'<span class="tab-note">{_n} saved simulation{"s" if _n != 1 else ""}</span>' if _n else ""
    page(mode, f"{scn_tabs}{_badge}") if mode == "scenarios" else None
    return


@app.cell(hide_code=True)
def _(get_saved, get_scn_bands, get_scn_config, set_saved, set_scn_bands):
    # Controls shown with a result: its name, Save, and the band switch.
    _config = get_scn_config()
    _available = _config is not None and _config[0] == "age_point" and _config[1] is None
    scn_name = mo.ui.text(value=scenario_title(*_config[:2]) if _config else "", full_width=True)
    scn_bands = mo.ui.switch(value=get_scn_bands() and _available, disabled=not _available, on_change=set_scn_bands,
                             label=f"**Show 95% Bayesian band ({N_BAND_DRAWS} posterior draws)**")

    def _save(clicks):
        model, scn, period = get_scn_config()
        tr = run_scenario(model, scn)
        base = scn_name.value.strip() or scenario_title(model, scn)
        names = {s["name"] for s in get_saved()}
        final, k = base, 2
        while final in names:
            final, k = f"{base} ({k})", k + 1
        i0, i1 = period[0], period[1] + 1
        item = dict(name=final, times=[day_to_time(d) for d in range(i0, i1)], data={k_: v[i0:i1] for k_, v in tr.items()})
        set_saved(lambda saved: saved + [item])
        mo.status.toast(f"Saved as “{final}”", "Open Compare to compare it with other saved simulations.")
        return clicks + 1

    scn_save = mo.ui.button(label="Save simulation", value=0, on_click=_save)
    return scn_bands, scn_name, scn_save


@app.cell(hide_code=True)
def _(
    get_scn_bands,
    get_scn_config,
    get_scn_tab,
    mode,
    scn_bands,
    scn_name,
    scn_ov_left,
    scn_ov_preset,
    scn_ov_right,
    scn_save,
    scn_view,
):
    # Results of the last scenario run
    def results(config):
        model, scn, period = config
        tr = run_scenario(model, scn)
        times = [day_to_time(d) for d in range(N_DAYS)]
        start, end = times[period[0]], times[period[1]]
        window = scenario_window(scn)
        title = scenario_title(model, scn)
        inp = load("sim_inputs")[model]
        b4, b5 = inp["betas"][3], inp["betas"][4]
        bands_available = model == "age_point" and scn is None
        bands = baseline_bands() if bands_available and get_scn_bands() else None
        metrics = compute_metrics(times, tr, times, tr["Rt"], start, end)

        def panel(s):
            layers = []
            if s != "Rt" and bands:
                layers.append(("band", "95% predictive", bands[s]["lo"], bands[s]["hi"], COLORS[s], 0.22))
            layers.append(("line", "Scenario" if scn else "Simulated", tr[s], COLORS[s]))
            return time_panel(times, layers, start, end, None if s == "Rt" else SERIES_SHORT[s], window=window,
                              rt_line=s == "Rt", y_digits=1 if s == "Rt" else None, y_title="Rₜ" if s == "Rt" else None)

        view = scn_view.value
        if view == "Plots":
            leg = [("line", CT["muted"], "Scenario" if scn else "Simulated (fitted baseline)")]
            if bands:
                leg.append(("band", CT["muted"], f"95% predictive band ({N_BAND_DRAWS} posterior draws)"))
            if window:
                leg.append(("window", CT["window"], f"Window where it differs ({window[2]})"))
            leg.append(("rule", CT["rule"], "Fitted changepoint"))
            body = (legend(leg) + panel_grid([chart_card(chart(panel(s), 250), COLORS[s]) for s in SERIES])
                    + card(chart(panel("Rt"), 260), "Effective reproduction number Rₜ"))
        elif view == "Epidemic metrics":
            body = metrics_view(metrics)
        else:
            names = {"Rₜ": "Rt", "Daily cases": "cases", "General ward": "ward", "ICU": "icu", "Daily deaths": "deaths"}
            a, b = overlay_pair(scn_ov_preset, scn_ov_left, scn_ov_right)
            side = lambda n: dict(name=n, times=times, values=tr[names[n]], color=COLORS[names[n]], is_rt=names[n] == "Rt")
            body = card(f'<div class="stack-sm">{overlay_picker_html(scn_ov_preset, scn_ov_left, scn_ov_right)}'
                        f'{chart(overlay_chart(start, end, side(a), side(b), window), 360)}</div>', "Overlay",
                        "Compare two simulated series on the same time axis, each with its own y-axis, so that their "
                        "different scales do not matter.")
        band_hint = ("Integrates 500 posterior draws of Model 3b and adds the fitted observation noise, corrected for residual "
                     "autocorrelation." if bands_available else "Available for the Model 3b baseline (no counterfactual scenario).")
        return f"""<div class="stack">
          <div><div class="result-head"><h2>{title}</h2></div>
          <p class="small" style="font-size:13px;margin-top:4px">Period: {fmt_date(start)} – {fmt_date(end)}. Only simulated values
          are shown. For reference, the fitted switch on 06/11/2020 already reduces β by {100 * (1 - b5 / b4):.0f}% in this model
          (β₄ = {b4:.4f} → β₅ = {b5:.4f}).</p></div>
          <div class="soft-panel" style="display:flex;flex-wrap:wrap;align-items:flex-end;gap:8px">
            <div style="flex:1;min-width:220px">{field(f"{scn_name}", "Name for this simulation")}</div>{scn_save}</div>
          <div class="soft-panel">{scn_bands}<span class="hint-under">{band_hint}</span></div>
          <div>{scn_view}</div>
          <div class="stack">{body}</div></div>"""

    _out = None
    if mode == "scenarios" and get_scn_tab() == "Results":
        _config = get_scn_config()
        if _config is None:
            _html = empty_state("arrow-left", "Set up a scenario",
                                "Choose a model and a scenario: an extra lockdown (start date, duration, Rₜ reduction) or a later "
                                "school reopening (delay). Then press <strong>Run scenario simulation</strong>. Save the results and "
                                "open <strong>Compare</strong> to set them against the baseline.")
        else:
            _html = results(_config)
        _out = page(mode, f'<div style="margin-top:20px">{_html}</div>')
    _out
    return


@app.cell(hide_code=True)
def _(get_saved, set_saved):
    # Compare: which saved simulations, which series, and removing them.
    _names = [s["name"] for s in get_saved()]
    cmp_chosen = mo.ui.multiselect(_names, value=_names, full_width=True)
    cmp_series = mo.ui.radio({SERIES_SHORT[s]: s for s in (*SERIES, "Rt")}, value="Daily cases", inline=True)
    cmp_drop = mo.ui.multiselect(_names, full_width=True)

    def _remove(clicks):
        drop = set(cmp_drop.value)
        if drop:
            set_saved(lambda saved: [s for s in saved if s["name"] not in drop])
        return clicks + 1

    cmp_remove = mo.ui.button(label="Remove selected", value=0, on_click=_remove, kind="danger")
    return cmp_chosen, cmp_drop, cmp_remove, cmp_series


@app.cell(hide_code=True)
def _(cmp_chosen, cmp_drop, cmp_remove, cmp_series, get_saved, get_scn_tab, mode):
    def compare(saved):
        if not saved:
            return empty_state("bookmark", "No saved simulations yet",
                               "Run a scenario in <strong>Results</strong>, press <strong>Save simulation</strong>, and repeat "
                               "for the scenarios you want to compare. Saved simulations stay while this page is open.")
        chosen = cmp_chosen.value
        items = [s for s in saved if s["name"] in chosen]
        series = cmp_series.value
        is_rt = series == "Rt"
        label = SERIES_SHORT[series]
        out = card(f'<div class="stack-sm">{cmp_chosen}{cmp_series}</div>', "Saved simulations",
                   "Differences in the table are relative to the first one selected.")
        if items:
            fig = compare_chart([dict(name=s["name"], times=s["times"], values=s["data"][series]) for s in items], label, is_rt)
            ms = [compute_metrics(s["times"], s["data"], s["times"], s["data"]["Rt"], s["times"][0], s["times"][-1]) for s in items]
            ref = ms[0]

            def pct(v, r):
                if r == 0:
                    return "—"
                d = 100 * (v / r - 1)
                cls = "muted" if abs(d) < 0.05 else ("pos" if d < 0 else "neg")
                return f'<span class="{cls}">{"+" if d > 0 else ""}{d:.1f}%</span>'

            rows = []
            for s, m in zip(items, ms):
                peak = (m["rt_max"], m["rt_day"]) if is_rt else m["peaks"][series]
                rows.append(dict(name=s["name"], period=f"{fmt_date(s['times'][0])} – {fmt_date(s['times'][-1])}",
                                 cases=fmt_int(m["cases"]), dc=pct(m["cases"], ref["cases"]),
                                 deaths=fmt_int(m["deaths"]), dd=pct(m["deaths"], ref["deaths"]),
                                 peak=fmt_num(peak[0], 2) if is_rt else fmt_int(peak[0]), day=fmt_date(peak[1])))
            csv_rows = [[iso(t), *[round(s["data"][k][i], 4) for k in SERIES], round(s["data"]["Rt"][i], 5), s["name"]]
                        for s in items for i, t in enumerate(s["times"])]
            dl_csv = mo.download(data=to_csv(["date", *SERIES, "Rt", "simulation"], csv_rows).encode(),
                                 filename="saved_scenarios.csv", label="Trajectories (CSV)")
            dl_plot = mo.download(data=lambda: figures_html(f"{label} — saved simulations", [fig]).encode(),
                                  filename=f"compare_{series}.html", label="Comparison plot (HTML)")
            out += f"""{chart_card(chart(fig, 380 + 20 * len(items)))}
              <div><h4 class="sec">Summary <span class="muted" style="font-weight:400">(differences relative to
              <strong>{items[0]["name"]}</strong>)</span></h4>
              {table([("name", "Simulation", "b"), ("period", "Period", "nw soft"), ("cases", "Total cases", "r"), ("dc", "vs first", "r"),
                      ("deaths", "Total deaths", "r"), ("dd", "vs first", "r"),
                      ("peak", "Highest Rₜ" if is_rt else f"Peak {label}", "r"), ("day", "Peak day", "r")], rows)}</div>
              <div style="display:flex;flex-wrap:wrap;gap:8px">{dl_csv}{dl_plot}</div>"""
        out += f"""<details class="exp"><summary><span class="ic">{icon("trash-2")}</span>Manage saved simulations
          <span class="chev">{icon("chevron-down")}</span></summary><div class="in stack-sm">{cmp_drop}{cmp_remove}</div></details>"""
        return f'<div class="stack">{out}</div>'

    _out = None
    if mode == "scenarios" and get_scn_tab() == "Compare":
        _out = page(mode, f'<div style="margin-top:20px">{compare(get_saved())}</div>')
    _out
    return


# =========================================================================
# Mode 4 and the footer
# =========================================================================
@app.cell(hide_code=True)
def _(mode):
    _out = None
    if mode == "vaccination":
        _out = mo.Html('<div data-mode="vaccination">' + empty_state(
            "syringe", "Model with vaccination — coming soon",
            "<p style='margin:0'>This mode is not available yet. It will extend one of the calibrated models (most likely Model 3b) "
            "with vaccination, for later stages of the pandemic. Meanwhile, explore the fitted models (modes 1 and 2) or the "
            "transmission scenarios (mode 3).</p>") + "</div>")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.Html("""<div class="footer-anchor"></div><div class="page-end"></div><footer class="footer"><div class="line"></div><div class="in">
      <span>Data: <a href="https://github.com/dssg-pt/covid19pt-data">DSSG Portugal</a>,
      <a href="https://github.com/owid/covid-19-data">Our World in Data</a> ·
      Model based on <a href="https://doi.org/10.3390/v14102114">Dings et al. (2022)</a></span>
      <span>Part of the EUVABECO project</span></div></footer>""")
    return


if __name__ == "__main__":
    app.run()
