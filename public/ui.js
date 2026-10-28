// Small UI kit for the marimo app: every control and chart on the page is one
// `UI` widget (see app.py), and `kind` picks how it is drawn. Python sends the
// options in `props`; the user's choice comes back in `value` (or `clicks`).
// Charts are Plotly figures built in Python (figure.to_json()).

const FONT = '"Inter Variable", system-ui, sans-serif';
const PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js";
const svg = (inner, size = 16) =>
  `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${inner || ""}</svg>`;
const h = (html) => {
  const t = document.createElement("template");
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

// ---------------------------------------------------------------------------
// Plotly: loaded once from its CDN. Its stylesheet lives in the page <head>;
// widgets are drawn inside a shadow root, so the rules are copied there too.
// ---------------------------------------------------------------------------
let plotlyPromise = null;
function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!plotlyPromise) {
    plotlyPromise = new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = PLOTLY_URL;
      s.onload = () => resolve(window.Plotly);
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }
  return plotlyPromise;
}

function shareStyles(node) {
  const root = node.getRootNode();
  if (!(root instanceof ShadowRoot) || root.querySelector("style[data-plotly]")) return;
  const css = [];
  for (const sheet of document.styleSheets) {
    const owner = sheet.ownerNode;
    if (!owner || !(owner.id || "").startsWith("plotly")) continue;
    try { for (const r of sheet.cssRules) css.push(r.cssText); } catch (_) { /* cross-origin */ }
  }
  const st = document.createElement("style");
  st.dataset.plotly = "";
  st.textContent = css.join("\n");
  root.appendChild(st);
}

export const ZOOM_HINT = "Ctrl + scroll (or pinch) to zoom, drag to move, double-click to reset.";
const CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false, doubleClick: "reset" };

/** Ctrl + wheel zooms the time axis around the pointer; plain wheel scrolls the page. */
function ctrlWheelZoom(div, Plotly) {
  const onWheel = (e) => {
    if (!e.ctrlKey) return;
    const xa = div._fullLayout && div._fullLayout.xaxis;
    if (!xa || xa.fixedrange) return;
    e.preventDefault();
    const box = div.querySelector(".nsewdrag") || div;
    const r = box.getBoundingClientRect();
    const frac = Math.min(Math.max((e.clientX - r.left) / r.width, 0), 1);
    const [a, b] = xa.range.map((v) => (typeof v === "string" ? Date.parse(v.replace(" ", "T") + "Z") : v));
    const x = a + frac * (b - a);
    const k = e.deltaY > 0 ? 1.25 : 0.8;
    const na = x - (x - a) * k, nb = x + (b - x) * k;
    const isDate = xa.type === "date";
    const fmt = (v) => (isDate ? new Date(v).toISOString().replace("T", " ").replace("Z", "") : v);
    Plotly.relayout(div, { "xaxis.range": [fmt(na), fmt(nb)] });
  };
  div.addEventListener("wheel", onWheel, { passive: false });
  return () => div.removeEventListener("wheel", onWheel);
}

function mountChart(div, figure) {
  let alive = true, undo = () => {};
  loadPlotly().then((Plotly) => {
    if (!alive) return;
    const fig = JSON.parse(JSON.stringify(figure));
    Plotly.newPlot(div, fig.data, { ...fig.layout, autosize: true }, CONFIG).then(() => shareStyles(div));
    const off = ctrlWheelZoom(div, Plotly);
    const ro = new ResizeObserver(() => Plotly.Plots.resize(div));
    ro.observe(div);
    undo = () => { off(); ro.disconnect(); Plotly.purge(div); };
  });
  return () => { alive = false; undo(); };
}

// PNG export: each figure is drawn off-screen, then laid out on one canvas.
async function renderOne(figure, w, ht) {
  const Plotly = await loadPlotly();
  const fig = JSON.parse(JSON.stringify(figure));
  const url = await Plotly.toImage({ data: fig.data, layout: { ...fig.layout, width: w, height: ht, paper_bgcolor: "#fff" } },
    { format: "png", width: w, height: ht, scale: 2 });
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

async function exportPng({ figures, cols = 1, width = 820, height = 300, title = "" }) {
  const imgs = await Promise.all(figures.map((f) => renderOne(f, width, height)));
  const rows = Math.ceil(imgs.length / cols);
  const gap = 16, top = title ? 46 : 12, px = 2;
  const W = cols * width + (cols - 1) * gap + 24, H = top + rows * height + (rows - 1) * gap + 12;
  const canvas = document.createElement("canvas");
  canvas.width = W * px;
  canvas.height = H * px;
  const ctx = canvas.getContext("2d");
  ctx.scale(px, px);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, W, H);
  if (title) {
    ctx.fillStyle = "#172033";
    ctx.font = `600 17px ${FONT}`;
    ctx.fillText(title, 14, 30);
  }
  imgs.forEach((img, i) => {
    const r = Math.floor(i / cols), c = i % cols;
    ctx.drawImage(img, 12 + c * (width + gap), top + r * (height + gap), width, height);
  });
  return new Promise((res) => canvas.toBlob((b) => res(b), "image/png"));
}

function saveBlob(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ---------------------------------------------------------------------------
// Modal dialog (Enlarge, About, Model scheme)
// ---------------------------------------------------------------------------
function openModal(host, title, wide, fill) {
  const back = h(`<div class="modal-back"><div class="modal ${wide ? "wide" : ""}" role="dialog" aria-label="${esc(title)}">
    <div class="modal-head"><h2>${esc(title)}</h2><button class="icon-btn" aria-label="Close">${svg('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>', 18)}</button></div>
    <div class="modal-body"></div></div></div>`);
  let cleanup = null;
  const close = () => { if (cleanup) cleanup(); back.remove(); window.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  back.addEventListener("click", (e) => { if (e.target === back) close(); });
  back.querySelector(".icon-btn").onclick = close;
  window.addEventListener("keydown", onKey);
  host.appendChild(back);
  cleanup = fill(back.querySelector(".modal-body"));
}

// ---------------------------------------------------------------------------
// One render function per kind
// ---------------------------------------------------------------------------
const KINDS = {
  modes(model, el) {
    const p = model.get("props");
    const wrap = h(`<div class="modes" role="tablist"></div>`);
    for (const o of p.options) {
      const active = o.value === model.get("value");
      const b = h(`<button role="tab" class="mode-card ${active ? "active" : ""}" data-mode="${o.value}">
        <span class="bar"></span>
        <span class="mode-icon">${svg(o.icon, 18)}</span>
        <span class="mode-text"><span class="kicker">Mode ${o.n}${o.soon ? " · soon" : ""}</span>
        <span class="title">${esc(o.title)}</span><span class="caption">${esc(o.caption)}</span></span></button>`);
      b.onclick = () => { model.set("value", o.value); model.save_changes(); };
      wrap.appendChild(b);
    }
    el.appendChild(wrap);
  },

  radiocards(model, el) {
    const wrap = h(`<div class="radiocards" role="radiogroup"></div>`);
    for (const o of model.get("props").options) {
      const active = o.value === model.get("value");
      const b = h(`<button class="radio-card ${active ? "active" : ""}"><span class="dot"><span></span></span>
        <span class="txt"><span class="label">${esc(o.label)}</span>${o.caption ? `<span class="caption">${esc(o.caption)}</span>` : ""}</span></button>`);
      b.onclick = () => { model.set("value", o.value); model.save_changes(); };
      wrap.appendChild(b);
    }
    el.appendChild(wrap);
  },

  segmented(model, el) {
    const p = model.get("props");
    const wrap = h(`<div class="segmented ${p.size === "sm" ? "sm" : ""} ${p.full ? "full" : ""}" role="tablist"></div>`);
    for (const o of p.options) {
      const b = h(`<button role="tab" class="${o.value === model.get("value") ? "active" : ""}">${esc(o.label)}</button>`);
      b.onclick = () => { model.set("value", o.value); model.save_changes(); };
      wrap.appendChild(b);
    }
    el.appendChild(wrap);
  },

  tabs(model, el) {
    const wrap = h(`<div class="tabs" role="tablist"></div>`);
    for (const o of model.get("props").options) {
      const active = o.value === model.get("value");
      const badge = o.badge === "dot" ? `<span class="badge-dot"></span>` : o.badge ? `<span class="badge">${esc(o.badge)}</span>` : "";
      const b = h(`<button role="tab" class="${active ? "active" : ""}">${o.icon ? svg(o.icon) : ""}${esc(o.label)}${badge}</button>`);
      b.onclick = () => { model.set("value", o.value); model.save_changes(); };
      wrap.appendChild(b);
    }
    el.appendChild(wrap);
  },

  toggle(model, el) {
    const p = model.get("props");
    const on = !!model.get("value");
    const lab = h(`<label class="toggle ${p.disabled ? "disabled" : ""}"><input type="checkbox" ${on ? "checked" : ""} ${p.disabled ? "disabled" : ""}>
      <span class="track"><span class="knob"></span></span>
      <span><span class="label">${esc(p.label)}</span>${p.hint ? `<span class="hint">${esc(p.hint)}</span>` : ""}</span></label>`);
    lab.querySelector("input").onchange = (e) => { model.set("value", e.target.checked); model.save_changes(); };
    el.appendChild(lab);
  },

  button(model, el) {
    const p = model.get("props");
    const b = h(`<button class="btn ${p.variant || "secondary"} ${p.size === "sm" ? "sm" : ""} ${p.full ? "full" : ""}" ${p.disabled ? "disabled" : ""} ${p.title ? `title="${esc(p.title)}"` : ""}>${p.icon ? svg(p.icon, p.size === "sm" ? 14 : 16) : ""}${esc(p.label)}</button>`);
    b.onclick = () => { model.set("clicks", model.get("clicks") + 1); model.save_changes(); };
    el.appendChild(b);
  },

  slider(model, el) {
    const p = model.get("props");
    const fmt = (v) => p.fmt === "pct" ? `${v}%` : p.fmt === "days" ? `${v} day${v === 1 ? "" : "s"}` : String(v);
    const v0 = model.get("value");
    const pct = (v) => (p.max > p.min ? ((v - p.min) / (p.max - p.min)) * 100 : 0);
    const w = h(`<div class="slider"><div class="val">${fmt(v0)}</div><div class="rail"><div class="track"></div><div class="fill" style="width:${pct(v0)}%"></div>
      <input type="range" min="${p.min}" max="${p.max}" step="${p.step || 1}" value="${v0}"></div>
      <div class="ends"><span>${fmt(p.min)}</span><span>${fmt(p.max)}</span></div></div>`);
    const input = w.querySelector("input");
    input.oninput = () => { w.querySelector(".val").textContent = fmt(+input.value); w.querySelector(".fill").style.width = `${pct(+input.value)}%`; };
    input.onchange = () => { model.set("value", +input.value); model.save_changes(); };
    el.appendChild(w);
  },

  range(model, el) {
    const p = model.get("props");
    let [a, b] = model.get("value");
    const lab = (i) => p.labels[i];
    const pc = (v) => ((v - p.min) / (p.max - p.min)) * 100;
    const w = h(`<div class="slider"><div class="vals"><span></span><span></span></div><div class="rail dual"><div class="track"></div><div class="fill"></div>
      <input type="range" aria-label="Start" min="${p.min}" max="${p.max}" value="${a}"><input type="range" aria-label="End" min="${p.min}" max="${p.max}" value="${b}"></div></div>`);
    const [i0, i1] = w.querySelectorAll("input");
    const draw = () => {
      const s = w.querySelectorAll(".vals span");
      s[0].textContent = lab(a); s[1].textContent = lab(b);
      const f = w.querySelector(".fill");
      f.style.left = `${pc(a)}%`; f.style.width = `${pc(b) - pc(a)}%`;
    };
    i0.oninput = () => { a = Math.min(+i0.value, b - 1); i0.value = a; draw(); };
    i1.oninput = () => { b = Math.max(+i1.value, a + 1); i1.value = b; draw(); };
    const commit = () => { model.set("value", [a, b]); model.save_changes(); };
    i0.onchange = commit; i1.onchange = commit;
    draw();
    el.appendChild(w);
  },

  date(model, el) {
    const p = model.get("props");
    const i = h(`<input class="input" type="date" value="${model.get("value")}" min="${p.min}" max="${p.max}">`);
    i.onchange = () => { if (i.value && i.value >= p.min && i.value <= p.max) { model.set("value", i.value); model.save_changes(); } };
    el.appendChild(i);
  },

  text(model, el) {
    const i = h(`<input class="input" type="text">`);
    i.value = model.get("value");
    i.onchange = () => { model.set("value", i.value); model.save_changes(); };
    el.appendChild(i);
  },

  select(model, el) {
    const s = h(`<select class="input">${model.get("props").options.map((o) => `<option ${o === model.get("value") ? "selected" : ""}>${esc(o)}</option>`).join("")}</select>`);
    s.onchange = () => { model.set("value", s.value); model.save_changes(); };
    el.appendChild(s);
  },

  checklist(model, el) {
    const val = model.get("value") || [];
    const wrap = h(`<div class="checklist"></div>`);
    for (const o of model.get("props").options) {
      const on = val.includes(o);
      const b = h(`<button class="${on ? "on" : ""}"><span class="box">${on ? svg('<path d="M20 6 9 17l-5-5"/>', 10) : ""}</span><span class="t">${esc(o)}</span></button>`);
      b.onclick = () => {
        const cur = model.get("value") || [];
        const next = cur.includes(o) ? cur.filter((x) => x !== o) : [...cur, o];
        model.set("value", model.get("props").options.filter((x) => next.includes(x)));
        model.save_changes();
      };
      wrap.appendChild(b);
    }
    el.appendChild(wrap);
  },

  chart(model, el) {
    const p = model.get("props");
    const w = h(`<div class="chart"><div class="canvas" style="height:${p.height || 260}px"></div>
      <button class="enlarge" aria-label="Enlarge plot" title="Enlarge plot">${svg('<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>', 14)}Enlarge</button></div>`);
    el.appendChild(w);
    const cleanup = mountChart(w.querySelector(".canvas"), p.figure);
    w.querySelector(".enlarge").onclick = () => openModal(el, "Enlarged plot", true, (body) => {
      const div = h(`<div style="height:${Math.min(620, Math.max(420, window.innerHeight - 240))}px"></div>`);
      body.appendChild(div);
      body.appendChild(h(`<p class="hint" style="margin-top:8px">${ZOOM_HINT}</p>`));
      return mountChart(div, p.figure);
    });
    return cleanup;
  },

  download(model, el) {
    const p = model.get("props");
    const b = h(`<button class="btn ${p.variant || "secondary"}" ${p.title ? `title="${esc(p.title)}"` : ""}>${p.icon ? svg(p.icon) : ""}<span>${esc(p.label)}</span></button>`);
    b.onclick = async () => {
      const label = b.querySelector("span");
      b.disabled = true;
      label.textContent = "Preparing…";
      try {
        const files = { ...(p.files || {}) };
        for (const img of p.images || []) files[img.name] = await exportPng(img);
        const names = Object.keys(files);
        if (!p.zip && names.length === 1) {
          const f = files[names[0]];
          saveBlob(f instanceof Blob ? f : new Blob([f], { type: "text/csv" }), names[0]);
        } else {
          const { default: JSZip } = await import("https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm");
          const zip = new JSZip();
          for (const [n, f] of Object.entries(files)) zip.file(n, f);
          saveBlob(await zip.generateAsync({ type: "blob" }), p.filename);
        }
      } finally {
        b.disabled = false;
        label.textContent = p.label;
      }
    };
    el.appendChild(b);
  },

  header(model, el) {
    const p = model.get("props");
    const bar = h(`<header class="topbar"><div class="hero-line"></div><div class="topbar-in">
      <img src="${p.logo}" alt="" class="logo">
      <div class="brand"><div class="name">COVID-19 Model Simulator</div><div class="sub">EUVABECO · CEMAT · Portugal 2020</div></div>
      <nav>
        <button class="top-btn about">${svg(p.icons.info)}<span>About</span></button>
        <button class="top-btn scheme">${svg(p.icons.sigma)}<span>Model scheme</span></button>
        <a class="top-btn pdf" href="${p.pdf}" download>${svg(p.icons.pdf)}<span>Equations (PDF)</span></a>
        <a class="top-btn square" href="${p.repo}" target="_blank" rel="noreferrer" aria-label="Source code on GitHub">${svg(p.icons.repo)}</a>
      </nav></div></header>`);
    bar.querySelector(".about").onclick = () => openModal(el, "Welcome to the Online COVID-19 Simulator", false, (body) => { body.innerHTML = p.about; });
    bar.querySelector(".scheme").onclick = () => openModal(el, "Model scheme and equations", true, (body) => { body.innerHTML = p.scheme; });
    el.appendChild(bar);
  },
};

function render({ model, el }) {
  let cleanup = null;
  const draw = () => {
    if (cleanup) cleanup();
    el.innerHTML = "";
    cleanup = KINDS[model.get("kind")](model, el) || null;
  };
  draw();
  model.on("change:props", draw);
  model.on("change:value", draw);
  return () => { if (cleanup) cleanup(); };
}

export default { render };
