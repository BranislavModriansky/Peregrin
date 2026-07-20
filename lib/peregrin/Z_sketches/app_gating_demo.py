from shiny import App, ui, render
import pandas as pd
import numpy as np
import json
from textwrap import dedent

CSV_PATH = r"C:\Users\modri\Desktop\Lab\Peregrin project\Data\mem_cxcl12\#memory-cxcl12#BC39#Stablized_spots.csv"

# Load once
_df = pd.read_csv(CSV_PATH)
_numeric_cols = [c for c in _df.columns if pd.api.types.is_numeric_dtype(_df[c])]

# ---- Head script: defines window.renderGate() and auto-inits new .gate-root elements
HEAD_JS = ui.tags.script(dedent("""
(function(){
  function initGate(el) {
    if (!el || el.dataset.inited === "1") return;
    el.dataset.inited = "1";

    const payload = JSON.parse(el.dataset.payload || "{}");
    const W = payload.width, H = payload.height, PAD = payload.pad;
    const xs = payload.x || [], ys = payload.y || [];
    const xmin = payload.bounds?.xmin ?? 0, xmax = payload.bounds?.xmax ?? 1;
    const ymin = payload.bounds?.ymin ?? 0, ymax = payload.bounds?.ymax ?? 1;
    const xlog = payload.log?.x || false;
    const ylog = payload.log?.y || false;

    const labelEl = el.querySelector(".axis-labels");
    if (labelEl && payload.labels) {
      labelEl.textContent = payload.labels.x + " vs " + payload.labels.y;
    }
    const canvas = el.querySelector("canvas");
    if (!canvas) return;
    canvas.width = W; canvas.height = H;
    const ctx = canvas.getContext("2d");

    const sx = x => PAD + (x - xmin) * (W - 2*PAD) / (xmax - xmin || 1);
    const sy = y => H - PAD - (y - ymin) * (H - 2*PAD) / (ymax - ymin || 1);
    const invx = X => xmin + (X - PAD) * (xmax - xmin || 1) / (W - 2*PAD);
    const invy = Y => ymin + (H - PAD - Y) * (ymax - ymin || 1) / (H - 2*PAD);
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

    let poly = [];
    let closed = false;
    let selectedIdx = new Set();

    // vertex drag state
    const VERT_R = 6;
    let draggingVert = -1;
    let dragging = false;

    function drawAll() {
      ctx.clearRect(0,0,W,H);

      // border
      ctx.strokeStyle = '#e0e0e0';
      ctx.lineWidth = 1;
      ctx.strokeRect(PAD, PAD, W-2*PAD, H-2*PAD);

      // ticks (labels show raw values when in log mode)
      ctx.fillStyle = '#666';
      ctx.font = '12px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif';
      const xticks = 5, yticks = 5;
      for (let i=0;i<=xticks;i++) {
        const xv = xmin + i*(xmax-xmin)/xticks;  // value in plot space (log if log)
        const X = sx(xv);
        ctx.beginPath(); ctx.moveTo(X, H-PAD); ctx.lineTo(X, H-PAD+4); ctx.stroke();
        const raw = xlog ? Math.pow(10, xv) : xv;
        const label = xlog ? (raw >= 1e-3 && raw < 1e4 ? raw.toPrecision(3) : raw.toExponential(1))
                           : xv.toFixed(1);
        ctx.fillText(label, X-12, H-PAD+16);
      }
      for (let i=0;i<=yticks;i++) {
        const yv = ymin + i*(ymax-ymin)/yticks;
        const Y = sy(yv);
        ctx.beginPath(); ctx.moveTo(PAD-4, Y); ctx.lineTo(PAD, Y); ctx.stroke();
        const raw = ylog ? Math.pow(10, yv) : yv;
        const label = ylog ? (raw >= 1e-3 && raw < 1e4 ? raw.toPrecision(3) : raw.toExponential(1))
                           : yv.toFixed(1);
        ctx.fillText(label, 2, Y+4);
      }

      // points (xs/ys are already log10 if log is on)
      ctx.globalAlpha = 0.9;
      for (let i=0;i<xs.length;i++) {
        const X = sx(xs[i]), Y = sy(ys[i]);
        ctx.beginPath();
        ctx.arc(X, Y, selectedIdx.has(i) ? 2.2 : 1.6, 0, Math.PI*2);
        ctx.fillStyle = selectedIdx.has(i) ? '#e74c3c' : 'rgba(33,150,243,0.55)';
        ctx.fill();
      }

      // polygon
      if (poly.length) {
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#34495e';
        ctx.fillStyle = 'rgba(52,152,219,0.12)';
        ctx.beginPath();
        ctx.moveTo(poly[0].X, poly[0].Y);
        for (let i=1;i<poly.length;i++) ctx.lineTo(poly[i].X, poly[i].Y);
        if (closed) ctx.closePath();
        ctx.stroke();
        if (closed) ctx.fill();

        // vertices (constant style)
        ctx.fillStyle = '#2c3e50';
        for (const v of poly) {
          ctx.beginPath();
          ctx.arc(v.X, v.Y, 3, 0, Math.PI*2);
          ctx.fill();
        }
      }
    }

    function pip(X, Y, poly) {
      let inside = false;
      for (let i=0, j=poly.length-1; i<poly.length; j=i++) {
        const xi = poly[i].X, yi = poly[i].Y;
        const xj = poly[j].X, yj = poly[j].Y;
        const intersect = ((yi>Y)!=(yj>Y)) && (X < (xj-xi)*(Y-yi)/(yj-yi+1e-12)+xi);
        if (intersect) inside = !inside;
      }
      return inside;
    }

    function recomputeSelection() {
      selectedIdx.clear();
      if (!closed || poly.length < 3) return;
      for (let i=0;i<xs.length;i++) {
        const X = sx(xs[i]), Y = sy(ys[i]);
        if (pip(X, Y, poly)) selectedIdx.add(i);
      }
    }

    function sendToShiny() {
      if (!(window.Shiny && Shiny.setInputValue)) return;
      if (!poly.length) {
        Shiny.setInputValue('gates', { xs: [], ys: [] }, {priority:'event'});
        Shiny.setInputValue('gate_selection_n', 0, {priority:'event'});
        return;
      }
      // Get polygon vertices in plot coords
      let xsP = poly.map(p => invx(p.X));
      let ysP = poly.map(p => invy(p.Y));
      // Convert back to RAW (un-logged) values for Shiny
      if (xlog) xsP = xsP.map(v => Math.pow(10, v));
      if (ylog) ysP = ysP.map(v => Math.pow(10, v));

      Shiny.setInputValue('gates', { xs: [xsP], ys: [ysP] }, {priority:'event'});
      Shiny.setInputValue('gate_selection_n', selectedIdx.size, {priority:'event'});
    }

    function pickVertex(X, Y) {
      for (let i=0;i<poly.length;i++) {
        const dx = poly[i].X - X;
        const dy = poly[i].Y - Y;
        if (dx*dx + dy*dy <= VERT_R*VERT_R) return i;
      }
      return -1;
    }

    let lastClickTime = 0;
    canvas.addEventListener('click', (ev) => {
      if (closed) return;
      const rect = canvas.getBoundingClientRect();
      const X = ev.clientX - rect.left, Y = ev.clientY - rect.top;
      const now = Date.now();
      const isDbl = (now - lastClickTime) < 280;
      lastClickTime = now;
      if (isDbl && poly.length >= 3) {
        closed = true;
        recomputeSelection();
        sendToShiny();
        drawAll();
        return;
      }
      poly.push({X, Y});
      drawAll();
    });

    canvas.addEventListener('mousedown', (ev) => {
      if (!closed) return;
      const rect = canvas.getBoundingClientRect();
      const X = ev.clientX - rect.left, Y = ev.clientY - rect.top;
      const idx = pickVertex(X, Y);
      if (idx >= 0) {
        draggingVert = idx;
        dragging = true;
        canvas.style.cursor = 'grabbing';
      }
    });

    window.addEventListener('mousemove', (ev) => {
      const rect = canvas.getBoundingClientRect();
      const X = ev.clientX - rect.left, Y = ev.clientY - rect.top;

      if (closed) {
        if (dragging && draggingVert >= 0) {
          poly[draggingVert].X = clamp(X, PAD, W - PAD);
          poly[draggingVert].Y = clamp(Y, PAD, H - PAD);
          recomputeSelection();
          drawAll();
          sendToShiny();
        } else {
          const idx = pickVertex(X, Y);
          canvas.style.cursor = (idx >= 0) ? 'grab' : 'default';
        }
      } else {
        canvas.style.cursor = 'crosshair';
      }
    });

    window.addEventListener('mouseup', () => {
      if (dragging) {
        dragging = false;
        draggingVert = -1;
        canvas.style.cursor = closed ? 'default' : 'crosshair';
      }
    });

    document.addEventListener('shiny:inputchanged', (e) => {
      if (e.detail.name === 'reset') {
        poly = [];
        closed = false;
        selectedIdx.clear();
        draggingVert = -1; dragging = false;
        canvas.style.cursor = 'crosshair';
        sendToShiny();
        drawAll();
      }
    });

    drawAll();
    sendToShiny();
  }

  // Auto-init
  const obs = new MutationObserver((mut) => {
    for (const m of mut) {
      for (const el of m.addedNodes) {
        if (!(el instanceof HTMLElement)) continue;
        if (el.matches && el.matches('.gate-root')) initGate(el);
        for (const child of el.querySelectorAll ? el.querySelectorAll('.gate-root') : []) {
          initGate(child);
        }
      }
    }
  });
  obs.observe(document.documentElement || document.body, { childList: true, subtree: true });

  window.addEventListener('load', () => {
    document.querySelectorAll('.gate-root').forEach(initGate);
  });
})();
"""))




app_ui = ui.page_fluid(
    ui.head_content(HEAD_JS),
    ui.h3("Polygon gating on CSV data (Canvas, no external JS libs)"),
    ui.row(
        ui.column(3,
            ui.input_select("xcol", "X axis", choices=_numeric_cols, selected=_numeric_cols[0] if _numeric_cols else None),
            ui.input_select("ycol", "Y axis", choices=_numeric_cols, selected=_numeric_cols[1] if len(_numeric_cols)>1 else (_numeric_cols[0] if _numeric_cols else None)),
            ui.input_action_button("reset", "Reset polygon", class_="btn-secondary"),
            ui.hr(),
            ui.input_checkbox("xlog", "Log scale (X)", value=False),
            ui.input_checkbox("ylog", "Log scale (Y)", value=False),
            ui.hr(),
            ui.h5("Selected points:"),
            ui.output_text_verbatim("sel_count", placeholder=True),
            ui.hr(),
            ui.h5("Gate coordinates (xs/ys):"),
            ui.output_text_verbatim("gate_txt", placeholder=True),
        ),
        ui.column(9,
            ui.output_ui("gate_plot")
        )
    ),
    ui.tags.style(dedent("""
        .gate-canvas { border: 1px solid #ddd; border-radius: 6px; cursor: crosshair; }
        .note { color: #666; font-size: 0.9em; }
    """))
)

def server(input, output, session):

    def mk_payload():
        xcol = input.xcol()
        ycol = input.ycol()
        xlog = bool(input.xlog())
        ylog = bool(input.ylog())

        if not xcol or not ycol:
            return {
                "x": [], "y": [],
                "bounds": {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1},
                "width": 720, "height": 520, "pad": 36,
                "labels": {"x": xcol or "", "y": ycol or ""},
                "log": {"x": xlog, "y": ylog},
            }

        # Finite-only; for log, require > 0
        xraw = _df[xcol].astype(float).replace([np.inf, -np.inf], np.nan)
        yraw = _df[ycol].astype(float).replace([np.inf, -np.inf], np.nan)
        if xlog: xraw = xraw.where(xraw > 0)
        if ylog: yraw = yraw.where(yraw > 0)

        xy = [(a, b) for a, b in zip(xraw, yraw) if pd.notna(a) and pd.notna(b)]
        if not xy:
            # default 1..10 decades in log space, or 0..1 linear
            if xlog:
                xmin, xmax = 0.0, 1.0  # log10 1..10
            else:
                xmin, xmax = 0.0, 1.0
            if ylog:
                ymin, ymax = 0.0, 1.0
            else:
                ymin, ymax = 0.0, 1.0
            x2, y2 = [], []
        else:
            xs_lin = [float(a) for a, _ in xy]
            ys_lin = [float(b) for _, b in xy]
            x2 = [np.log10(v) if xlog else v for v in xs_lin]
            y2 = [np.log10(v) if ylog else v for v in ys_lin]

            xmin, xmax = float(min(x2)), float(max(x2))
            ymin, ymax = float(min(y2)), float(max(y2))

            # Nice bounds for log: full decades
            if xlog:
                xmin, xmax = np.floor(xmin), np.ceil(xmax)
            if ylog:
                ymin, ymax = np.floor(ymin), np.ceil(ymax)
            # Avoid zero span
            if xmin == xmax: xmax = xmin + 1e-6
            if ymin == ymax: ymax = ymin + 1e-6

        return {
            "x": x2, "y": y2,  # already in log10 when xlog/ylog True
            "bounds": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
            "width": 720, "height": 520, "pad": 36,
            "labels": {"x": xcol, "y": ycol},
            "log": {"x": xlog, "y": ylog},
        }



    @output
    @render.ui
    def gate_plot():
        payload = mk_payload()
        # NOTE: no <script> tag here. Just a container with data attributes.
        html = f'''
          <div class="gate-root" data-payload='{json.dumps(payload)}'>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
              <strong class="axis-labels"></strong>
              <div class="note">Click to add vertices; double-click to close polygon.</div>
            </div>
            <canvas class="gate-canvas"></canvas>
          </div>
        '''
        return ui.HTML(html)

    @output
    @render.text
    def gate_txt():
        g = input.gates()
        if not g or not g.get("xs"):
            return "No gates yet."
        xs = g['xs'][0] if g['xs'] else []
        ys = g['ys'][0] if g['ys'] else []
        return f"Gate 1 (N={len(xs)}):\n  xs={xs}\n  ys={ys}"

    @output
    @render.text
    def sel_count():
        n = input.gate_selection_n() or 0
        return f"{n} points inside gate"

app = App(app_ui, server)
