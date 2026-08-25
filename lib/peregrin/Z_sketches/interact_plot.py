"""
Shiny for Python app that embeds the matplotlib-SVG + click-tooltip trajectory
editor built earlier, running live inside the app via an <iframe srcdoc="...">.

Why an iframe:
  - Shiny/Bootstrap load their own JS and CSS. Injecting the tooltip's raw
    <script>/<style> directly into the page risks id/variable collisions
    (e.g. a global `activeElement`, a `#tooltip` id already used elsewhere).
  - An iframe with `srcdoc` gives the embedded HTML its own isolated
    document, window, and DOM -- it behaves exactly like the standalone
    file from before, just rendered inside a panel of the Shiny app.

Run:
    pip install shiny matplotlib numpy
    shiny run --reload shiny_embedded_tooltip_app.py
"""

import io
import html as html_escape
import numpy as np
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from shiny import App, ui


# ---------------------------------------------------------------------------
# 1. Build the matplotlib SVG with gid-tagged trajectories (same as before).
# ---------------------------------------------------------------------------
def build_trajectory_svg() -> str:
    rng = np.random.default_rng(0)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_title("Click a track to customize it")

    for i, color in enumerate(palette):
        n = 60
        x = np.cumsum(rng.normal(size=n))
        y = np.cumsum(rng.normal(size=n))

        (line,) = ax.plot(x, y, color=color, linewidth=2)
        line.set_gid(f"track-line-{i}")

        head = ax.scatter([x[-1]], [y[-1]], color=color, s=80, zorder=3)
        head.set_gid(f"track-head-{i}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    buf = io.StringIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 2. Wrap the SVG in a *complete* standalone HTML document. This whole
#    string becomes the iframe's srcdoc, so it needs its own <html>/<head>.
# ---------------------------------------------------------------------------
def build_editor_html(svg_markup: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0.5rem; }}
  #plot-wrapper {{ position: relative; display: inline-block; }}
  svg g[id^="track-"] {{ cursor: pointer; }}
  #tooltip {{
    position: absolute;
    display: none;
    background: white;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    padding: 12px 14px;
    font-size: 13px;
    z-index: 10;
    min-width: 180px;
  }}
  #tooltip h4 {{ margin: 0 0 8px 0; font-size: 13px; }}
  #tooltip label {{ display: block; margin-top: 8px; }}
  #tooltip input[type="range"] {{ width: 100%; }}
  #tooltip .close-btn {{
    position: absolute; top: 6px; right: 8px; cursor: pointer;
    border: none; background: none; font-size: 14px; color: #888;
  }}
</style>
</head>
<body>

<div id="plot-wrapper">
  {svg_markup}
  <div id="tooltip">
    <button class="close-btn" onclick="closeTooltip()">&times;</button>
    <h4 id="tooltip-title">Track</h4>
    <label>Color
      <input type="color" id="color-input">
    </label>
    <label>Line width
      <input type="range" id="width-input" min="1" max="10" step="0.5">
    </label>
  </div>
</div>

<script>
let activeElement = null;

function rgbToHex(rgb) {{
  const m = rgb.match(/\\d+/g);
  if (!m) return rgb;
  return "#" + m.slice(0,3).map(n => parseInt(n).toString(16).padStart(2,"0")).join("");
}}

function openTooltip(evt, el, label) {{
  activeElement = el;
  const tooltip = document.getElementById("tooltip");
  const wrapperRect = document.getElementById("plot-wrapper").getBoundingClientRect();

  tooltip.style.left = (evt.clientX - wrapperRect.left + 12) + "px";
  tooltip.style.top  = (evt.clientY - wrapperRect.top + 12) + "px";
  tooltip.style.display = "block";
  document.getElementById("tooltip-title").textContent = label;

  const cs = getComputedStyle(el);
  const currentColor = cs.stroke;
  document.getElementById("color-input").value = rgbToHex(currentColor);
  document.getElementById("width-input").value = parseFloat(cs.strokeWidth) || 2;
}}

function closeTooltip() {{
  document.getElementById("tooltip").style.display = "none";
  activeElement = null;
}}

document.getElementById("color-input").addEventListener("input", (e) => {{
  if (!activeElement) return;
  activeElement.style.stroke = e.target.value;
}});

document.getElementById("width-input").addEventListener("input", (e) => {{
  if (!activeElement) return;
  activeElement.style.strokeWidth = e.target.value;
}});

window.addEventListener("DOMContentLoaded", () => {{
  const svg = document.querySelector("#plot-wrapper svg");
  const groups = svg.querySelectorAll('g[id^="track-line-"], g[id^="track-head-"]');

  groups.forEach((g) => {{
    const clickable = g.querySelector("path") || g;
    clickable.addEventListener("click", (evt) => {{
      evt.stopPropagation();
      const label = g.id.replace("track-line-", "Track ").replace("track-head-", "Track ")
                    + " (" + (g.id.includes("head") ? "head marker" : "line") + ")";
      openTooltip(evt, clickable, label);
    }});
  }});

  document.body.addEventListener("click", (evt) => {{
    if (!evt.target.closest("#tooltip") && !evt.target.closest("g[id^='track-']")) {{
      closeTooltip();
    }}
  }});
}});
</script>

</body>
</html>
"""


SVG_MARKUP = build_trajectory_svg()
EDITOR_HTML = build_editor_html(SVG_MARKUP)


# ---------------------------------------------------------------------------
# 3. Shiny app: just an iframe hosting the editor, sized to fit.
# ---------------------------------------------------------------------------
app_ui = ui.page_fillable(
    ui.h2("Cell Trajectories"),
    ui.p("Click any line or marker to edit its color and width, right here in the app."),
    ui.tags.iframe(
        srcdoc=EDITOR_HTML,
        style="width: 100%; height: 650px; border: none;",
    ),
)


def server(input, output, session):
    pass  # everything interactive lives inside the iframe's own JS


app = App(app_ui, server)