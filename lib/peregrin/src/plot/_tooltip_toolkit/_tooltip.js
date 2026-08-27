(function () {
  const S = window.PEREGRIN_STATE;
  if (!S) return;

  // ---- inject CSS once ----
  if (!document.getElementById('peregrin-tracks-style')) {
    const style = document.createElement('style');
    style.id = 'peregrin-tracks-style';
    style.textContent = "__PEREGRIN_CSS__";
    document.head.appendChild(style);
  }

  // ---- build DOM ----
  const ROOT = document.createElement('div');
  ROOT.className = 'peregrin-tracks';

  const cv = document.createElement('canvas');
  cv.className = 'pg-canvas';
  cv.width = 800;
  cv.height = 800;

  const tip = document.createElement('div');
  tip.className = 'pg-tooltip';

  ROOT.appendChild(cv);
  ROOT.appendChild(tip);
  (document.currentScript ? document.currentScript.parentElement : document.body).appendChild(ROOT);

  const ctx = cv.getContext('2d');
  const AX = (!S.polar && S.axes) ? S.axes : null;

  /*
   * The canvas is only the outer container. The actual axes rectangle is
   * calculated later from the x/y data ranges so one data unit has the same
   * pixel size in both directions.
   */
  const AXES_AREA = AX
    ? {
        x0: 92,
        y0: 58,
        x1: cv.width - 24,
        y1: cv.height - 82
      }
    : {
        x0: 40,
        y0: 40,
        x1: cv.width - 40,
        y1: cv.height - 40
      };

  function bounds() {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const [xs, ys] of S.tracks) {
      for (let i = 0; i < xs.length; i++) {
        let x = xs[i], y = ys[i];
        if (S.polar) {
          const r = y, t = xs[i];
          x = r * Math.cos(t);
          y = r * Math.sin(t);
        }
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
    }

    if (!isFinite(xmin)) {
      xmin = 0;
      xmax = 1;
      ymin = 0;
      ymax = 1;
    }

    return { xmin, xmax, ymin, ymax };
  }

  const B = bounds();
  const XDOM = AX ? AX.xlim : [B.xmin, B.xmax];
  const YDOM = AX ? AX.ylim : [B.ymin, B.ymax];

  const XSPAN = Math.abs(XDOM[1] - XDOM[0]) || 1;
  const YSPAN = Math.abs(YDOM[1] - YDOM[0]) || 1;

  /**
   * Fit an axes rectangle with the data aspect ratio inside AXES_AREA.
   *
   * For example, a data range twice as wide as it is high produces an axes
   * rectangle twice as wide as it is high. Remaining canvas space is left
   * blank and distributed evenly around the axes.
   */
  function fitAxesToData() {
    const availableWidth = AXES_AREA.x1 - AXES_AREA.x0;
    const availableHeight = AXES_AREA.y1 - AXES_AREA.y0;
    const dataRatio = XSPAN / YSPAN;
    const availableRatio = availableWidth / availableHeight;

    let width;
    let height;

    if (dataRatio >= availableRatio) {
      width = availableWidth;
      height = width / dataRatio;
    } else {
      height = availableHeight;
      width = height * dataRatio;
    }

    const x0 = AXES_AREA.x0 + (availableWidth - width) / 2;
    const y0 = AXES_AREA.y0 + (availableHeight - height) / 2;

    return {
      x0,
      y0,
      x1: x0 + width,
      y1: y0 + height,
      w: width,
      h: height
    };
  }

  const PLOT = fitAxesToData();

  function project(x, y) {
    if (S.polar) {
      const r = y, t = x;
      x = r * Math.cos(t);
      y = r * Math.sin(t);
    }

    const xFraction = (x - XDOM[0]) / (XDOM[1] - XDOM[0] || 1);
    const yFraction = (y - YDOM[0]) / (YDOM[1] - YDOM[0] || 1);

    return [
      PLOT.x0 + xFraction * PLOT.w,
      PLOT.y1 - yFraction * PLOT.h
    ];
  }

  function inRange(v, lo, hi) {
    const mn = Math.min(lo, hi), mx = Math.max(lo, hi);
    return v >= mn - 1e-9 && v <= mx + 1e-9;
  }

  function tickLabel(v) {
    if (!AX) return String(v);
    const d = Number.isFinite(AX.tickDecimals) ? AX.tickDecimals : 0;
    const x = Number(v).toFixed(d);
    return d === 0 ? String(Math.round(Number(x))) : x;
  }

  function drawGrid() {
    if (!S.style.showGrid) return;

    ctx.save();
    ctx.strokeStyle = S.style.gridColor;
    ctx.lineWidth = S.style.gridLw;

    if (AX) {
      const xtMinor = AX.xticksMinor || [];
      const ytMinor = AX.yticksMinor || [];
      for (const t of xtMinor) {
        if (!inRange(t, XDOM[0], XDOM[1])) continue;
        const x = project(t, YDOM[0])[0];
        ctx.beginPath(); ctx.moveTo(x, PLOT.y0); ctx.lineTo(x, PLOT.y1); ctx.stroke();
      }
      for (const t of ytMinor) {
        if (!inRange(t, YDOM[0], YDOM[1])) continue;
        const y = project(XDOM[0], t)[1];
        ctx.beginPath(); ctx.moveTo(PLOT.x0, y); ctx.lineTo(PLOT.x1, y); ctx.stroke();
      }

      const xtMajor = AX.xticksMajor || [];
      const ytMajor = AX.yticksMajor || [];
      for (const t of xtMajor) {
        if (!inRange(t, XDOM[0], XDOM[1])) continue;
        const x = project(t, YDOM[0])[0];
        ctx.beginPath(); ctx.moveTo(x, PLOT.y0); ctx.lineTo(x, PLOT.y1); ctx.stroke();
      }
      for (const t of ytMajor) {
        if (!inRange(t, YDOM[0], YDOM[1])) continue;
        const y = project(XDOM[0], t)[1];
        ctx.beginPath(); ctx.moveTo(PLOT.x0, y); ctx.lineTo(PLOT.x1, y); ctx.stroke();
      }
    } else {
      const n = 8;
      for (let i = 0; i <= n; i++) {
        const gx = PLOT.x0 + i * PLOT.w / n;
        const gy = PLOT.y0 + i * PLOT.h / n;
        ctx.beginPath(); ctx.moveTo(gx, PLOT.y0); ctx.lineTo(gx, PLOT.y1); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PLOT.x0, gy); ctx.lineTo(PLOT.x1, gy); ctx.stroke();
      }
    }

    ctx.restore();
  }

  function drawAxes() {
    if (!AX) return;

    const ann = S.style.annotationColor || '#000';
    const frame = S.style.frameColor || '#000';

    ctx.save();

    // frame / spines
    ctx.strokeStyle = frame;
    ctx.lineWidth = 1;
    ctx.strokeRect(PLOT.x0, PLOT.y0, PLOT.w, PLOT.h);

    // ticks + labels
    ctx.fillStyle = ann;
    ctx.strokeStyle = ann;
    ctx.lineWidth = 1;
    ctx.font = '12px system-ui, Arial, sans-serif';

    const xTicks = AX.xticksMajor || [];
    const yTicks = AX.yticksMajor || [];

    for (const t of xTicks) {
      if (!inRange(t, XDOM[0], XDOM[1])) continue;
      const x = project(t, YDOM[0])[0];
      ctx.beginPath(); ctx.moveTo(x, PLOT.y1); ctx.lineTo(x, PLOT.y1 + 6); ctx.stroke();
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(tickLabel(t), x, PLOT.y1 + 10);
    }

    for (const t of yTicks) {
      if (!inRange(t, YDOM[0], YDOM[1])) continue;
      const y = project(XDOM[0], t)[1];
      ctx.beginPath(); ctx.moveTo(PLOT.x0 - 6, y); ctx.lineTo(PLOT.x0, y); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(tickLabel(t), PLOT.x0 - 10, y);
    }

    // axis labels
    const textColor = S.style.textColor || '#000';
    ctx.fillStyle = textColor;
    ctx.font = '20px system-ui, Arial, sans-serif';

    const xlabel = AX.xlabel || '';
    const ylabel = AX.ylabel || '';
    if (xlabel) {
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(
        xlabel,
        PLOT.x0 + PLOT.w / 2,
        PLOT.y1 + 42
      );
    }

    if (ylabel) {
      ctx.save();
      ctx.translate(PLOT.x0 - 66, PLOT.y0 + PLOT.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(ylabel, 0, 0);
      ctx.restore();
    }

    const title = S.style.title || AX.title || '';
    if (title) {
      ctx.fillStyle = textColor;
      ctx.font = '22px system-ui, Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(
        title,
        PLOT.x0 + PLOT.w / 2,
        PLOT.y0 - 18
      );
    }

    ctx.restore();
  }

  function drawHead(px, py, color) {
    const s0 = Math.max(3, Math.sqrt(S.style.headSize));
    const s = Number.isFinite(s0) ? s0 : 3;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    switch (S.style.headShape) {
      case 's': ctx.rect(px - s, py - s, 2 * s, 2 * s); break;
      case '^': ctx.moveTo(px, py - s); ctx.lineTo(px - s, py + s); ctx.lineTo(px + s, py + s); ctx.closePath(); break;
      case 'v': ctx.moveTo(px, py + s); ctx.lineTo(px - s, py - s); ctx.lineTo(px + s, py - s); ctx.closePath(); break;
      case 'D': ctx.moveTo(px, py - s); ctx.lineTo(px - s, py); ctx.lineTo(px, py + s); ctx.lineTo(px + s, py); ctx.closePath(); break;
      case 'x':
        ctx.moveTo(px - s, py - s); ctx.lineTo(px + s, py + s);
        ctx.moveTo(px + s, py - s); ctx.lineTo(px - s, py + s);
        ctx.stroke(); return;
      case '*':
      case 'o':
      default:
        ctx.arc(px, py, s, 0, 2 * Math.PI);
    }
    ctx.stroke();
  }

  function render() {
    // background
    ctx.save();
    ctx.fillStyle = S.style.faceColor;
    ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.restore();

    drawGrid();

    // clip to plotting area like matplotlib axes box
    ctx.save();
    ctx.beginPath();
    ctx.rect(PLOT.x0, PLOT.y0, PLOT.w, PLOT.h);
    ctx.clip();

    ctx.lineWidth = Math.max(0.5, S.style.lw);
    S.tracks.forEach((tk, ti) => {
      const [xs, ys] = tk;
      ctx.strokeStyle = S.trackColors[ti] || '#000';
      ctx.beginPath();
      for (let i = 0; i < xs.length; i++) {
        const [px, py] = project(xs[i], ys[i]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
    });

    if (S.style.showHeads) {
      S.heads.forEach((h, i) => {
        const [px, py] = project(h[0], h[1]);
        drawHead(px, py, S.trackColors[i] || '#000');
      });
    }

    ctx.restore();

    drawAxes();
  }

  function distToSeg(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1, l2 = dx * dx + dy * dy;
    let t = l2 ? ((px - x1) * dx + (py - y1) * dy) / l2 : 0;
    t = Math.max(0, Math.min(1, t));
    const cx = x1 + t * dx, cy = y1 + t * dy;
    return Math.hypot(px - cx, py - cy);
  }

  function nearTrack(mx, my) {
    const tol = 6;
    for (const [xs, ys] of S.tracks) {
      for (let i = 1; i < xs.length; i++) {
        const a = project(xs[i - 1], ys[i - 1]);
        const b = project(xs[i], ys[i]);
        if (distToSeg(mx, my, a[0], a[1], b[0], b[1]) < tol) return true;
      }
    }
    return false;
  }

  function onGrid(mx, my) {
    if (!S.style.showGrid) return false;
    return mx > PLOT.x0 && mx < PLOT.x1 && my > PLOT.y0 && my < PLOT.y1;
  }

  function showTip(x, y, html) {
    tip.innerHTML = html;
    tip.style.left = Math.min(x, cv.width - 210) + 'px';
    tip.style.top = y + 'px';
    tip.style.display = 'block';
    tip.querySelectorAll('[data-k]').forEach(el => {
      el.addEventListener('input', () => {
        const k = el.dataset.k, grp = el.dataset.grp;
        let v = el.value;
        if (el.type === 'range') v = parseFloat(v);
        applyEdit(grp, k, v);
        render();
      });
    });
    const close = tip.querySelector('.pg-close');
    if (close) close.onclick = () => tip.style.display = 'none';
  }
  function row(label, input) { return `<div class="pg-row"><label>${label}</label><br>${input}</div>`; }
  function header(t) { return `<div class="pg-header">${t}<span class="pg-close">✕</span></div>`; }

  function tracksTip(x, y) {
    const sc = S.schema, cm = sc.color;
    let colorHtml = '';
    if (cm.mode === 'uniform') {
      colorHtml = row('Color (all tracks)', `<input type="color" data-grp="colorUniform" data-k="value" value="${cm.value.slice(0, 7)}">`);
    } else if (cm.mode === 'per_track') {
      const opts = cm.palettes.map(p => `<option ${p === cm.source ? 'selected' : ''}>${p}</option>`).join('');
      colorHtml = row('Palette (per-track)', `<select data-grp="colorPalette" data-k="source">${opts}</select>`);
    } else {
      const opts = cm.luts.map(p => `<option ${p === cm.source ? 'selected' : ''}>${p}</option>`).join('');
      colorHtml = row('LUT map', `<select data-grp="colorLut" data-k="source">${opts}</select>`);
    }
    const lw = sc.tracks.lw, hs = sc.tracks.headShape, hz = sc.tracks.headSize;
    showTip(x, y, header('Tracks') + colorHtml +
      row('Line width', `<input type="range" data-grp="tracks" data-k="lw" min="${lw.min}" max="${lw.max}" step="${lw.step}" value="${S.style.lw}">`) +
      row('Head shape', `<select data-grp="tracks" data-k="headShape">${hs.options.map(o => `<option ${o === S.style.headShape ? 'selected' : ''}>${o}</option>`).join('')}</select>`) +
      row('Head size', `<input type="range" data-grp="tracks" data-k="headSize" min="${hz.min}" max="${hz.max}" step="${hz.step}" value="${S.style.headSize}">`)
    );
  }

  function bgTip(x, y) {
    showTip(x, y, header('Background') +
      row('Backdrop color', `<input type="color" data-grp="background" data-k="faceColor" value="${S.style.faceColor.slice(0, 7)}">`));
  }

  function gridTip(x, y) {
    const g = S.schema.grid.gridLw;
    showTip(x, y, header('Grid') +
      row('Grid color', `<input type="color" data-grp="grid" data-k="gridColor" value="${S.style.gridColor.slice(0, 7)}">`) +
      row('Grid width', `<input type="range" data-grp="grid" data-k="gridLw" min="${g.min}" max="${g.max}" step="${g.step}" value="${S.style.gridLw}">`));
  }

  function rgb(r, g, b) { return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join(''); }
  function rnd() { return Math.floor(Math.random() * 256); }
  function randColor() { return rgb(rnd(), rnd(), rnd()); }
  const PALETTES = {
    'random': () => randColor(),
    'random greys': () => { const v = Math.floor(Math.random() * 180 + 40); return rgb(v, v, v); }
  };
  const LUTS = {
    viridis: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
    plasma: ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921'],
    inferno: ['#000004', '#57106e', '#bc3754', '#f98e09', '#fcffa4'],
    magma: ['#000004', '#51127c', '#b73779', '#fc8961', '#fcfdbf'],
    cividis: ['#00204d', '#414d6b', '#7c7b78', '#bcaf6f', '#ffe945'],
    turbo: ['#30123b', '#28bceb', '#a2fc3c', '#fb8022', '#7a0403'],
    coolwarm: ['#3b4cc0', '#88a0e0', '#dddddd', '#e58267', '#b40426'],
    jet: ['#00007f', '#0000ff', '#00ffff', '#ffff00', '#ff0000']
  };
  function hx(h) { h = h.replace('#', ''); return [0, 2, 4].map(i => parseInt(h.substr(i, 2), 16)); }
  function lerpHex(a, b, t) {
    const pa = hx(a), pb = hx(b);
    return rgb(
      Math.round(pa[0] + (pb[0] - pa[0]) * t),
      Math.round(pa[1] + (pb[1] - pa[1]) * t),
      Math.round(pa[2] + (pb[2] - pa[2]) * t)
    );
  }
  function sampleLut(name, t) {
    const c = LUTS[name]; const p = t * (c.length - 1); const i = Math.floor(p);
    if (i >= c.length - 1) return c[c.length - 1];
    return lerpHex(c[i], c[i + 1], p - i);
  }

  function applyEdit(grp, k, v) {
    if (grp === 'colorUniform') {
      S.trackColors = S.trackColors.map(() => v);
    } else if (grp === 'colorPalette') {
      S.schema.color.source = v;
      const gen = PALETTES[v] || PALETTES['random'];
      S.trackColors = S.trackColors.map(() => gen());
    } else if (grp === 'colorLut') {
      S.schema.color.source = v;
      const n = S.trackColors.length;
      S.trackColors = S.trackColors.map((_, i) => sampleLut(v, n > 1 ? i / (n - 1) : 0));
    } else if (grp === 'tracks') {
      S.style[k] = v;
    } else if (grp === 'background') {
      S.style[k] = v;
    } else if (grp === 'grid') {
      S.style[k] = v;
    }
  }

  function pos(e) {
    const r = cv.getBoundingClientRect();
    return [(e.clientX - r.left) * cv.width / r.width, (e.clientY - r.top) * cv.height / r.height];
  }

  cv.addEventListener('click', e => {
    const [mx, my] = pos(e);
    if (nearTrack(mx, my)) tracksTip(e.offsetX, e.offsetY);
    else tip.style.display = 'none';
  });

  cv.addEventListener('contextmenu', e => {
    e.preventDefault();
    const [mx, my] = pos(e);
    if (onGrid(mx, my) && !nearTrack(mx, my)) gridTip(e.offsetX, e.offsetY);
    else bgTip(e.offsetX, e.offsetY);
  });

  render();
})();