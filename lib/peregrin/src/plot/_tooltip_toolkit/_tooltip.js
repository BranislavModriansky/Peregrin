(function () {
  const S = window.GRAPH_STATE || {};

  // ---- inject CSS once ----
  if (!document.getElementById('peregrin-graph-style')) {
    const style = document.createElement('style');
    style.id = 'peregrin-graph-style';
    style.textContent = "__PEREGRIN_CSS__";
    document.head.appendChild(style);
  }

  document.querySelectorAll('.peregrin-graph:not([data-pg-init])').forEach(init);

  function init(root) {
    root.dataset.pgInit = '1';
    const svg = root.querySelector('svg');
    if (!svg) return;

    const DRAWABLE = 'path,line,rect,circle,polygon,polyline,use';

    // Collect every element whose own id starts with `prefix`, OR that lives
    // inside a <g> whose id starts with `prefix`. Matplotlib emits gids as ids
    // on wrapping <g> groups (and sometimes appends suffixes / dedup numbers),
    // so we resolve to the drawable leaves inside those groups.
    function collect(prefix) {
      const roots = svg.querySelectorAll(`[id^="${prefix}"]`);
      const out = new Set();
      roots.forEach(node => {
        if (node.matches(DRAWABLE)) out.add(node);
        node.querySelectorAll(DRAWABLE).forEach(el => out.add(el));
      });
      return [...out];
    }

    const groups = {
      tracks: collect('pg:tracks'),
      heads: collect('pg:heads'),
      grid: collect('pg:grid'),
      background: collect('pg:background'),
      figure: collect('pg:figure'),
    };

    // Hit targets = the original gid-tagged nodes (groups), so right-click works
    // even on the transparent group wrapper.
    const hitNodes = prefix => [...svg.querySelectorAll(`[id^="${prefix}"]`)];

    const tip = document.createElement('div');
    tip.className = 'pg-tooltip';
    root.appendChild(tip);

    // ---- attribute editing helpers (mutate SVG in place) ----
    // Only override an attribute if it isn't explicitly "none" (preserves
    // fill-only / stroke-only artists like unfilled head markers).
    function setStroke(nodes, v) {
      nodes.forEach(p => {
        if (p.getAttribute('stroke') !== 'none') p.setAttribute('stroke', v);
        // matplotlib sometimes puts color in inline style
        if (p.style && p.style.stroke && p.style.stroke !== 'none') p.style.stroke = v;
      });
    }
    function setFill(nodes, v) {
      nodes.forEach(p => {
        if (p.getAttribute('fill') !== 'none') p.setAttribute('fill', v);
        if (p.style && p.style.fill && p.style.fill !== 'none') p.style.fill = v;
      });
    }
    function setStrokeWidth(nodes, v) {
      nodes.forEach(p => {
        p.setAttribute('stroke-width', v);
        if (p.style && p.style.strokeWidth) p.style.strokeWidth = v;
      });
    }
    function setScale(nodes, value) {
      const scale = Number(value);
      if (!Number.isFinite(scale) || scale <= 0) return;

      const uses = nodes.filter(
        node => node.localName === 'use' && !node.closest('defs')
      );

      // Resolve the marker paths referenced by Matplotlib's <use> elements.
      // Scaling the definition preserves every <use x="..." y="..."> position.
      const definitions = new Set();

      uses.forEach(use => {
        const href =
          use.getAttribute('href') ||
          use.getAttribute('xlink:href') ||
          use.getAttributeNS('http://www.w3.org/1999/xlink', 'href');

        if (!href || !href.startsWith('#')) return;

        const id = href.slice(1);
        const definition = [...svg.querySelectorAll('defs [id]')]
          .find(node => node.id === id);

        if (definition) definitions.add(definition);
      });

      function scaleAroundCenter(node) {
        if (node.dataset.pgBaseTransform === undefined) {
          node.dataset.pgBaseTransform =
            node.getAttribute('transform') || '';
        }

        let cx = 0;
        let cy = 0;

        try {
          const box = node.getBBox();
          cx = box.x + box.width / 2;
          cy = box.y + box.height / 2;
        } catch (_) {
          // Matplotlib marker definitions normally use (0, 0) as their center.
        }

        const scaling =
          `translate(${cx} ${cy}) scale(${scale}) translate(${-cx} ${-cy})`;

        const transform = [
          node.dataset.pgBaseTransform,
          scaling,
        ].filter(Boolean).join(' ');

        node.setAttribute('transform', transform);
      }

      if (definitions.size) {
        definitions.forEach(scaleAroundCenter);
        return;
      }

      // Fallback for SVG markers rendered directly instead of through <use>.
      nodes
        .filter(node => !node.closest('defs'))
        .forEach(scaleAroundCenter);
    }

    const EDITORS = {
      tracks: {
        color: v => setStroke(groups.tracks, v),
        lw: v => setStrokeWidth(groups.tracks, v),
        headColor: v => {
          setStroke(groups.heads, v);
          setFill(groups.heads, v);
        },
        headSize: v => setScale(groups.heads, v),
      },
      grid: {
        gridColor: v => setStroke(groups.grid, v),
        gridLw: v => setStrokeWidth(groups.grid, v),
      },
      background: {
        faceColor: v => {
          setFill(groups.background, v);
          setFill(groups.figure, v);
        },
      },
    };

    function applyEdit(grp, key, value) {
      const fn = EDITORS[grp] && EDITORS[grp][key];
      if (fn) fn(value);
      if (S.style) S.style[key] = value;
    }

    // ---- tooltip UI ----
    const row = (label, input) =>
      `<div class="pg-row"><label>${label}</label><br>${input}</div>`;
    const header = t =>
      `<div class="pg-header">${t}<span class="pg-close">✕</span></div>`;

    function showTip(x, y, html) {
      tip.innerHTML = html;
      tip.style.left = x + 'px';
      tip.style.top = y + 'px';
      tip.style.display = 'block';
      tip.querySelectorAll('[data-k]').forEach(el => {
        el.addEventListener('input', () => {
          let v = el.value;
          if (el.type === 'range') v = parseFloat(v);
          applyEdit(el.dataset.grp, el.dataset.k, v);
        });
      });
      const close = tip.querySelector('.pg-close');
      if (close) close.onclick = () => (tip.style.display = 'none');
    }

    const st = S.style || {};
    const hex = (c, fallback) => (c || fallback).slice(0, 7); // color input needs #rrggbb

    const MENUS = {
      tracks: (x, y) => showTip(x, y, header('Tracks')
        + row('Color', `<input type="color" data-grp="tracks" data-k="color" value="${hex(st.trackColor, '#000000')}">`)
        + row('Head color', `<input type="color" data-grp="tracks" data-k="headColor" value="${hex(st.trackColor, '#000000')}">`)
        + row('Line width', `<input type="range" data-grp="tracks" data-k="lw" min="0.25" max="6" step="0.25" value="${st.lw ?? 1}">`)
        + row('Head size', `<input type="range" data-grp="tracks" data-k="headSize" min="0.25" max="6" step="0.25" value="${st.headSize ?? 1}">`)),
      grid: (x, y) => showTip(x, y, header('Grid')
        + row('Grid color', `<input type="color" data-grp="grid" data-k="gridColor" value="${hex(st.gridColor, '#dcdcdc')}">`)
        + row('Grid width', `<input type="range" data-grp="grid" data-k="gridLw" min="0" max="4" step="0.25" value="${st.gridLw ?? 0.75}">`)),
      background: (x, y) => showTip(x, y, header('Background')
        + row('Backdrop color', `<input type="color" data-grp="background" data-k="faceColor" value="${hex(st.faceColor, '#ffffff')}">`)),
    };

    // ---- event wiring: right-click on the tagged element opens its menu ----
    function bind(nodes, menu) {
      nodes.forEach(n => {
        n.style.pointerEvents = 'all';
        n.addEventListener('contextmenu', e => {
          e.preventDefault();
          e.stopPropagation();
          const r = root.getBoundingClientRect();
          MENUS[menu](e.clientX - r.left, e.clientY - r.top);
        });
      });
    }

    bind(hitNodes('pg:tracks'), 'tracks');
    bind(hitNodes('pg:heads'), 'tracks');
    bind(hitNodes('pg:grid'), 'grid');
    // background last so figure/background wrappers both open the bg menu
    bind(hitNodes('pg:background'), 'background');
    bind(hitNodes('pg:figure'), 'background');

    svg.addEventListener('contextmenu', e => e.preventDefault());
  }
})();