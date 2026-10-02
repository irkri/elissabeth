"""The page both benchmark reports are built on: tokens, the plotly
styling, view selectors, tiles and the HTML shell.

A report is one self-contained HTML file: plotly.js is inlined, the
figures are drawn with light colours and restyled from the page's CSS
tokens when the viewer's theme is dark (``THEME_JS``), and a chart with
several views gets a ``<select>`` above it that switches which traces are
visible. Every y axis of line charts is fitted to the traces shown
(``rescale`` in ``THEME_JS``): on load, on every view or legend change and
after a double-click, with one range for panels that share their y axis
and a labelled tick at every gridline of a log axis. Reference lines are
shapes, not traces, so they never stretch an axis.

Formulas are LaTeX (``m`` inline, ``md`` displayed), typeset by KaTeX when
the page loads. KaTeX is vendored in ``katex/`` (MIT) and inlined with its
fonts, so a report needs no network.
"""
import base64
import html
import json
import math
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'

# Categorical slots in their validated order (light, dark).
SLOTS = [
    ("#2a78d6", "#3987e5"),   # blue
    ("#eb6834", "#d95926"),   # orange
    ("#1baf7a", "#199e70"),   # aqua
    ("#eda100", "#c98500"),   # yellow
    ("#e87ba4", "#d55181"),   # magenta
    ("#008300", "#008300"),   # green
    ("#4a3aa7", "#9085e9"),   # violet
    ("#e34948", "#e66767"),   # red
]
NEUTRAL = "#898781"
"""A baseline series (eager, attention): the muted ink, same in both
themes."""
RAMP = [  # ordinal blue (light, dark): lighter = smaller
    ("#86b6ef", "#184f95"),
    ("#5598e7", "#256abf"),
    ("#2a78d6", "#3987e5"),
    ("#1c5cab", "#6da7ec"),
    ("#104281", "#9ec5f4"),
    ("#0d366b", "#cde2fb"),
]
DIVERGING = {  # blue (shrinks) <-> red (grows), grey midpoint
    "light": [[0.0, "#184f95"], [0.25, "#6da7ec"], [0.5, "#f0efec"],
              [0.75, "#ec8a89"], [1.0, "#b42c2c"]],
    "dark": [[0.0, "#3987e5"], [0.25, "#256abf"], [0.5, "#383835"],
             [0.75, "#a83a3a"], [1.0, "#e66767"]],
}

LIGHT = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e",
             muted="#898781", grid="#e1e0d9", axis="#c3c2b7")

SEMIRING_COLOR = {
    "reals": SLOTS[0][0], "log": SLOTS[1][0], "arctic": SLOTS[2][0],
    "bayesian": SLOTS[3][0], "reals · cosine": SLOTS[4][0],
}
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d"]}


def ramp(n: int) -> list[str]:
    """``n`` ordinal steps of the blue ramp (light values)."""
    if n <= 1:
        return [RAMP[2][0]]
    idx = [round(i * (len(RAMP) - 1) / (n - 1)) for i in range(n)]
    return [RAMP[i][0] for i in idx]


def style(fig: go.Figure, height: int, legend_title: str | None = None
          ) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height,
        font=dict(family=FONT, size=12.5, color=LIGHT["ink2"]),
        margin=dict(l=62, r=18, t=84, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=LIGHT["surface"],
        legend=dict(orientation="h", yref="container", yanchor="top",
                    y=0.995, x=0, xanchor="left",
                    title_text=legend_title or "", font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor=LIGHT["grid"], linecolor=LIGHT["axis"],
                     zeroline=False, showline=True, ticks="")
    fig.update_yaxes(gridcolor=LIGHT["grid"], linecolor=LIGHT["axis"],
                     zeroline=False, showline=True, ticks="")
    fig.update_annotations(font=dict(size=12.5, color=LIGHT["ink"]))
    return fig


def tick_label(v: float) -> str:
    """A tick label: plain from 0.001 to 99,999, else a power of ten."""
    if 1e-3 <= abs(v) < 1e5:
        return f"{v:,.6g}"
    exp = math.floor(math.log10(abs(v)) + 1e-9)
    mant = v / 10**exp
    head = "" if abs(mant - 1) < 1e-6 else f"{mant:g}·"
    return f"{head}10<sup>{exp}</sup>".replace("<sup>-", "<sup>−")


def nice_log_ticks(values: Iterable[float]) -> dict:
    """1-2-5 ticks over a narrow log range, decades over a wide one, every
    k-th decade over a very wide one."""
    vals = [abs(v) for v in values if v and math.isfinite(v)]
    if not vals:
        return {}
    lo, hi = math.log10(min(vals)), math.log10(max(vals))
    span = hi - lo
    steps = [1, 2, 5] if span <= 2.2 else [1]
    every = max(1, math.ceil(span / 9))
    ticks = []
    for e in range(math.floor(lo) - 1, math.ceil(hi) + 2):
        if e % every:
            continue
        ticks += [m * 10.0**e for m in steps]
    margin = 0.15 if span > 0.3 else 0.05
    ticks = [t for t in ticks
             if lo - margin <= math.log10(t) <= hi + margin]
    if len(ticks) < 2:   # a range inside one 1-2-5 step
        ticks = [10 ** lo, 10 ** hi]
    return dict(tickvals=ticks, ticktext=[tick_label(t) for t in ticks])


def apply_log_ticks(fig: go.Figure, shared: bool) -> None:
    """Explicit tick labels on every logarithmic y axis of ``fig``, from
    the data of its traces (all views), one set per axis or one shared."""
    by_axis: dict[str, list[float]] = {}
    for trace in fig.data:
        ys = [float(y) for y in (trace.y if trace.y is not None else [])
              if y is not None]
        by_axis.setdefault(trace.yaxis or "y", []).extend(ys)
    every = [v for vs in by_axis.values() for v in vs]
    for axis, ys in by_axis.items():
        name = "yaxis" + axis[1:]
        layout_axis = fig.layout[name]
        if layout_axis.type != "log":
            continue
        layout_axis.update(**nice_log_ticks(every if shared else ys))


def log_ticks(values: Iterable[float]) -> dict:
    """Tick labels at the given values of a log axis, compactly written."""
    vals = sorted({v for v in values if v and math.isfinite(v)})
    return dict(tickvals=vals, ticktext=[compact(v) for v in vals])


def compact(v: float) -> str:
    if v >= 1e6 and v % 1e6 == 0:
        return f"{v / 1e6:g}M"
    if v >= 1000 and v % 1024 == 0:
        return f"{v / 1024:g}k" if v < 2**20 else f"{v / 2**20:g}M"
    if v >= 1000 and v % 1000 == 0:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def sci(v: float | None, digits: int = 1) -> str:
    """A number for prose and tiles: plain between 0.001 and 10,000,
    otherwise ``3.2·10⁻⁴`` in HTML."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if not math.isfinite(v):
        return "∞"
    if v == 0:
        return "0"
    exp = math.floor(math.log10(abs(v)))
    if -3 <= exp < 4:
        return f"{v:,.{max(digits - exp, 0)}f}"
    mant = v / 10**exp
    head = "" if round(mant, digits) == 1 else f"{mant:.{digits}f}·"
    return f"{head}10<sup>{exp}</sup>".replace("<sup>-", "<sup>−")


# --------------------------------------------------------------------------
# HTML pieces
# --------------------------------------------------------------------------
KATEX = Path(__file__).with_name("katex")


def m(tex: str) -> str:
    """Inline LaTeX, typeset by KaTeX on the page."""
    return r"\(" + html.escape(tex, quote=False) + r"\)"


def eq(name: str, value) -> str:
    """``name = value`` as inline math, the value upright (``16k``, ``131,072``)."""
    return m(f"{name} = \\text{{{value}}}")


def md(tex: str) -> str:
    """A displayed LaTeX formula."""
    return ("<div class='formula'>" + r"\[" + html.escape(tex, quote=False)
            + r"\]" + "</div>")


def katex() -> str:
    """KaTeX's stylesheet (its woff2 fonts as data URIs) and scripts,
    inlined; empty without the vendored copy, and the TeX then shows as
    written."""
    if not (KATEX / "katex.min.js").exists():
        return ""

    def font(match: re.Match) -> str:
        data = (KATEX / "fonts" / f"{match.group(1)}.woff2").read_bytes()
        return ('src:url(data:font/woff2;base64,'
                + base64.b64encode(data).decode() + ') format("woff2")')

    css = re.sub(r'src:url\(fonts/([^)]+?)\.woff2\) format\("woff2"\)'
                 r'(?:,url\([^)]*\) format\("[a-z]+"\))*', font,
                 (KATEX / "katex.min.css").read_text())
    js = ((KATEX / "katex.min.js").read_text() + "\n"
          + (KATEX / "auto-render.min.js").read_text())
    return f"<style>{css}</style>\n<script>{js}</script>"


MATH_JS = r"""
if (window.renderMathInElement) renderMathInElement(document.body, {
  delimiters: [{left: '\\[', right: '\\]', display: true},
               {left: '\\(', right: '\\)', display: false}],
  ignoredClasses: ['js-plotly-plot'], throwOnError: false,
});
"""

def figure(fig: go.Figure | None, views: dict[str, list[bool]] | None = None,
           view_label: str = "View", note: str = "") -> str:
    """A figure, optionally with a selector over ``views`` (label ->
    visibility of every trace; the first is shown)."""
    if fig is None:
        return f"<p class='empty'>{html.escape(note or 'No data for this chart.')}</p>"
    control = ""
    if views:
        first = next(iter(views.values()))
        for trace, visible in zip(fig.data, first):
            trace.visible = visible
        options = "".join(f"<option>{html.escape(k)}</option>" for k in views)
        data = html.escape(json.dumps(views), quote=True)
        control = (f"<label class='view'>{html.escape(view_label)} "
                   f"<select data-views=\"{data}\">{options}</select></label>")
    div = pio.to_html(fig, include_plotlyjs=False, full_html=False,
                      config=PLOT_CONFIG, default_width="100%")
    return f"<figure class='chart'>{control}{div}</figure>"


def section(title: str, intro: str, *body: str, anchor: str = "") -> str:
    ident = f" id='{anchor}'" if anchor else ""
    return (f"<section{ident}><h2>{html.escape(title)}</h2>"
            f"<div class='prose'>{intro}</div>{''.join(body)}</section>")


def reading(text: str) -> str:
    """The measured outcome under a chart."""
    return f"<p class='reading'>{text}</p>"


def tiles(items: Sequence[tuple[str, str, str]]) -> str:
    cells = "".join(
        f"<div class='tile'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>"
        for label, value, sub in items
    )
    return f"<div class='tiles'>{cells}</div>"


def table(header: Sequence[str], rows: Sequence[Sequence[str]],
          numeric: Sequence[int] = ()) -> str:
    head = "".join(f"<th{' class=num' if i in numeric else ''}>{h}</th>"
                   for i, h in enumerate(header))
    body = "".join(
        "<tr>" + "".join(f"<td{' class=num' if i in numeric else ''}>{c}</td>"
                         for i, c in enumerate(row)) + "</tr>"
        for row in rows
    )
    return (f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead>"
            f"<tbody>{body}</tbody></table></div>")


def glossary(groups: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> str:
    blocks = []
    for heading, items in groups:
        rows = "".join(
            f"<dt>{k if k.startswith(chr(92) + '(') else '<code>' + html.escape(k) + '</code>'}"
            f"</dt><dd>{v}</dd>" for k, v in items)
        blocks.append(f"<div><h3>{html.escape(heading)}</h3><dl>{rows}</dl></div>")
    return ("<details class='glossary'><summary>Terms used on this page"
            "</summary><div class='gloss-grid'>" + "".join(blocks)
            + "</div></details>")


CSS = """
/* Layout: one reading column of prose, charts at full width beneath it. */
:root {
  --plane: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10); --code: #efeee9; --accent: #2a78d6;
  --critical: #d03b3b;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255, 255, 255, 0.10); --code: #262624; --accent: #3987e5;
    --critical: #e66767; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --plane: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255, 255, 255, 0.10); --code: #262624; --accent: #3987e5;
  --critical: #e66767; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--plane); color: var(--ink);
       font-family: var(--font); font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 36px 80px;
        padding-inline: max(16px, 3vw); }
header h1 { font-size: 28px; line-height: 1.2; margin: 0 0 6px;
            letter-spacing: -0.01em; text-wrap: balance; }
header .meta { color: var(--ink-2); font-size: 13px; margin: 0;
               font-variant-numeric: tabular-nums; }
h2 { font-size: 20px; margin: 0 0 8px; text-wrap: balance; }
h3 { font-size: 13px; margin: 0 0 6px; text-transform: uppercase;
     letter-spacing: .05em; color: var(--ink-2); }
.prose { max-width: 74ch; }
.prose p, .lead p { margin: 0 0 10px; }
.lead { max-width: 74ch; margin: 22px 0 8px; }
.formula { margin: 4px 0 10px; overflow-x: auto; overflow-y: hidden; }
.katex { font-size: 1.08em; }
code { font-family: var(--mono); font-size: .88em; background: var(--code);
       padding: 1px 5px; border-radius: 4px; }
section { border-top: 1px solid var(--grid); padding-block: 26px 10px;
          margin-top: 22px; }
figure.chart { margin: 14px 0 6px; background: var(--surface);
               border: 1px solid var(--border); border-radius: 8px;
               padding: 8px 8px 2px; min-width: 0; }
label.view { display: inline-flex; gap: 8px; align-items: center;
             font-size: 13px; color: var(--ink-2); margin: 4px 0 0 8px; }
label.view select { font: inherit; color: var(--ink); background: var(--plane);
                    border: 1px solid var(--axis); border-radius: 6px;
                    padding: 3px 8px; }
label.view select:focus-visible { outline: 2px solid var(--accent);
                                  outline-offset: 1px; }
p.reading { max-width: 74ch; font-size: 14px; color: var(--ink);
            border-left: 3px solid var(--accent); padding: 2px 0 2px 12px;
            margin: 10px 0 14px; }
p.empty { color: var(--muted); font-style: italic; }
.tiles { display: grid; gap: 12px; margin: 22px 0 6px;
         grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.tile { background: var(--surface); border: 1px solid var(--border);
        border-radius: 8px; padding: 12px 14px; min-width: 0; }
.tile .label { font-size: 12px; text-transform: uppercase;
               letter-spacing: .05em; color: var(--ink-2); }
.tile .value { font-size: 26px; font-weight: 650; letter-spacing: -0.01em;
               margin: 2px 0; }
.tile .sub { font-size: 12.5px; color: var(--ink-2); }
.table-wrap { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; font-size: 13px; min-width: 100%;
        font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 12px 5px 0;
         border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
td.num, th.num { text-align: right; }
td.fail { color: var(--critical); }
details.glossary { margin: 18px 0 0; border: 1px solid var(--border);
                   border-radius: 8px; background: var(--surface);
                   padding: 4px 16px; }
details.glossary summary { cursor: pointer; font-weight: 600;
                           padding: 8px 0; }
.gloss-grid { display: grid; gap: 8px 28px; padding-bottom: 10px;
              grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.gloss-grid > div { min-width: 0; }
dl { margin: 0 0 10px; font-size: 13.5px; }
dt { margin-top: 6px; }
dd { margin: 2px 0 0 0; color: var(--ink-2); }
footer { color: var(--ink-2); font-size: 12.5px; margin-top: 36px;
         border-top: 1px solid var(--grid); padding-top: 14px; }
sup { line-height: 0; }
"""

THEME_JS = """
(function () {
  const SERIES = %SERIES%;
  const DIVERGING = %DIVERGING%;
  const toDark = new Map(SERIES.map(([l, d]) => [l, d]));
  const toLight = new Map(SERIES.map(([l, d]) => [d, l]));
  function dark() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return matchMedia('(prefers-color-scheme: dark)').matches;
  }
  function swap(c, map) {
    return typeof c === 'string' && map.has(c.toLowerCase())
      ? map.get(c.toLowerCase()) : c;
  }
  function apply() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    const isDark = dark(), map = isDark ? toDark : toLight;
    for (const gd of document.querySelectorAll('.js-plotly-plot')) {
      if (!gd.layout) continue;
      const lay = {'font.color': v('--ink-2'), 'plot_bgcolor': v('--surface'),
                   'paper_bgcolor': 'rgba(0,0,0,0)'};
      for (const k of Object.keys(gd.layout)) {
        if (/^[xy]axis\\d*$/.test(k)) {
          lay[k + '.gridcolor'] = v('--grid');
          lay[k + '.linecolor'] = v('--axis');
        }
      }
      (gd.layout.annotations || []).forEach((a, i) => {
        lay['annotations[' + i + '].font.color'] = v('--ink');
      });
      (gd.layout.shapes || []).forEach((s, i) => {
        if (s.line && s.line.color) lay['shapes[' + i + '].line.color'] = swap(s.line.color, map);
      });
      Plotly.relayout(gd, lay);
      gd.data.forEach((tr, i) => {
        const upd = {};
        if (tr.line && tr.line.color) upd['line.color'] = swap(tr.line.color, map);
        if (tr.marker && typeof tr.marker.color === 'string')
          upd['marker.color'] = swap(tr.marker.color, map);
        if (tr.marker && tr.marker.line && tr.marker.line.color)
          upd['marker.line.color'] = isDark ? v('--surface') : v('--surface');
        if (tr.type === 'heatmap') upd['colorscale'] = [DIVERGING[isDark ? 'dark' : 'light']];
        if (Object.keys(upd).length) Plotly.restyle(gd, upd, [i]);
      });
    }
  }
  function views() {
    for (const sel of document.querySelectorAll('select[data-views]')) {
      const views = JSON.parse(sel.dataset.views);
      const gd = sel.closest('figure').querySelector('.js-plotly-plot');
      sel.addEventListener('change', () => {
        Plotly.restyle(gd, {visible: views[sel.value]});
      });
    }
  }
  // A tick label as report.tick_label writes it.
  function label(v) {
    const a = Math.abs(v);
    if (a >= 1e-3 && a < 1e5)
      return (+v.toPrecision(6)).toLocaleString('en-US', {maximumFractionDigits: 6});
    const e = Math.floor(Math.log10(a) + 1e-9), m = v / Math.pow(10, e);
    const head = Math.abs(m - 1) < 1e-6 ? '' : (+m.toPrecision(6)) + '\u00b7';
    return head + '10<sup>' + (e < 0 ? '\u2212' + (-e) : e) + '</sup>';
  }
  // Ticks inside [a, b] (log10 units): 1-2-3-5 or every digit on a narrow
  // range, 1-2-5 or 1-3 on a medium one, decades (or every k-th) on a wide
  // one; every tick gets a label.
  function logTicks(a, b) {
    const span = b - a;
    const sets = span <= 1 ? [[1, 2, 3, 5], [1, 2, 3, 4, 5, 6, 7, 8, 9]]
      : span <= 3.5 ? [[1, 2, 5]] : span <= 6 ? [[1, 3]] : [[1]];
    const every = Math.max(1, Math.ceil(span / 10));
    let vals = [];
    for (const steps of sets) {
      vals = [];
      for (let e = Math.floor(a) - 1; e <= Math.ceil(b) + 1; e++) {
        if (((e % every) + every) % every) continue;
        for (const m of steps) {
          const v = m * Math.pow(10, e), l = Math.log10(v);
          if (l >= a && l <= b) vals.push(v);
        }
      }
      if (vals.length >= 3) break;
    }
    if (vals.length < 2)
      vals = [a + 0.15 * span, b - 0.15 * span].map(l => +Math.pow(10, l).toPrecision(2));
    return vals;
  }
  // Fit every y axis carrying line traces to the traces now visible; axes
  // that match one another (panels sharing y) get one range together.
  function rescale(gd) {
    const fl = gd._fullLayout, full = gd._fullData;
    if (!fl || !full) return;
    const names = Object.keys(fl).filter(k => /^yaxis\\d*$/.test(k));
    const axis = r => 'yaxis' + r.slice(1);
    const root = n => {
      for (let i = 0; i < 20 && fl[n] && fl[n].matches; i++) n = axis(fl[n].matches);
      return n;
    };
    const lo = {}, hi = {}, other = {};
    for (const tr of full) {
      const g = root(axis(tr.yaxis || 'y'));
      if (tr.type !== 'scatter') { other[g] = true; continue; }
      if (tr.visible !== true) continue;
      const log = fl[g] && fl[g].type === 'log', ys = tr.y || [];
      for (let i = 0; i < ys.length; i++) {
        const v = +ys[i];
        if (!isFinite(v) || (log && v <= 0)) continue;
        if (!(v >= lo[g])) lo[g] = v;
        if (!(v <= hi[g])) hi[g] = v;
      }
    }
    const upd = {};
    for (const g of Object.keys(lo)) {
      if (other[g] || !fl[g]) continue;
      let range, ticks = null;
      if (fl[g].type === 'log') {
        const a = Math.log10(lo[g]), b = Math.log10(hi[g]);
        const pad = Math.max(0.04, 0.05 * (b - a));
        range = [a - pad, b + pad];
        ticks = logTicks(range[0], range[1]);
      } else {
        const pad = (hi[g] - lo[g]) * 0.06 || Math.abs(hi[g]) * 0.1 || 1;
        range = [lo[g] - pad, hi[g] + pad];
      }
      for (const n of names) {
        if (root(n) !== g) continue;
        upd[n + '.range'] = range;
        upd[n + '.autorange'] = false;
        if (ticks) {
          upd[n + '.tickvals'] = ticks;
          upd[n + '.ticktext'] = ticks.map(label);
        }
      }
    }
    if (Object.keys(upd).length) Plotly.relayout(gd, upd);
  }
  function fit() {
    for (const gd of document.querySelectorAll('.js-plotly-plot')) {
      rescale(gd);
      // a view or a legend entry switched traces on or off
      gd.on('plotly_restyle', (d) => {
        if (d && d[0] && 'visible' in d[0]) rescale(gd);
      });
      // a double-click reset the axes to plotly's own autorange
      gd.on('plotly_relayout', (d) => {
        if (d && Object.keys(d).some(k => /^yaxis\\d*\\.autorange$/.test(k) && d[k] === true))
          rescale(gd);
      });
    }
  }
  function start() {
    apply(); views(); fit();
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', apply);
    new MutationObserver(apply).observe(document.documentElement,
      {attributes: true, attributeFilter: ['data-theme']});
  }
  if (document.readyState === 'complete') start();
  else window.addEventListener('load', start);
})();
"""


def page(title: str, heading: str, meta: str, body: str,
         fragment: bool = False) -> str:
    """The whole HTML document, or with ``fragment`` only what goes inside
    a host's own skeleton (title, style, scripts, content)."""
    series = [list(p) for p in SLOTS + RAMP]
    script = MATH_JS + (THEME_JS.replace("%SERIES%", json.dumps(series))
                        .replace("%DIVERGING%", json.dumps(DIVERGING)))
    math_assets = katex()
    if fragment:
        return f"""<title>{html.escape(title)}</title>
<style>{CSS}</style>
{math_assets}
<script>{get_plotlyjs()}</script>
<div class="wrap">
<header><h1>{html.escape(heading)}</h1><p class="meta">{meta}</p></header>
{body}
</div>
<script>{script}</script>
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<style>{CSS}</style>
{math_assets}
<script>{get_plotlyjs()}</script>
</head><body><div class="wrap">
<header><h1>{html.escape(heading)}</h1><p class="meta">{meta}</p></header>
{body}
</div>
<script>{script}</script>
</body></html>"""
