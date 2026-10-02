"""Render ``stability.py`` CSVs as one self-contained HTML report.

Several CSVs are concatenated (e.g. a main run and a separate
``--experiments primitive`` run); the JSON next to the first one supplies
the run's device and settings. Every section is drawn from the experiment
it needs and says so when that experiment is missing.

    cd project
    python benchmark/plot_stability.py benchmark/stability01.csv
    python benchmark/plot_stability.py benchmark/stability01.csv --fragment stability01.frag.html
"""
import argparse
import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from report import (PLOT_CONFIG, DIVERGING, NEUTRAL, RAMP, SLOTS, apply_log_ticks,
                    compact, figure, glossary, log_ticks, page, ramp, reading,
                    sci, section, style, table, tiles)

SEMIRINGS = ["reals", "log", "arctic", "bayesian"]
INPUTS = ["constant", "tokens", "drift", "gaussian", "sparse"]
NORMS = ["none", "mean", "sqrt", "ema"]
DTYPES = ["float32", "bf16-mixed", "bfloat16", "float16"]
DTYPE_COLOR = {"float32": SLOTS[0][0], "bf16-mixed": SLOTS[2][0],
               "bfloat16": SLOTS[1][0], "float16": SLOTS[3][0]}
NORM_COLOR = {"none": NEUTRAL, "mean": SLOTS[0][0], "sqrt": SLOTS[1][0],
              "ema": SLOTS[2][0]}
SEMIRING_COLOR = {s: SLOTS[i][0] for i, s in enumerate(SEMIRINGS)}
FLOAT32_MAX = 3.4028e38
FLOAT16_MAX = 65504.0
CEILING = 10.0
"""Errors are drawn up to this; a non-finite output is drawn at it."""
LOG_DOMAIN = ("log", "arctic")


def load(paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df["t1"] = df["t"] + 1          # positions counted from one: log axes
    # early primitive CSVs wrote an infinite position error as float max
    df.loc[df["err"] > 1e300, "err"] = np.inf
    df["err_plot"] = df["err"].replace(np.inf, CEILING).clip(upper=CEILING)
    for col in ("normalize", "input", "kernels"):
        df[col] = df[col].fillna("")
    # CSVs from before --scans evaluated the PyTorch path only
    df["scan"] = df["scan"].fillna("torch") if "scan" in df else "torch"
    return df


def present(values, order) -> list:
    have = set(values)
    return [v for v in order if v in have]


def lines(
    data: pd.DataFrame, *, panel: str, panels: list, x: str, y: str,
    series: str, order: list, colors: dict, label=str, view: str | None = None,
    views: list | None = None, log_x: bool = True, log_y: bool = True,
    x_title: str = "", y_title: str = "", hover: str = "", height: int = 360,
    shared_y: bool = True, panel_title=None, cols: int | None = None,
    hlines: list[tuple[float, str]] = (), markers: bool = False,
    hline_panels: list | None = None,
) -> tuple[go.Figure | None, dict | None]:
    """Small multiples of lines, ``panel`` by ``panel``, one line per
    ``series``, with one set of traces per ``view``."""
    if data.empty or not panels:
        return None, None
    cols = cols or len(panels)
    rows = math.ceil(len(panels) / cols)
    fig = make_subplots(rows=rows, cols=cols, shared_yaxes=shared_y,
                        subplot_titles=[(panel_title or str)(p) for p in panels],
                        horizontal_spacing=0.035, vertical_spacing=0.12)
    keys = views if views else [None]
    vis: dict = {str(k): [] for k in keys}
    for key in keys:
        sub = data if key is None else data[data[view] == key]
        shown: set = set()
        for i, pn in enumerate(panels):
            r, c = divmod(i, cols)
            for s in order:
                d = sub[(sub[panel] == pn) & (sub[series] == s)].sort_values(x)
                d = d[np.isfinite(d[y].astype(float))]
                if log_y:
                    d = d[d[y] > 0]
                if d.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=d[x], y=d[y], name=label(s), legendgroup=str(s),
                    showlegend=s not in shown,
                    mode="lines+markers" if markers else "lines",
                    line=dict(color=colors[s], width=2),
                    marker=dict(size=6, color=colors[s]),
                    hovertemplate=f"{label(s)} · {pn}<br>{hover}<extra></extra>",
                ), row=r + 1, col=c + 1)
                shown.add(s)
                for k in keys:
                    vis[str(k)].append(k == key)
    for value, text in hlines:
        for i, pn in enumerate(panels):
            if hline_panels is not None and pn not in hline_panels:
                continue
            r, c = divmod(i, cols)
            fig.add_hline(y=value, line=dict(color=NEUTRAL, width=1, dash="dot"),
                          row=r + 1, col=c + 1)
    fig.update_xaxes(type="log" if log_x else "linear")
    fig.update_yaxes(type="log" if log_y else "linear")
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text=x_title, row=rows, col=c)
    for r in range(1, rows + 1):
        fig.update_yaxes(title_text=y_title, row=r, col=1)
    style(fig, height)
    if log_y:
        apply_log_ticks(fig, shared_y)
    return fig, (vis if views else None)


def t_axis(fig: go.Figure, T: int) -> None:
    """Position ticks: every 16-fold step, or every 256-fold one when the
    figure has four or more panels side by side."""
    panels = sum(1 for k in fig.layout if k.startswith("xaxis"))
    step = 256 if panels >= 4 else 16
    ticks = [step**k for k in range(8) if step**k <= T * 1.01]
    fig.update_xaxes(tickvals=ticks, ticktext=[compact(v) for v in ticks],
                     tickangle=0)


def exponent(g: pd.DataFrame, semiring: str, octaves: int = 4) -> float:
    """Growth exponent of the represented quantity over the last
    ``octaves`` doublings of t: d log|y| / d log t for a stored number, d y
    / d ln t for a stored logarithm (the same thing in the log domain)."""
    g = g.sort_values("t1")
    tail = g[g["t1"] >= g["t1"].max() / 2**octaves]
    tail = tail[np.isfinite(tail["mag_med"]) & (tail["mag_med"] > 0)]
    if len(tail) < 2:
        return math.nan
    x = np.log(tail["t1"].to_numpy(float))
    if semiring in LOG_DOMAIN:
        y = tail["mag_med"].to_numpy(float)
    else:
        y = np.log(tail["mag_med"].to_numpy(float))
    return float(np.polyfit(x, y, 1)[0])


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def lead(meta: dict) -> str:
    shape = meta.get("shape", {})
    return f"""<div class="lead">
<p>At position <i>t</i> a LISS level of depth <i>p</i> combines every index tuple
<i>t</i><sub>1</sub> &lt; … &lt; <i>t</i><sub>p</sub> ≤ <i>t</i>, and there are
C(<i>t</i>+1, <i>p</i>) ≈ <i>t</i><sup>p</sup>/<i>p</i>! of them. What the stored number
does with that depends on the semiring and on the values:</p>
<ul>
<li>In the <b>reals</b> the level stores the sum itself. If the values of a channel
keep their sign over time, the products add up and the sum grows like
<i>t</i><sup>p</sup>; if the signs are random they partly cancel, and it grows like
<i>t</i><sup>p/2</sup>. float32 holds numbers up to 3.4·10<sup>38</sup>, float16 up to
65,504.</li>
<li>The <b>log</b> semiring stores the logarithm of a sum of positive terms, so the
same growth is an additive <i>p</i>·ln <i>t</i>: it cannot overflow, but each logsumexp
step rounds.</li>
<li>The <b>arctic</b> semiring keeps the best tuple's sum of values, which grows only
as fast as the extreme values of the input; the <b>bayesian</b> one keeps the best
tuple's product.</li>
<li>A <b>decay</b> λ<sup>gap</sup> with λ &lt; 1 caps the window at about
<i>T<sub>c</sub></i>/α positions; with λ &gt; 1 (a negative rate, which favours the
distant past) it grows exponentially, e<sup>α t/T<sub>c</sub></sup>, once the length
passes the context length <i>T<sub>c</sub></i>.</li>
<li><b>Normalisation</b> divides each level's partial sums by the number of positions
summed: <code>mean</code> by <i>c</i>, <code>sqrt</code> by √<i>c</i>. It is defined for
the reals and the log semiring. <code>ema</code> is a prototype that exists only in
this benchmark: it divides by the decayed count Σ λ<sup>t−s</sup>, which equals
<code>mean</code> without a decay.</li>
</ul>
<p>Every run evaluates one level ({shape.get('B', 2)} sequences, {shape.get('n_is', 4)}
heads, value width {shape.get('d_values', 8)}, input width {shape.get('d_in', 16)}; the
values are LayerNorm'd as by default) with fixed weights in float64, the reference,
and in float32, bfloat16, float16 and bfloat16 autocast (<code>bf16-mixed</code>,
float32 weights with matrix products in bfloat16, what Lightning's
<code>precision: bf16-mixed</code> does). The level is causal, so one run to length
<i>T</i> gives its output at every <i>t</i> ≤ <i>T</i>. The inputs are LayerNorm'd per
position, as the pre-norm hands them to a mixer:</p>
<ul>
<li><code>constant</code>: one vector at every position, the worst case for growth;</li>
<li><code>tokens</code>: a random sequence over 8 token vectors;</li>
<li><code>drift</code>: a vector that turns slowly, correlated over 1,000 positions;</li>
<li><code>gaussian</code>: independent random vectors;</li>
<li><code>sparse</code>: zero except at 1 % of the positions, like blanks between marks.</li>
</ul>
<p>Errors are against float64: ‖y − y<sub>64</sub>‖/‖y<sub>64</sub>‖ over a position's
outputs in the reals and the bayesian semiring, and the largest absolute error in the
log domain, where an absolute error of a stored logarithm is the relative error of the
number it stands for.</p></div>"""


def kpis(df: pd.DataFrame) -> str:
    items = []
    G = df[(df.experiment == "growth") & (df.dtype == "float32")]
    R = G[(G.semiring == "reals") & (G.normalize == "none")]
    bad = R[R.nonfinite > 0]
    if not R.empty:
        if bad.empty:
            items.append(("float32 overflow, reals", "none",
                          f"no depth up to {int(R.p.max())} up to t = {compact(R['T'].max())}"))
        else:
            first = bad.sort_values("t").iloc[0]
            items.append(("float32 overflow, reals", f"t = {compact(first.t1)}",
                          f"first non-finite output: p = {first.p}, {first.input} input"))
    last = G[G.t == G["T"] - 1]
    if not last.empty:
        fin = last[np.isfinite(last.err)]
        items.append(("float32 error at the end", sci(fin.err.median()),
                      f"median over every growth run at t = {compact(last.t1.max())},"
                      f" largest {sci(fin.err.max())}"))
    B16 = df[(df.experiment == "growth") & (df.dtype == "bfloat16")
             & (df.semiring == "reals") & (df.t == df["T"] - 1)]
    if not B16.empty:
        items.append(("bfloat16 error at the end", sci(B16.err.median()),
                      "reals, median over runs: the sums stop growing"))
    K = df[(df.experiment == "kernel") & (df.dtype == "float32")
           & (df.semiring == "reals") & (df.scale == 1)]
    if not K.empty:
        per = K.groupby("offset")["nonfinite"].mean()
        broken = per[per > 0]
        items.append(("Kernel factor overflow",
                      f"offset {broken.index.min():g}" if not broken.empty else "none",
                      "shared shift of queries and keys that breaks the reals,"
                      " with the kernel unchanged"))
    D = df[(df.experiment == "decay") & (df.dtype == "float32")
           & (df.semiring == "reals") & (df.alpha < 0) & (df.nonfinite > 0)]
    if not D.empty:
        D = D.assign(x=D.t1 / D.Tc, a=D.alpha.abs() * D.t1 / D.Tc)
        first = D.groupby("alpha")["x"].min()
        items.append(("Growing decay overflows", "α·t/T<sub>c</sub> ≈ "
                      f"{D.groupby('alpha')['a'].min().median():.0f}",
                      "reals, float32: "
                      + ", ".join(f"α = {a:g} at t = {x:.0f} T<sub>c</sub>"
                                  for a, x in first.items())))
    return tiles(items)


def growth_section(df: pd.DataFrame) -> str:
    G = df[(df.experiment == "growth") & (df.dtype == "float64")].copy()
    if G.empty:
        return section("How an iterated sum grows", "<p>No growth runs in these CSVs.</p>")
    G["view"] = G.semiring + " · " + G.normalize
    views = [f"{s} · {n}" for s in SEMIRINGS for n in NORMS
             if not G[(G.semiring == s) & (G.normalize == n)].empty]
    depths = sorted(G.p.unique())
    fig, vis = lines(
        G, panel="input", panels=present(G.input.unique(), INPUTS), x="t1",
        y="mag_max", series="p", order=depths,
        colors=dict(zip(depths, ramp(len(depths)))), label=lambda p: f"p = {p}",
        view="view", views=views, x_title="position t",
        y_title="largest |stored value|", hover="t = %{x:,}<br>%{y:.3g}",
        hlines=[(FLOAT32_MAX, "float32"), (FLOAT16_MAX, "float16")], height=400,
    )
    t_axis(fig, int(G["T"].max()))
    # the reading: exponents of the reals without normalisation
    R = G[(G.semiring == "reals") & (G.normalize == "none")]
    text = ""
    if not R.empty:
        pmax = R.p.max()
        ex = {kind: exponent(R[(R.input == kind) & (R.p == pmax)], "reals")
              for kind in present(R.input.unique(), INPUTS)}
        text = (f"In the reals without normalisation the largest value at p = {pmax}"
                f" grows like t to the power " + ", ".join(
                    f"{v:.1f} ({k})" for k, v in ex.items()) +
                f". A constant input reaches the full exponent p; independent inputs"
                f" about half of it. The dotted lines are the largest float32 and"
                f" float16 numbers. In the log domain the stored value is a logarithm"
                f" and stays below a few hundred.")
    intro = ("<p>The largest stored value at each position (float64, so nothing has"
             " overflowed yet), one panel per input, one line per depth. The"
             " selector picks the semiring and the normalisation.</p>")
    return section("How an iterated sum grows", intro,
                   figure(fig, vis, "Semiring · normalisation"),
                   reading(text) if text else "", anchor="growth")


def exponent_section(df: pd.DataFrame) -> str:
    G = df[(df.experiment == "growth") & (df.dtype == "float64")]
    if G.empty:
        return ""
    rows = []
    for (s, n, kind, p), g in G.groupby(["semiring", "normalize", "input", "p"]):
        rows.append(dict(semiring=s, normalize=n, input=kind, p=p,
                         a=exponent(g, s)))
    E = pd.DataFrame(rows)
    combos = [(s, n) for s in SEMIRINGS for n in NORMS
              if not E[(E.semiring == s) & (E.normalize == n)].empty]
    depths = sorted(E.p.unique())
    inputs = present(E.input.unique(), INPUTS)
    lim = max(1.0, float(np.nanmax(np.abs(E.a))))
    fig = go.Figure()
    vis: dict = {f"{s} · {n}": [] for s, n in combos}
    for s, n in combos:
        d = E[(E.semiring == s) & (E.normalize == n)]
        z = d.pivot_table(index="input", columns="p", values="a").reindex(
            index=inputs, columns=depths).round(1) + 0.0     # no "-0.0"
        fig.add_trace(go.Heatmap(
            z=z.to_numpy(), x=[f"p = {p}" for p in depths], y=inputs,
            zmid=0, zmin=-lim, zmax=lim, colorscale=DIVERGING["light"],
            texttemplate="%{z:.1f}", textfont=dict(size=12.5),
            colorbar=dict(title=dict(text="exponent", side="right"),
                          thickness=10, outlinewidth=0),
            xgap=2, ygap=2,
            hovertemplate="%{y} · %{x}<br>grows like t<sup>%{z:.2f}</sup><extra></extra>",
        ))
        for k in vis:
            vis[k].append(k == f"{s} · {n}")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    style(fig, 320)
    fig.update_layout(margin=dict(t=64, l=90))
    R = E[(E.semiring == "reals")]
    text = ""
    if not R.empty:
        def at(n, kind):
            v = R[(R.normalize == n) & (R.input == kind) & (R.p == R.p.max())].a
            return f"{round(v.iloc[0], 1) + 0.0:.1f}" if len(v) else "–"
        pm = R.p.max()
        text = (f"At p = {pm}: without normalisation a constant input grows with"
                f" exponent {at('none', 'constant')} and an independent one with"
                f" {at('none', 'gaussian')}. <code>mean</code> holds the constant"
                f" input at {at('mean', 'constant')} but drives the independent one"
                f" down at {at('mean', 'gaussian')}; <code>sqrt</code> does the"
                f" reverse ({at('sqrt', 'constant')} and {at('sqrt', 'gaussian')})."
                f" No count normalisation keeps both at zero, because the right"
                f" divisor depends on how coherent the values are.")
    intro = ("<p>The exponent <i>a</i> in |ISS| ∝ <i>t</i><sup>a</sup> of the represented"
             " quantity over the last four doublings of <i>t</i>: the slope of"
             " log |y| against log <i>t</i> for a stored number, and of the stored"
             " logarithm against ln <i>t</i> in the log domain, which is the same"
             " quantity. Zero means the level keeps its scale whatever the length;"
             " red grows, blue shrinks.</p>")
    return section("Growth exponents", intro, figure(fig, vis, "Semiring · normalisation"),
                   reading(text) if text else "", anchor="exponents")


def precision_section(df: pd.DataFrame) -> str:
    G = df[(df.experiment == "growth") & (df.dtype != "float64")].copy()
    if G.empty:
        return ""
    G["view"] = ("p = " + G.p.astype(str) + " · " + G.input + " · "
                 + G.normalize)
    views = []
    for p in sorted(G.p.unique()):
        for kind in ("gaussian", "constant", "tokens"):
            for n in ("none", "mean"):
                v = f"p = {p} · {kind} · {n}"
                if v in set(G.view):
                    views.append(v)
    default = "p = 4 · gaussian · none"
    if default in views:
        views.remove(default)
        views.insert(0, default)
    dtypes = present(G.dtype.unique(), DTYPES)
    fig, vis = lines(
        G, panel="semiring", panels=present(G.semiring.unique(), SEMIRINGS),
        x="t1", y="err_plot", series="dtype", order=dtypes, colors=DTYPE_COLOR,
        view="view", views=views, x_title="position t",
        y_title="error against float64", hover="t = %{x:,}<br>%{y:.2e}",
        hlines=[(1.0, "100 %")], height=380,
    )
    t_axis(fig, int(G["T"].max()))
    F = G[(G.dtype == "float32") & (G.t == G["T"] - 1)]
    text = ""
    if not F.empty:
        per = F.groupby("semiring")["err"].median().reindex(
            present(F.semiring.unique(), SEMIRINGS))
        text = ("float32 at the last position, median over runs: " + ", ".join(
            f"{s} {sci(v)}" for s, v in per.items()) +
            ". The sums (reals, log) lose accuracy roughly in proportion to t; the"
            " maximum (arctic, bayesian) does not accumulate rounding at all."
            " Pure bfloat16 and float16 are wrong by order one within a few"
            " thousand positions; under bf16 autocast the scans run in float32"
            " and the error stays at bfloat16's own resolution.")
    intro = ("<p>Error against float64 at every position, one panel per semiring,"
             " one line per format. Points drawn at 10 are non-finite outputs; the"
             " dotted line is an error of 100 %.</p>")
    return section("Precision against float64", intro,
                   figure(fig, vis, "Depth · input · normalisation"),
                   reading(text) if text else "", anchor="precision")


def primitive_section(df: pd.DataFrame) -> str:
    P = df[(df.experiment == "primitive") & (df.dtype != "float64")].copy()
    if P.empty:
        return ""
    scans = P[P.quantity != "arange"]
    names = present(scans.semiring.unique(), [
        "cumsum, time not innermost", "cumsum, time innermost",
        "logcumsumexp, time not innermost"])
    dtypes = present(scans.dtype.unique(), DTYPES)
    fig, _ = lines(
        scans, panel="semiring", panels=names, x="t1", y="err_plot",
        series="dtype", order=dtypes, colors=DTYPE_COLOR,
        x_title="position t", y_title="error against float64",
        hover="t = %{x:,}<br>%{y:.2e}", hlines=[(1.0, "100 %")], height=360,
    )
    t_axis(fig, int(P["T"].max()))
    A = P[P.quantity == "arange"].copy()
    A["err_pos"] = A["err"].replace(np.inf, np.nan)
    fig2, _ = lines(
        A[A.err_pos > 0], panel="semiring", panels=["positions arange(T)"],
        x="t1", y="err_pos", series="dtype", order=dtypes, colors=DTYPE_COLOR,
        x_title="position t", y_title="position error [positions]",
        hover="t ≤ %{x:,}<br>off by up to %{y:,.0f}", height=300, markers=True,
        panel_title=lambda _: "",
    )
    if fig2 is not None:
        t_axis(fig2, int(P["T"].max()))
    stuck = P[(P.semiring == "cumsum, time not innermost") & (P.dtype == "bfloat16")]
    text = ""
    if not stuck.empty:
        text = (f"A bfloat16 cumulative sum along a non-innermost dimension stops at"
                f" {stuck.mag_max.max():.0f}: PyTorch's CUDA scan over an outer"
                f" dimension accumulates in the input's own precision, and once the"
                f" running sum is 256 times its increments, adding one changes"
                f" nothing. A level's state has time on dimension 1, so every LISS"
                f" scan in pure bfloat16 stagnates this way. In float32 the same"
                f" scan accumulates position by position, which is why its error"
                f" grows with t, about ten times faster than the innermost scan's"
                f" tree. bfloat16 also cannot tell neighbouring positions apart"
                f" beyond 256, and float16 cannot represent positions past 65,504,"
                f" so a decay or a normalisation count computed in those formats is"
                f" wrong before any sum is.")
    intro = ("<p>The PyTorch operations a level is built from, alone, on random"
             " inputs in a level's state layout (time on dimension 1 of"
             " <code>(B, T, N, R, d_v, w)</code>) and with time innermost. The lower"
             " chart is the furthest <code>arange(T)</code> in each format puts any"
             " position up to t from where it is; decays and normalisation counts"
             " are computed from it in the level's dtype.</p>")
    return section("Where the error comes from", intro, figure(fig),
                   figure(fig2, note="Every format holds the positions exactly."),
                   reading(text) if text else "", anchor="primitives")


def normalize_section(df: pd.DataFrame) -> str:
    G = df[(df.experiment == "growth") & (df.dtype == "float64")
           & (df.semiring == "reals")].copy()
    if G.empty:
        return ""
    G["view"] = "p = " + G.p.astype(str)
    depths = sorted(G.p.unique())
    views = [f"p = {p}" for p in depths]
    if "p = 4" in views:
        views.remove("p = 4")
        views.insert(0, "p = 4")
    fig, vis = lines(
        G, panel="input", panels=present(G.input.unique(), INPUTS), x="t1",
        y="mag_med", series="normalize", order=present(G.normalize.unique(), NORMS),
        colors=NORM_COLOR, view="view", views=views, x_title="position t",
        y_title="median |ISS|", hover="t = %{x:,}<br>%{y:.3g}", height=380,
    )
    t_axis(fig, int(G["T"].max()))
    intro = ("<p>The reals without a decay, normalised three ways: the median"
             " output at each position (float64), one panel per input. A"
             " normalisation that fits keeps the line flat.</p>")
    return section("Normalising over the time axis", intro,
                   figure(fig, vis, "Depth"), anchor="normalize")


def decay_section(df: pd.DataFrame) -> str:
    D = df[(df.experiment == "decay") & (df.dtype == "float64")].copy()
    if D.empty:
        return ""
    D["x"] = D.t1 / D.Tc
    D["view"] = (D.normalize + " · p = " + D.p.astype(str) + " · " + D.input)
    views = []
    for n in ("none", "mean", "ema"):
        for p in sorted(D.p.unique()):
            for kind in ("constant", "gaussian"):
                v = f"{n} · p = {p} · {kind}"
                if v in set(D.view):
                    views.append(v)
    alphas = sorted(D.alpha.unique(), key=lambda a: (a < 0, abs(a)))
    colors = {}
    pos = [a for a in alphas if a > 0]
    neg = [a for a in alphas if a < 0]
    for i, a in enumerate(pos):
        colors[a] = (RAMP[2][0], RAMP[5][0])[min(i, 1)]
    for i, a in enumerate(neg):
        colors[a] = (SLOTS[1][0], SLOTS[7][0])[min(i, 1)]
    fig, vis = lines(
        D, panel="semiring", panels=present(D.semiring.unique(), SEMIRINGS),
        x="x", y="mag_max", series="alpha", order=alphas, colors=colors,
        label=lambda a: f"α = {a:+g}", view="view", views=views,
        x_title="t / context length", y_title="largest |stored value|",
        hover="t = %{x:.3g} T<sub>c</sub><br>%{y:.3g}",
        hlines=[(FLOAT32_MAX, "float32")], height=380, shared_y=False,
        hline_panels=["reals", "bayesian"],
    )
    fig.update_xaxes(tickvals=[0.01, 0.1, 1, 10, 100],
                     ticktext=["0.01", "0.1", "1", "10", "100"])
    F = df[(df.experiment == "decay") & (df.dtype == "float32")]
    text = ""
    bad = F[(F.nonfinite > 0)]
    if not bad.empty:
        first = bad.groupby(["semiring", "alpha"]).apply(
            lambda g: (g.t1 / g.Tc).min(), include_groups=False)
        parts = [f"{s} α = {a:+g} at t = {x:.0f} T<sub>c</sub>"
                 for (s, a), x in first.items()]
        text = ("float32 outputs turn non-finite for: " + "; ".join(parts) +
                ". Only the growing decays (α &lt; 0) of the reals and the bayesian"
                " semiring overflow: once e<sup>α t/T<sub>c</sub></sup> times the"
                " number of tuples passes 3.4·10<sup>38</sup> (88 nats), whatever the"
                " normalisation, which divides by a count and cannot undo an"
                " exponential. The log domain carries the same growth as a sum.")
    M = D[(D.semiring == "reals") & (D.alpha > 0) & (D.normalize.isin(["mean", "ema"]))
          & (D.input == "constant") & (D.p == D.p.min())]
    if not M.empty:
        end = M[M.t == M["T"] - 1].set_index(["normalize", "alpha"])["mag_med"]
        one = M.iloc[(M.x - 1).abs().argsort()].groupby(["normalize", "alpha"])["mag_med"].first()
        ratio = (end / one).dropna()
        if not ratio.empty:
            text += (" With a shrinking decay (α &gt; 0), <code>mean</code> divides by"
                     " t while only about T<sub>c</sub>/α positions still count, so"
                     " the output keeps falling past the context length ("
                     + ", ".join(f"{n} α = {a:+g}: {v:.2g}× its value at t = T<sub>c</sub>"
                                 for (n, a), v in ratio.items())
                     + " at the end, constant input).")
    intro = ("<p>A decay of rate α/T<sub>c</sub> per position (λ = e<sup>−α/T<sub>c</sub></sup>,"
             " the rate the kernel's <code>alpha_0·tanh(a)</code> reaches at saturation),"
             " run to 128 context lengths. Positive α shrinks old tuples; negative α"
             " favours the distant past and grows. The selector picks the"
             " normalisation (<code>ema</code> is the benchmark-only prototype), the"
             " depth and the input. These runs are long enough for the real and the"
             " bayesian scan to take their Hillis–Steele path.</p>")
    return section("Decays and the context length", intro,
                   figure(fig, vis, "Normalisation · depth · input"),
                   reading(text) if text else "", anchor="decay")


def kernel_section(df: pd.DataFrame) -> str:
    K = df[(df.experiment == "kernel")]
    if K.empty:
        return ""
    agg = K.groupby(["semiring", "offset", "scale", "restrict", "dtype"],
                    as_index=False).agg(nonfinite=("nonfinite", "mean"),
                                        mag=("mag_max", "max"))
    F = agg[agg.dtype == "float32"].copy()
    off = F[F.scale == 1].copy()
    off["panel"] = "common offset of queries and keys"
    off["xv"] = off["offset"]
    sca = F[(F.scale > 1) | (F.offset == 0)].copy()
    sca = sca[sca.offset == 0]
    sca["panel"] = np.where(sca["restrict"].astype(str).str.lower() == "true",
                            "scale, restrict on", "scale, restrict off")
    sca["xv"] = sca["scale"]
    both = pd.concat([off, sca])
    panels = ["common offset of queries and keys", "scale, restrict off",
              "scale, restrict on"]
    fig, _ = lines(
        both, panel="panel", panels=[p for p in panels if p in set(both.panel)],
        x="xv", y="nonfinite", series="semiring",
        order=present(both.semiring.unique(), SEMIRINGS), colors=SEMIRING_COLOR,
        log_x=False, log_y=False, x_title="", y_title="non-finite outputs",
        hover="%{x}<br>%{y:.0%}", height=340, markers=True,
    )
    fig.update_yaxes(tickformat=".0%", range=[-0.05, 1.05])
    fig.update_xaxes(title_text="offset", row=1, col=1)
    for c in (2, 3):
        fig.update_xaxes(title_text="query/key weight scale", type="log",
                         tickvals=[1, 2, 4, 8, 16, 32],
                         ticktext=["1", "2", "4", "8", "16", "32"], row=1, col=c)
    ref = agg[(agg.dtype == "float64") & (agg.scale == 1) & (agg.semiring == "reals")]
    text = ""
    if not ref.empty:
        spread = ref.mag.max() / ref.mag.min() - 1
        text = (f"In float64 the reals output is the same at every offset (the"
                f" largest values agree to {sci(spread)}), since exp(q + c − (k + c)) ="
                f" exp(q − k). The factors exp(q + c) and exp(−k − c) are"
                f" stored separately, and float32 loses them at an offset of about 88"
                f" (e<sup>88</sup> ≈ 1.7·10<sup>38</sup>). Nothing in the loss holds the"
                f" offset back, since it does not change the kernel: only weight decay"
                f" does. The log-domain semirings add q and −k and never form the"
                f" exponential. A larger scale grows the kernel itself in every"
                f" semiring that stores products, until <code>restrict</code> bounds"
                f" queries and keys with tanh.")
    intro = ("<p>The exponential kernel exp(q(x<sub>t′</sub>) − k(x<sub>t</sub>)) in"
             " the reals and the bayesian semiring is evaluated as a product of"
             " exp(q) at the later index and exp(−k) at the earlier one. Left: both"
             " biases shifted by the same constant, which leaves the kernel exactly"
             " unchanged. Middle and right: the query and key weights scaled up, as"
             " training may do, without and with <code>restrict</code>. Each point is"
             " the fraction of non-finite float32 outputs over a run of 4,096"
             " positions at p = 2.</p>")
    return section("The exponential kernel's factors", intro, figure(fig),
                   reading(text) if text else "", anchor="kernel")


def gradient_section(df: pd.DataFrame) -> str:
    G = df[df.experiment == "gradient"].copy()
    if G.empty:
        return ""
    G["series"] = G.normalize + " · p = " + G.p.astype(str)
    order = [f"{n} · p = {p}" for n in ("none", "mean") for p in sorted(G.p.unique())]
    order = [o for o in order if o in set(G.series)]
    colors = {o: c for o, c in zip(order, [NEUTRAL, "#52514e", SLOTS[0][0], RAMP[4][0]])}
    ref = G[G.dtype == "float64"]
    fig, _ = lines(
        ref, panel="semiring", panels=present(ref.semiring.unique(), SEMIRINGS),
        x="t1", y="mag_max", series="series", order=order, colors=colors,
        x_title="input position s", y_title="largest |∂L/∂x<sub>s</sub>|",
        hover="s = %{x:,}<br>%{y:.3g}", height=360, shared_y=False,
    )
    t_axis(fig, int(G["T"].max()))
    E = G[G.dtype != "float64"].copy()
    E["view"] = E.dtype
    views = present(E.dtype.unique(), DTYPES)
    fig2, vis2 = lines(
        E, panel="semiring", panels=present(E.semiring.unique(), SEMIRINGS),
        x="t1", y="err_plot", series="series", order=order, colors=colors,
        view="view", views=views, x_title="input position s",
        y_title="gradient error against float64",
        hover="s = %{x:,}<br>%{y:.2e}", height=340,
    )
    if fig2 is not None:
        t_axis(fig2, int(G["T"].max()))
    R = ref[(ref.semiring == "reals")]
    text = ""
    if not R.empty:
        first = R[R.t == 0].set_index("series")["mag_max"]
        text = ("The gradient reaching the first input position of a reals level,"
                f" T = {compact(R['T'].max())}: " + ", ".join(
                    f"{k} {sci(v)}" for k, v in first.items()) +
                ". It grows with the number of outputs a position feeds, the same"
                " polynomial as the forward pass; normalisation tames it too.")
    nf = E[E.nonfinite > 0]
    if nf.empty and not E.empty:
        text += " No gradient was non-finite in any format tested."
    intro = ("<p>The gradient of sum(y · r), r fixed and random, with respect to the"
             " input of the level, at the default kernels (decay and exponential at"
             " initialisation) and an independent input. Top: its size at every"
             " input position (float64). Bottom: its error against float64.</p>")
    return section("Gradients", intro, figure(fig), figure(fig2, vis2, "Format"),
                   reading(text) if text else "", anchor="gradient")


SCAN_COLOR = {"torch": NEUTRAL, "triton": SLOTS[0][0]}
SCAN_LABEL = {"torch": "PyTorch scans", "triton": "fused Triton scans"}


def scan_section(df: pd.DataFrame) -> str:
    """The PyTorch path against ``scan: triton``, run for run."""
    X = df[df.experiment.isin(["growth", "decay", "gradient"])
           & (df.dtype != "float64")]
    if "triton" not in set(X.scan):
        return ""
    X = X[X.set_index(["experiment", "semiring", "p", "normalize", "input",
                       "kernels", "alpha"]).index.isin(
        X[X.scan == "triton"].set_index(["experiment", "semiring", "p",
                                         "normalize", "input", "kernels",
                                         "alpha"]).index)].copy()
    agg = X.groupby(["experiment", "semiring", "scan", "dtype", "t1"],
                    as_index=False)["err_plot"].median()
    agg["view"] = agg.experiment + " · " + agg.dtype
    views = [f"{e} · {d}" for e in ("growth", "decay", "gradient")
             for d in DTYPES if f"{e} · {d}" in set(agg.view)]
    fig, vis = lines(
        agg, panel="semiring",
        panels=present(agg.semiring.unique(), SEMIRINGS), x="t1",
        y="err_plot", series="scan", order=present(agg.scan.unique(),
                                                    ["torch", "triton"]),
        colors=SCAN_COLOR, label=lambda k: SCAN_LABEL.get(k, k), view="view",
        views=views, x_title="position t",
        y_title="median error against float64",
        hover="t = %{x:,}<br>%{y:.2e}", hlines=[(1.0, "100 %")], height=380,
    )
    t_axis(fig, int(X["T"].max()))
    last = X[(X.experiment == "growth") & (X.dtype == "float32")
             & (X.t == X["T"] - 1)]
    text = ""
    if not last.empty:
        per = last.groupby(["semiring", "scan"])["err"].median().unstack()
        parts = [f"{s} {sci(r['torch'])} against {sci(r['triton'])}"
                 for s, r in per.reindex(present(per.index, SEMIRINGS)).iterrows()
                 if {"torch", "triton"} <= set(r.index)]
        if parts:
            text = (f"float32 at t = {compact(int(last['T'].max()))}, median over"
                    " the growth runs, PyTorch against Triton: " + ", ".join(parts)
                    + ". A fused scan rounds through one chunk and the chain of"
                    " chunk states, not through every position, and applies the"
                    " decay step by step instead of as e<sup>±βt</sup> or an"
                    " offset βt that grows with the position.")
    intro = ("<p>The same runs on <code>scan: triton</code>, the fused Triton"
             " scans (reals, log, arctic), against the same float64 PyTorch"
             " reference. The lines are medians over the runs of an experiment;"
             " the kernels accumulate in float32 whatever the storage format, so"
             " the pure bfloat16 and float16 rows measure the inputs' rounding"
             " only.</p>")
    return section("PyTorch scans against the fused Triton scans", intro,
                   figure(fig, vis, "Experiment · format"),
                   reading(text) if text else "", anchor="triton")


def summary_section(df: pd.DataFrame) -> str:
    G = df[(df.experiment == "growth") & (df.dtype != "float64")]
    if G.empty:
        return ""
    rows = []
    for dtype in present(G.dtype.unique(), DTYPES):
        cells = [f"<code>{html.escape(dtype)}</code>"]
        for s in SEMIRINGS:
            g = G[(G.dtype == dtype) & (G.semiring == s)]
            if g.empty:
                cells.append("–")
                continue
            last = g[g.t == g["T"] - 1]
            fin = last[np.isfinite(last.err)]
            bad = g[g.nonfinite > 0]
            parts = [f"error {sci(fin.err.median())}" if not fin.empty else "error –"]
            for limit in (0.01, 0.1):
                worse = g[g.err > limit]
                parts.append(f"{limit:.0%} from t = {compact(worse.t1.min())}"
                             if not worse.empty else f"never {limit:.0%}")
            parts.append(f"non-finite from t = {compact(bad.t1.min())}"
                         if not bad.empty else "always finite")
            cells.append("<br>".join(parts))
        rows.append(cells)
    T = compact(G["T"].max())
    intro = (f"<p>Every growth run (all normalisations, depths and inputs) by format"
             f" and semiring: the median error at t = {T}, the first positions where"
             f" any run's error passes 1 % and 10 %, and the first non-finite"
             f" output.</p>")
    return section("Where each format breaks", intro,
                   table(["format"] + SEMIRINGS, rows), anchor="summary")


GLOSSARY = [
    ("Quantities", [
        ("p", "depth of the level: indices per tuple."),
        ("t, T", "a position (counted from one in the charts) and the run's length."),
        ("T<sub>c</sub>", "the context length: decays are measured in t/T<sub>c</sub>."),
        ("α", "the decay rate times T<sub>c</sub>; λ = e<sup>−α/T<sub>c</sub></sup> per position."),
    ]),
    ("Formats", [
        ("float32", "the training default; 24-bit mantissa, range 3.4·10<sup>38</sup>."),
        ("bfloat16", "8-bit mantissa, float32's range; the whole level cast to it."),
        ("float16", "11-bit mantissa, range 65,504; the whole level cast to it."),
        ("bf16-mixed", "float32 weights under torch.autocast(bfloat16)."),
    ]),
    ("Normalisation", [
        ("none", "partial sums as they are."),
        ("mean", "divided by the count of positions summed."),
        ("sqrt", "divided by the square root of the count."),
        ("ema", "prototype, this benchmark only: divided by the decayed count Σ λ<sup>t−s</sup>."),
    ]),
]


def build(df: pd.DataFrame, meta: dict, sources: list[str], fragment: bool) -> str:
    head = " · ".join(x for x in [
        html.escape(meta.get("device", "")),
        f"torch {html.escape(meta.get('torch', ''))}",
        meta.get("date", "")[:10],
        f"{df.groupby(['experiment', 'semiring', 'p', 'normalize', 'input', 'alpha', 'offset', 'scale', 'restrict']).ngroups} runs",
    ] if x)
    everything, df = df, df[df.scan == "torch"]
    body = "".join([
        lead(meta), kpis(df), glossary(GLOSSARY),
        growth_section(df), exponent_section(df), normalize_section(df),
        precision_section(df), primitive_section(df), decay_section(df),
        kernel_section(df), gradient_section(df), scan_section(everything),
        summary_section(df),
        "<footer>Generated by <code>plot_stability.py</code> from "
        + ", ".join(f"<code>{html.escape(s)}</code>" for s in sources)
        + ". Interactive: hover for values, click legend entries to hide series,"
        " drag to zoom.</footer>",
    ])
    return page("LISS Numerics", "LISS numerics over long sequences", head, body,
                fragment=fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fragment", type=Path, default=None,
                        help="also write the page without its html/head/body"
                             " skeleton (for publishing as an artifact)")
    args = parser.parse_args()
    df = load(args.csv)
    meta_path = args.csv[0].with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = args.out or args.csv[0].with_suffix(".html")
    names = [p.name for p in args.csv]
    out.write_text(build(df, meta, names, False))
    print(f"Wrote {out}")
    if args.fragment:
        # an artifact viewer cannot save files, so no "download as png"
        PLOT_CONFIG["modeBarButtonsToRemove"].append("toImage")
        args.fragment.write_text(build(df, meta, names, True))
        print(f"Wrote {args.fragment}")


if __name__ == "__main__":
    main()
