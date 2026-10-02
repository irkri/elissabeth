"""Render a ``benchmark.py`` CSV (and the JSON next to it) as one
self-contained HTML report.

The report adapts to what the CSV holds: a section whose experiment is
missing says so, and every implementation in the CSV appears in the
implementation charts, so a new entry of ``implementations.py`` needs no
change here. Times are the best of a cell's timed steps by default
(``--stat min``): on a GPU other jobs share, the fastest step is the one
least disturbed by them; ``--stat median`` uses the median instead.

    cd project
    python benchmark/plot_benchmark.py benchmark/liss01.csv          # -> liss01.html
    python benchmark/plot_benchmark.py benchmark/liss01.csv --fragment liss01.frag.html
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

from report import (PLOT_CONFIG, NEUTRAL, RAMP, SEMIRING_COLOR, SLOTS, apply_log_ticks,
                    compact, figure, glossary, log_ticks, page, ramp, reading,
                    sci, section, style, table, tiles)

SETUPS = ["reals", "log", "arctic", "bayesian", "reals · cosine"]
SETUP_NOTE = {
    "reals": "decay × exponential, rank 1",
    "log": "decay × exponential, rank 1",
    "arctic": "decay × exponential, rank 1",
    "bayesian": "decay × exponential, rank 1",
    "reals · cosine": "decay × cosine (d_qk 3), rank 8",
}
VIEW_ORDER = ["compiled · train", "triton · train", "eager · train",
              "native · train", "compiled · infer", "triton · infer",
              "eager · infer"]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(path: Path, stat: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    suffix = "_min" if stat == "min" else ""
    for name in ("total", "fwd", "bwd"):
        column = f"{name}_ms{suffix}"
        if column not in df:            # CSVs from before the _min columns
            column = f"{name}_ms"
        df[name] = pd.to_numeric(df[column], errors="coerce")
    df["ok"] = df["status"].eq("ok")
    cosine = df["kernels"].fillna("").eq("decay+cos")
    df["setup"] = np.where(cosine, df["semiring"] + " · cosine", df["semiring"])
    df["view"] = df["impl"] + " · " + df["task"]
    df["ns_per_entry"] = df["total"] * 1e6 / df["work"]
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    return df


def implementations(df: pd.DataFrame, meta: dict) -> list[str]:
    order = list(meta.get("implementations", {}))
    present = list(df["impl"].dropna().unique())
    return [i for i in order if i in present] + [
        i for i in present if i not in order]


def impl_color(impls: list[str]) -> dict[str, str]:
    """Eager is the neutral baseline; the others take the categorical slots
    in registry order, so a new implementation keeps the others' colours."""
    colors, slot = {}, 0
    for impl in impls:
        if impl == "eager":
            colors[impl] = NEUTRAL
        else:
            colors[impl] = SLOTS[slot % len(SLOTS)][0]
            slot += 1
    return colors


def p_color(depths: list[float]) -> dict[float, str]:
    return dict(zip(depths, ramp(len(depths))))


def t_ticks(values) -> dict:
    """Length ticks, every other one so five panels side by side stay
    legible, horizontal."""
    vals = sorted(v for v in values if v == v)
    if len(vals) > 4:
        vals = vals[::2]
    return dict(**log_ticks(vals), tickangle=0)


def present(values, order) -> list:
    have = set(values)
    return [v for v in order if v in have]


# --------------------------------------------------------------------------
# figure builder: small multiples of lines, optionally with views
# --------------------------------------------------------------------------
def panel_lines(
    data: pd.DataFrame, *, panel: str, panels: list, x: str, y: str,
    series: str, order: list, colors: dict, label=str, view: str | None = None,
    views: list | None = None, refs: pd.DataFrame | None = None,
    ref_name: str = "", log_x: bool = True, log_y: bool = True,
    x_title: str = "", y_title: str = "", hover: str = "", cols: int | None = None,
    height: int = 380, shared_y: bool = True, panel_title=None,
) -> tuple[go.Figure | None, dict | None]:
    if data.empty or not panels:
        return None, None
    cols = cols or len(panels)
    rows = math.ceil(len(panels) / cols)
    titles = [(panel_title or str)(p) for p in panels]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles,
                        shared_yaxes=shared_y, horizontal_spacing=0.035,
                        vertical_spacing=0.16 if rows > 1 else 0.1)
    keys = views if views else [None]
    vis: dict = {str(k): [] for k in keys}
    for key in keys:
        sub = data if key is None else data[data[view] == key]
        ref = None
        if refs is not None:
            ref = refs if key is None else refs[refs[view] == key]
        shown: set = set()
        for i, pn in enumerate(panels):
            r, c = divmod(i, cols)
            if ref is not None and not ref.empty:
                d = ref.sort_values(x)
                fig.add_trace(go.Scatter(
                    x=d[x], y=d[y], name=ref_name, legendgroup="ref",
                    showlegend="ref" not in shown, mode="lines",
                    line=dict(color=NEUTRAL, width=2, dash="dot"),
                    hovertemplate=f"{ref_name}<br>%{{x}}<br>%{{y:.3g}}<extra></extra>",
                ), row=r + 1, col=c + 1)
                shown.add("ref")
                for k in keys:
                    vis[str(k)].append(k == key)
            for s in order:
                d = sub[(sub[panel] == pn) & (sub[series] == s)].sort_values(x)
                d = d[np.isfinite(d[y].astype(float))]
                if d.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=d[x], y=d[y], name=label(s), legendgroup=str(s),
                    showlegend=s not in shown, mode="lines+markers",
                    line=dict(color=colors[s], width=2),
                    marker=dict(size=7, color=colors[s],
                                line=dict(width=1.5, color="#fcfcfb")),
                    customdata=d[["T", "p"]].to_numpy() if "T" in d else None,
                    hovertemplate=f"{label(s)} · {pn}<br>{hover}<extra></extra>",
                ), row=r + 1, col=c + 1)
                shown.add(s)
                for k in keys:
                    vis[str(k)].append(k == key)
    fig.update_xaxes(type="log" if log_x else "linear", title_text="")
    fig.update_yaxes(type="log" if log_y else "linear")
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text=x_title, row=rows, col=c)
    for r in range(1, rows + 1):
        fig.update_yaxes(title_text=y_title, row=r, col=1)
    style(fig, height)
    if log_y:
        apply_log_ticks(fig, shared_y)
    return fig, (vis if views else None)


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def memory_caps(meta: dict) -> list[float]:
    """The GPU memory caps of the sessions that wrote the CSV."""
    return sorted({s.get("args", {}).get("memory_cap_gb")
                   for s in meta.get("sessions", [])} - {None})


def lead(meta: dict, df: pd.DataFrame, stat: str) -> str:
    s = meta.get("sessions", [{}])[-1].get("args", {})
    B, N, dv, d = (s.get("batch", "?"), s.get("heads", "?"),
                   s.get("d_values", "?"), s.get("d_hidden", "?"))
    caps = memory_caps(meta)
    cap = "–".join(f"{c:g}" for c in (caps[0], caps[-1]) if c) if caps else ""
    if len(caps) > 1 and caps[0] != caps[-1]:
        cap += " (per session)"
    elif caps:
        cap = f"{caps[0]:g}"
    cap_text = (f" The process was capped at {cap} GiB of GPU memory, so"
                f" a cell that needs more reports out of memory." if caps else "")
    which = "the fastest of a cell's timed steps" if stat == "min" else "the median timed step"
    sessions = meta.get("sessions", [{}])
    together = (
        "Cells that a chart compares (the implementations and tasks of a"
        " configuration, the depths at one length, the ranks or options of a"
        " semiring)" if sessions and sessions[0].get("timing") == "groups"
        else "The implementations and tasks of each configuration"
    )
    return f"""<div class="lead">
<p>A LISS level of depth <i>p</i> sums, for every head and position <i>t</i>,
over all index tuples <i>t</i><sub>1</sub> &lt; … &lt; <i>t</i><sub>p</sub> ≤ <i>t</i>
the semiring product of the values at those indices and of the kernels between
neighbouring indices. Because every kernel factorises into a query feature at the
later index and a key feature at the earlier one, times a decay, the sum is not
evaluated over the <i>O</i>(<i>T</i><sup>p</sup>) tuples but as <i>p</i> scans:</p>
<div class="formula">S<sub>1</sub>(t) = ⊕<sub>s≤t</sub> λ<sub>1</sub><sup>t−s</sup> ⊗ ψ<sub>1</sub>(s) ⊗ v<sub>1</sub>(s)<br>
S<sub>l</sub>(t) = ⊕<sub>s≤t</sub> λ<sub>l</sub><sup>t−s</sup> ⊗ ψ<sub>l</sub>(s) ⊗ ⟨φ<sub>l−1</sub>(s), S<sub>l−1</sub>(s−1)⟩ ⊗ v<sub>l</sub>(s)<br>
ISS<sup>(p)</sup>(t) = ⟨φ<sub>p</sub>(t), S<sub>p</sub>(t)⟩ = ⊕<sub>r</sub> φ<sub>p,r</sub>(t) ⊗ S<sub>p,r</sub>(t)</div>
<p>Each scan carries a state of shape <code>(B, T, N, R, d_v)</code>: batch, time,
heads, kernel rank and value width. The work of a level is therefore
<i>p</i>·<i>B</i>·<i>T</i>·<i>N</i>·<i>R</i>·<i>d<sub>v</sub></i> state entries, linear in
both the length and the depth, and the backward pass keeps every state for the
gradient. The four semirings differ only in ⊕ and ⊗: a sum and a product
(<code>reals</code>), logsumexp and a sum (<code>log</code>), a maximum and a sum
(<code>arctic</code>), a maximum and a product (<code>bayesian</code>).</p>
<p>Each cell builds one causal LISS layer (<i>B</i> = {B}, <i>N</i> = {N} heads,
<i>d<sub>v</sub></i> = {dv}, model width {d}, context length = <i>T</i>) and times a
training step, the forward and backward pass of <code>sum(y · r)</code> for a fixed
random <code>r</code> (<b>train</b>), and the forward pass alone under
<code>no_grad</code> (<b>infer</b>). Times are {which}; the other is in each
point's hover. {together} were timed in turns, one step each, so they shared
whatever load other jobs put on the GPU. Memory is the peak a step allocates on top of the weights and the
input. Every cell's output and input gradient are compared with a float64
evaluation of the same weights.{cap_text}</p></div>"""


def kpis(df: pd.DataFrame) -> str:
    items = []
    L = df[(df.experiment == "length") & df.ok & (df.task == "train")]
    piv = L.pivot_table(index=["setup", "p", "T"], columns="impl",
                        values="total", aggfunc="min")
    if {"eager", "compiled"} <= set(piv.columns):
        sp = (piv["eager"] / piv["compiled"]).dropna()
        if not sp.empty:
            items.append(("Compiled over eager", f"{sp.median():.1f}×",
                          f"median training speed-up, {sp.min():.1f}–{sp.max():.1f}× over {len(sp)} cells"))
    C = L[L.impl == "compiled"]
    if not C.empty:
        Ts = sorted(C["T"].unique())
        Tref = Ts[len(Ts) // 2 if len(Ts) < 3 else -2]
        sub = C[C["T"] == Tref].pivot_table(index="setup", columns="p",
                                            values="total", aggfunc="min")
        if sub.shape[1] > 1:
            lo, hi = sub.columns.min(), sub.columns.max()
            ratio = (sub[hi] / sub[lo]).dropna()
            if not ratio.empty:
                items.append((f"Depth {hi:g} against depth {lo:g}",
                              f"{ratio.median():.1f}×",
                              f"compiled training step at T = {compact(Tref)}"
                              f" (work ratio {hi / lo:g}×)"))
        fit = C[(C["p"] == 3) & (C.setup == "reals")]
        if not fit.empty:
            items.append(("Longest length run",
                          f"{compact(fit['T'].max())}",
                          "positions, compiled training at p = 3 (reals)"))
    A = df[(df.experiment == "attention") & df.ok & (df.task == "train")
           & (df.impl == "compiled")]
    R = C[(C.setup == "reals") & (C["p"] == 2)] if not C.empty else C
    if not A.empty and not R.empty:
        common = sorted(set(A["T"]) & set(R["T"]))
        if common:
            T = common[-1]
            ratio = (A[A["T"] == T]["total"].min()
                     / R[R["T"] == T]["total"].min())
            items.append(("Attention against LISS p = 2",
                          f"{ratio:.1f}×",
                          f"softmax attention's step time over a reals level's,"
                          f" T = {compact(T)}"))
    E = df[df.ok & df.err_out.notna()]
    if not E.empty:
        items.append(("Largest output error", sci(E["err_out"].max()),
                      "relative to float64, any cell"))
    return tiles(items)


def length_section(df: pd.DataFrame) -> str:
    L = df[(df.experiment == "length") & df.ok]
    A = df[(df.experiment == "attention") & df.ok]
    if L.empty:
        return section("Time against length", "<p>No length experiment in this CSV.</p>")
    depths = sorted(L["p"].dropna().unique())
    colors = p_color(depths)
    views = present(L["view"].unique(), VIEW_ORDER) + sorted(
        set(L["view"]) - set(VIEW_ORDER))
    panels = present(L["setup"].unique(), SETUPS)
    fig, vis = panel_lines(
        L, panel="setup", panels=panels, x="T", y="total", series="p",
        order=depths, colors=colors, label=lambda p: f"p = {p:g}",
        view="view", views=views, refs=A, ref_name="softmax attention",
        x_title="length T", y_title="time per step [ms]",
        hover="T = %{x:,}<br>%{y:.3g} ms", height=400,
    )
    fig.update_xaxes(**t_ticks(L["T"].unique()))
    # the reading: exponent of T over the last doubling-range, compiled train
    C = L[(L.impl == "compiled") & (L.task == "train")]
    slopes = []
    for (setup, p), g in C.groupby(["setup", "p"]):
        g = g.sort_values("T")
        if len(g) >= 3:
            a, b = g.iloc[-3], g.iloc[-1]
            slopes.append(math.log(b["total"] / a["total"]) / math.log(b["T"] / a["T"]))
    text = ""
    if slopes:
        text = (f"Over the longest lengths the compiled training step grows like "
                f"T<sup>{np.median(slopes):.2f}</sup> (median over semirings and "
                f"depths, range {min(slopes):.2f}–{max(slopes):.2f}): linear, as the "
                f"recursion promises. Below a few thousand positions the step is "
                f"dominated by launch overhead and barely depends on T.")
    att = ""
    if not A.empty:
        att = (" The dotted line in every panel is softmax attention (PyTorch's"
               " scaled-dot-product kernel, causal, same width and heads) in the"
               " same implementation and task, for scale.")
    intro = (f"<p>Step time against the length, one panel per semiring, one line"
             f" per depth. Both axes are logarithmic, so a straight line is a power"
             f" law and a slope of one is linear cost. The selector switches the"
             f" implementation and between the training step and the forward pass"
             f" alone.{att}</p>")
    return section("Time against length", intro, figure(fig, vis, "Implementation · task"),
                   reading(text) if text else "", anchor="length")


def depth_section(df: pd.DataFrame) -> str:
    L = df[(df.experiment == "length") & df.ok]
    if L.empty:
        return ""
    Ts = sorted(L["T"].unique())
    pick = sorted({Ts[0], Ts[len(Ts) // 2], Ts[-2] if len(Ts) > 2 else Ts[-1]})
    data = L[L["T"].isin(pick)]
    setups = present(data["setup"].unique(), SETUPS)
    views = present(data["view"].unique(), VIEW_ORDER)
    fig, vis = panel_lines(
        data, panel="T", panels=pick, x="p", y="total", series="setup",
        order=setups, colors=SEMIRING_COLOR, view="view", views=views,
        log_x=False, log_y=False, shared_y=False, x_title="depth p",
        y_title="time per step [ms]", hover="p = %{x}<br>%{y:.3g} ms",
        panel_title=lambda T: f"T = {T:,}", height=380,
    )
    fig.update_xaxes(dtick=1)
    C = L[(L.impl == "compiled") & (L.task == "train") & (L["T"] == pick[-1])]
    text = ""
    if not C.empty:
        per = []
        for setup, g in C.groupby("setup"):
            g = g.sort_values("p")
            if len(g) >= 2:
                slope = np.polyfit(g["p"], g["total"], 1)
                per.append((setup, slope[0], slope[1]))
        if per:
            parts = ", ".join(f"{s} {a:.1f} ms" for s, a, _ in per)
            text = (f"At T = {pick[-1]:,} every extra level adds a near-constant"
                    f" time to the compiled training step: {parts} per level"
                    f" (least-squares slope over p).")
    intro = ("<p>The same cells against the depth <i>p</i>, at a short, a middle"
             " and a long length. The work grows linearly in <i>p</i>, and so"
             " should the time: each level is one more scan, one more"
             " contraction and one more saved state.</p>")
    return section("Time against depth", intro, figure(fig, vis, "Implementation · task"),
                   reading(text) if text else "", anchor="depth")


def implementation_section(df: pd.DataFrame, impls: list[str],
                           meta: dict) -> str:
    L = df[(df.experiment == "length") & (df.task == "train")]
    ok = L[L.ok]
    if ok.empty or "eager" not in set(ok.impl):
        return section("Implementations", "<p>No eager baseline in this CSV.</p>")
    colors = impl_color(impls)
    eager = ok[ok.impl == "eager"].set_index(["setup", "p", "T"])["total"]
    rows = ok[ok.impl != "eager"].copy()
    rows["speedup"] = [
        eager.get((r.setup, r.p, r.T), np.nan) / r.total
        for r in rows.itertuples()
    ]
    depths = sorted(rows["p"].dropna().unique())
    others = [i for i in impls if i != "eager" and i in set(rows.impl)]
    panels = present(rows["setup"].unique(), SETUPS)
    rows["pv"] = rows["p"].map(lambda p: f"p = {p:g}")
    views = [f"p = {p:g}" for p in depths]
    fig, vis = panel_lines(
        rows, panel="setup", panels=panels, x="T", y="speedup", series="impl",
        order=others, colors=colors, view="pv", views=views,
        x_title="length T", y_title="speed-up over eager [×]",
        hover="T = %{x:,}<br>%{y:.2f}×", height=380,
    )
    if fig is not None:
        fig.add_hline(y=1, line=dict(color=NEUTRAL, width=1, dash="dot"),
                      row="all", col="all")
        fig.update_xaxes(**t_ticks(rows["T"].unique()))
        fig.update_yaxes(**log_ticks([0.25, 0.5, 1, 2, 4, 8, 16]))
    # throughput
    ok2 = ok.copy()
    ok2["pv"] = ok2["p"].map(lambda p: f"p = {p:g}")
    order_all = [i for i in impls if i in set(ok2.impl)]
    fig2, vis2 = panel_lines(
        ok2, panel="setup", panels=panels, x="T", y="ns_per_entry",
        series="impl", order=order_all, colors=colors, view="pv", views=views,
        x_title="length T", y_title="ns per state entry and level",
        hover="T = %{x:,}<br>%{y:.3g} ns", height=360,
    )
    if fig2 is not None:
        fig2.update_xaxes(**t_ticks(ok2["T"].unique()))
    # failures of non-eager implementations, summarised
    fails = L[~L.ok & (L.impl != "eager")]
    fail_text = ""
    if not fails.empty:
        parts = []
        for (impl, setup, status), g in fails.groupby(["impl", "setup", "status"]):
            parts.append(f"<code>{html.escape(impl)}</code> on {html.escape(setup)}"
                         f" ({len(g)} cells, T {compact(g['T'].min())}–"
                         f"{compact(g['T'].max())}): {html.escape(status[:90])}")
        fail_text = "Cells that did not run: " + "; ".join(parts) + "."
    descr = meta.get("implementations", {})
    items = "".join(f"<li><code>{html.escape(i)}</code>: {html.escape(descr.get(i, ''))}</li>"
                    for i in impls)
    sp = rows.dropna(subset=["speedup"])
    text = ""
    if not sp.empty:
        best = sp.groupby("impl")["speedup"].median()
        text = "Median training speed-up over eager: " + ", ".join(
            f"<code>{html.escape(i)}</code> {v:.2f}×" for i, v in best.items()) + "."
        if {"compiled", "native"} <= set(rows.impl):
            comp = rows[rows.impl == "compiled"].set_index(["setup", "p", "T"])["total"]
            nat = rows[rows.impl == "native"].set_index(["setup", "p", "T"])["total"]
            gain = (comp / nat).dropna()
            if not gain.empty:
                where = ", ".join(sorted(gain.index.get_level_values(0).unique()))
                text += (f" Where <code>native</code> compiles ({where}) it is"
                         f" {gain.median():.1f}× faster than <code>compiled</code>"
                         f" (up to {gain.max():.1f}×). The custom op hands every"
                         " scan to ATen, whose scan along the time axis (not the"
                         " innermost one) is a sequential loop per column, and"
                         " nothing around it fuses; Inductor generates a parallel"
                         " scan of its own.")
    intro = (f"<p>Every implementation evaluates the same layer with the same"
             f" weights. Today they are all the PyTorch path, run differently:</p>"
             f"<ul class='prose'>{items}</ul><p>The first chart divides eager time by"
             f" each implementation's time (above the dotted line is faster than"
             f" eager); the second divides the step time by the work, which puts"
             f" every configuration on one scale: a constant line means the"
             f" implementation keeps a fixed throughput however long or deep the"
             f" level. A new kernel added to <code>implementations.py</code> shows"
             f" up here as one more line.</p>")
    return section("Implementations", intro,
                   figure(fig, vis, "Depth"), figure(fig2, vis2, "Depth"),
                   reading(text) if text else "",
                   reading(fail_text) if fail_text else "", anchor="impl")


def split_section(df: pd.DataFrame, impls: list[str]) -> str:
    L = df[(df.experiment == "length") & df.ok & (df.task == "train")]
    if L.empty:
        return ""
    Ts = sorted(L["T"].unique())
    T = Ts[-2] if len(Ts) > 2 else Ts[-1]
    p = 3 if 3 in set(L["p"]) else L["p"].max()
    d = L[(L["T"] == T) & (L["p"] == p)]
    setups = present(d["setup"].unique(), SETUPS)
    views = [i for i in impls if i in set(d.impl)]
    fig = go.Figure()
    vis: dict = {v: [] for v in views}
    for impl in views:
        dd = d[d.impl == impl].set_index("setup").reindex(setups)
        for phase, col, color in (("forward", "fwd", SLOTS[0][0]),
                                  ("backward", "bwd", RAMP[0][0])):
            fig.add_trace(go.Bar(
                x=setups, y=dd[col], name=phase, legendgroup=phase,
                marker=dict(color=color, line=dict(width=2, color="#fcfcfb"),
                            cornerradius=4),
                hovertemplate=f"{phase}<br>%{{x}}<br>%{{y:.3g}} ms<extra></extra>",
            ))
            for k in views:
                vis[k].append(k == impl)
    fig.update_layout(barmode="stack", bargap=0.55)
    fig.update_yaxes(title_text="time per step [ms]")
    style(fig, 340)
    intro = (f"<p>The training step split into its forward and backward pass, at"
             f" T = {T:,} and p = {p:g}. The backward of a scan is a reverse"
             f" scan of the same length (a reverse cumulative sum in the reals and"
             f" the log semiring, a scatter to the running argmax in the max"
             f" semirings), and the contractions and value products have"
             f" backwards of their own.</p>")
    return section("Forward against backward", intro,
                   figure(fig, vis, "Implementation"), anchor="split")


def memory_section(df: pd.DataFrame, meta: dict) -> str:
    L = df[(df.experiment == "length") & df.ok & df.peak_mb.notna()]
    if L.empty:
        return ""
    depths = sorted(L["p"].dropna().unique())
    views = present(L["view"].unique(), VIEW_ORDER) + sorted(
        set(L["view"]) - set(VIEW_ORDER))
    panels = present(L["setup"].unique(), SETUPS)
    fig, vis = panel_lines(
        L, panel="setup", panels=panels, x="T", y="peak_mb", series="p",
        order=depths, colors=p_color(depths), label=lambda p: f"p = {p:g}",
        view="view", views=views, x_title="length T",
        y_title="activation memory [MiB]",
        hover="T = %{x:,}<br>%{y:,.0f} MiB", height=380,
    )
    fig.update_xaxes(**t_ticks(L["T"].unique()))
    caps = memory_caps(meta)
    cap = caps[-1] if caps else None      # the length experiment's run
    if cap:
        fig.add_hline(y=cap * 1024, line=dict(color=NEUTRAL, width=1, dash="dot"),
                      row="all", col="all")
    C = L[(L.impl == "compiled") & (L.task == "train")].copy()
    text = ""
    if not C.empty:
        C["bytes_per_entry"] = C["peak_mb"] * 2**20 / C["work"]
        per = C.groupby("setup")["bytes_per_entry"].median()
        text = ("Bytes of activation memory per state entry and level in the"
                " compiled training step (median over cells): " + ", ".join(
                    f"{s} {v:.0f}" for s, v in per.reindex(
                        present(per.index, SETUPS)).items())
                + ". A float32 state entry is 4 bytes, so the backward keeps"
                " several tensors of the state's size per level.")
    oom = df[(df.experiment == "length") & (df.status == "OOM")]
    if not oom.empty:
        text += (f" {len(oom)} cells did not fit"
                 + (f" in the {cap:g} GiB cap" if cap else "")
                 + "; their lines stop where that happens.")
    intro = ("<p>Peak memory of a step above the weights and the input, against"
             " the length. Training keeps every level's state for the backward"
             " pass, so memory grows with <i>p</i>·<i>T</i>; the forward pass"
             " alone frees each state once the next level has read it."
             + (" The dotted line is the memory cap of the run." if cap else "")
             + "</p>")
    return section("Memory", intro, figure(fig, vis, "Implementation · task"),
                   reading(text) if text else "", anchor="memory")


def rank_section(df: pd.DataFrame) -> str:
    R = df[(df.experiment == "rank") & df.ok].copy()
    if R.empty:
        return ""
    R["family"] = np.where(R["kernels"].str.startswith("cos"),
                           R["semiring"] + " · cosine", R["semiring"])
    families = present(R["family"].unique(), SETUPS)
    R["quantity"] = "time"
    M = R.copy()
    M["quantity"] = "memory"
    M["total"] = M["peak_mb"]
    both = pd.concat([R, M])
    views = present(both["impl"].unique(), ["compiled", "eager"]) + sorted(
        set(both.impl) - {"compiled", "eager"})
    fig, vis = panel_lines(
        both, panel="quantity", panels=["time", "memory"], x="rank", y="total",
        series="family", order=families, colors=SEMIRING_COLOR, view="impl",
        views=views, shared_y=False, x_title="kernel rank R",
        hover="R = %{x}<br>%{y:.3g}", height=380,
        panel_title=lambda q: "time per step [ms]" if q == "time" else "activation memory [MiB]",
    )
    fig.update_xaxes(**log_ticks(R["rank"].unique()))
    T, p = int(R["T"].iloc[0]), int(R["p"].iloc[0])
    C = R[R.impl == "compiled"]
    text = ""
    if not C.empty:
        g = C.groupby("rank")["total"].median()
        if len(g) > 1:
            lo, hi = g.index.min(), g.index.max()
            text = (f"From rank {lo:g} to {hi:g} the compiled step grows"
                    f" {g[hi] / g[lo]:.1f}× (median over semirings) while the work"
                    f" grows {hi / lo:g}×.")
    intro = (f"<p>The kernel rank <i>R</i> is the number of separable terms a kernel"
             f" carries through the scans: <code>d_qk</code> for the exponential"
             f" kernel and (exponent + 1)<sup>d_qk</sup> for the cosine kernel. The"
             f" state, and with it the work, is <i>R</i> times as large. T = {T:,},"
             f" p = {p}, training step.</p>")
    return section("Kernel rank", intro, figure(fig, vis, "Implementation"),
                   reading(text) if text else "", anchor="rank")


def settings_section(df: pd.DataFrame) -> str:
    S = df[df.experiment == "settings"]
    if S.empty:
        return ""
    order = list(dict.fromkeys(S["variant"]))
    views = present(S["impl"].unique(), ["compiled", "eager"]) + sorted(
        set(S.impl) - {"compiled", "eager"})
    fig = go.Figure()
    vis: dict = {v: [] for v in views}
    semis = present(S["semiring"].unique(), ["reals", "log", "arctic", "bayesian"])
    for impl in views:
        for semiring in semis:
            d = S[(S.impl == impl) & (S.semiring == semiring)].set_index("variant")
            base = d.loc["base", "total"] if "base" in d.index else np.nan
            d = d.reindex(order)
            ratio = d["total"] / base
            text = [("out of memory" if st == "OOM" else "failed")
                    if isinstance(st, str) and st != "ok"
                    else ("" if not np.isfinite(r) else f"{r:.2f}×")
                    for st, r in zip(d["status"], ratio)]
            fig.add_trace(go.Bar(
                y=order, x=ratio.fillna(0), orientation="h", name=semiring,
                legendgroup=semiring, text=text, textposition="outside",
                cliponaxis=False, textfont=dict(size=11.5),
                marker=dict(color=SEMIRING_COLOR[semiring], cornerradius=4,
                            line=dict(width=2, color="#fcfcfb")),
                customdata=np.stack([d["total"].fillna(-1), d["peak_mb"].fillna(-1)], -1),
                hovertemplate=(f"{semiring} · %{{y}}<br>%{{x:.2f}}× base<br>"
                               "%{customdata[0]:.3g} ms, %{customdata[1]:,.0f} MiB"
                               "<extra></extra>"),
            ))
            for k in views:
                vis[k].append(k == impl)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="time relative to the base config [×]")
    fig.add_vline(x=1, line=dict(color=NEUTRAL, width=1, dash="dot"))
    style(fig, 120 + 34 * len(order))
    fig.update_layout(margin=dict(l=190, r=70))
    T, p = int(S["T"].iloc[0]), int(S["p"].dropna().iloc[0]) if S["p"].notna().any() else 3
    intro = (f"<p>One option of <code>LISSConfig</code> changed at a time against a"
             f" base layer (decay × exponential kernel, T = {T:,}, p = {p}), in the"
             f" reals and the arctic semiring. A bar of 2× doubles the training step."
             f" Normalisation is defined for the reals and the log semiring only,"
             f" so the arctic rows have none. The hover has the absolute time and"
             f" memory.</p>")
    return section("Layer options", intro, figure(fig, vis, "Implementation"),
                   anchor="settings")


def scan_section(df: pd.DataFrame) -> str:
    S = df[(df.experiment == "scan") & df.ok]
    if S.empty:
        return ""
    names = {"compiled": "rescaled cumsum", "hillis": "Hillis–Steele",
             "triton": "fused Triton scan"}
    colors = {"compiled": SLOTS[0][0], "hillis": SLOTS[2][0],
              "triton": SLOTS[1][0]}
    S = S.copy()
    S["quantity"] = "time"
    Cc = S.copy()
    Cc["quantity"] = "compile"
    Cc["total"] = Cc["compile_s"]
    both = pd.concat([S, Cc])
    both["panel"] = both["semiring"] + " · " + both["quantity"]
    panels = [f"{s} · {q}" for q in ("time", "compile")
              for s in present(S.semiring.unique(), ["reals", "bayesian"])]
    fig, _ = panel_lines(
        both, panel="panel", panels=panels, x="T", y="total", series="impl",
        order=[i for i in ("compiled", "hillis", "triton") if i in set(S.impl)],
        colors=colors, label=lambda i: names.get(i, i), x_title="length T",
        shared_y=False, hover="T = %{x:,}<br>%{y:.3g}", height=380,
        cols=len(panels) // 2 or 1,
        panel_title=lambda p: p.replace("· time", "· step [ms]").replace(
            "· compile", "· compile [s]"),
    )
    fig.update_xaxes(**t_ticks(S["T"].unique()))
    piv = S.pivot_table(index=["semiring", "T"], columns="impl", values="total")
    text = ""
    if {"compiled", "hillis"} <= set(piv.columns):
        r = (piv["hillis"] / piv["compiled"]).dropna()
        per_T = r.groupby(level="T").median()
        cmp = S.pivot_table(index=["semiring", "T"], columns="impl",
                            values="compile_s")
        mem = S.pivot_table(index=["semiring", "T"], columns="impl",
                            values="peak_mb")
        oom = df[(df.experiment == "scan") & (df.status == "OOM")]
        text = ("Hillis–Steele's step time over the rescaled cumsum's: " + ", ".join(
                    f"{v:.1f}× at T = {compact(T)}" for T, v in per_T.items())
                + (". Below one it is the faster scan per step, from"
                   f" T = {compact(per_T[per_T < 1].index.min())} on: its rounds"
                   " are elementwise operations Inductor fuses, while the cumsum is"
                   " ATen's sequential time-axis scan behind the custom op"
                   if (per_T < 1).any() else
                   ". It is the slower scan per step at every length measured")
                + ". It pays elsewhere: compiling takes"
                f" up to {cmp['hillis'].max() / 60:.0f} min against"
                f" {cmp['compiled'].max():.0f} s, it needs"
                f" {(mem['hillis'] / mem['compiled']).dropna().median():.1f}× the"
                " memory"
                + (f", and it ran out of memory at T = {compact(oom['T'].min())}"
                   if not oom.empty else "") + ".")
    if {"compiled", "triton"} <= set(piv.columns):
        r = (piv["compiled"] / piv["triton"]).dropna()
        per_T = r[r.index.get_level_values("semiring") == "reals"].groupby(
            level="T").median()
        if not per_T.empty:
            text += (" The fused Triton scan, which applies the decay step by"
                     " step and has neither bound nor fallback, is " + ", ".join(
                         f"{v:.1f}× faster at T = {compact(T)}"
                         for T, v in per_T.items()) + " than the rescaled cumsum.")
    intro = ("<p>On the PyTorch path a decayed scan in the reals or the bayesian"
             " semiring has two forms. While the decay stays small enough"
             " (rate·(T−1) ≤ 40) the library rescales,"
             " cumsum(u<sub>s</sub>e<sup>βs</sup>)·e<sup>−βt</sup>, one ordinary"
             " scan. Past that bound e<sup>βt</sup> would overflow, and it switches"
             " to a Hillis–Steele scan: log<sub>2</sub>T rounds of shifted"
             " additions, exact but <i>O</i>(T log T). The bound is met by a"
             " strong decay or by a length well past the context length, so a"
             " model trained at one length can take the slow path at another."
             " <code>scan: triton</code> (reals only) multiplies by the decay at"
             " every step instead, in O(T) at any rate. All are timed here under"
             " compile, at p = 3.</p>")
    return section("The decayed scans", intro, figure(fig), reading(text)
                   if text else "", anchor="scan")


def compile_section(df: pd.DataFrame, impls: list[str]) -> str:
    L = df[(df.experiment == "length") & df.ok & df.compile_s.notna()
           & (df.task == "train")]
    if L.empty:
        return ""
    g = L.groupby(["impl", "setup", "p"], as_index=False)["compile_s"].median()
    views = [i for i in impls if i in set(g.impl)]
    g["one"] = "compile"
    fig, vis = panel_lines(
        g, panel="one", panels=["compile"], x="p", y="compile_s",
        series="setup", order=present(g.setup.unique(), SETUPS),
        colors=SEMIRING_COLOR, view="impl", views=views, log_x=False,
        log_y=False, x_title="depth p", y_title="compile time [s]",
        hover="p = %{x}<br>%{y:.1f} s", height=320, panel_title=lambda _: "",
    )
    fig.update_xaxes(dtick=1)
    intro = ("<p>Wall time of the first training step minus a timed step: tracing,"
             " Inductor's code generation for the forward and the backward graph,"
             " and the first run, median over the lengths. The graph has one"
             " block of operations per level, so the compile time grows with"
             " <i>p</i>; it is paid once per shape.</p>")
    return section("Compile time", intro, figure(fig, vis, "Implementation"),
                   anchor="compile")


def accuracy_section(df: pd.DataFrame, impls: list[str]) -> str:
    L = df[(df.experiment == "length") & df.ok]
    rows = []
    for q, col in (("output", "err_out"), ("input gradient", "err_grad")):
        d = L[L[col].notna()].copy()
        if q == "input gradient":
            d = d[d.task == "train"]
        else:
            d = d[d.task == "train"]
        d["err"] = d[col]
        d["vq"] = q
        rows.append(d)
    E = pd.concat(rows)
    if E.empty:
        return ""
    E["v"] = E["vq"] + " · p = " + E["p"].map(lambda p: f"{p:g}")
    depths = sorted(E["p"].unique())
    views = [f"{q} · p = {p:g}" for q in ("output", "input gradient")
             for p in depths if not E[(E.vq == q) & (E.p == p)].empty]
    panels = present(E["setup"].unique(), SETUPS)
    colors = impl_color(impls)
    fig, vis = panel_lines(
        E, panel="setup", panels=panels, x="T", y="err", series="impl",
        order=[i for i in impls if i in set(E.impl)], colors=colors, view="v",
        views=views, x_title="length T",
        y_title="relative error against float64",
        hover="T = %{x:,}<br>%{y:.2e}", height=380,
    )
    fig.update_xaxes(**t_ticks(E["T"].unique()))
    g = E[(E.impl == "compiled") & (E.vq == "output")]
    text = ""
    if not g.empty:
        lo = g[g["T"] == g["T"].min()]["err"].median()
        hi = g[g["T"] == g["T"].max()]["err"].median()
        text = (f"The float32 output error grows from {sci(lo)} at"
                f" T = {compact(g['T'].min())} to {sci(hi)} at"
                f" T = {compact(g['T'].max())} (compiled, median over semirings"
                f" and depths): PyTorch's CUDA scans along a non-innermost"
                f" dimension accumulate one position after another, so rounding"
                f" errors add up along the sequence. The stability report has the"
                f" mechanism.")
    G = E[(E.vq == "input gradient") & E.setup.isin(["arctic", "bayesian"])]
    if not G.empty and G["err"].max() > 1e-4:
        text += (" The large gradient errors in the max semirings are not"
                 " drift: where two tuples are nearly tied, float32 and float64"
                 " pick different maximisers, and the gradient goes to a"
                 " different position.")
    intro = ("<p>Relative error of the output and of the input gradient against a"
             " float64 eager evaluation of the same weights and input"
             " (‖y − y<sub>64</sub>‖ / ‖y<sub>64</sub>‖). This is the check that a"
             " faster implementation computes the same layer; cells where the"
             " float64 reference did not fit in memory are missing.</p>")
    return section("Accuracy of each implementation", intro,
                   figure(fig, vis, "Quantity · depth"),
                   reading(text) if text else "", anchor="accuracy")


def failures(df: pd.DataFrame) -> str:
    F = df[~df.ok]
    if F.empty:
        return ""
    rows = []
    for (exp, setup, variant, impl, task, status), g in F.groupby(
        ["experiment", "setup", "variant", "impl", "task", "status"], dropna=False
    ):
        ps = sorted(g["p"].dropna().unique())
        rows.append([
            html.escape(exp), html.escape(str(setup)),
            html.escape(variant if variant != "base" else ""),
            html.escape(impl), html.escape(task),
            ", ".join(f"{p:g}" for p in ps) or "–",
            (compact(g['T'].min()) if g['T'].min() == g['T'].max() else
             f"{compact(g['T'].min())}–{compact(g['T'].max())}"),
            f"<span class='fail'>{html.escape(str(status)[:110])}</span>",
        ])
    return section(
        "Cells that did not run",
        "<p>Out of memory means the step did not fit the run's memory cap; a"
        " failure is an exception, usually from the compiler.</p>",
        table(["experiment", "semiring", "option", "impl", "task", "p", "T",
               "status"], rows),
        anchor="failures",
    )


GLOSSARY = [
    ("The layer", [
        ("p", "the depth of a level: the number of indices of the iterated sum."),
        ("T", "the length of the sequence; the context length is set to T."),
        ("R", "the kernel rank: separable terms a kernel carries through the scans."),
        ("N, d_v", "heads (<code>n_is</code>) and the width of the value vectors."),
        ("work", "state entries times scans, p·B·T·N·R·d<sub>v</sub> per level."),
    ]),
    ("Semirings", [
        ("reals", "⊕ = sum, ⊗ = product: decayed, gated linear attention chains."),
        ("log", "⊕ = logsumexp, ⊗ = sum: the reals of positive numbers, stored as logarithms."),
        ("arctic", "⊕ = max, ⊗ = sum: the best tuple, max-plus."),
        ("bayesian", "⊕ = max, ⊗ = product: the best tuple of products (Viterbi)."),
    ]),
    ("Kernels", [
        ("decay", "λ<sup>t′−t</sup>, a per-head, per-pair rate folded into each scan."),
        ("exponential", "exp(q(x<sub>t′</sub>) − k(x<sub>t</sub>)), rank d_qk."),
        ("cosine", "∏ cos(q − k)<sup>m</sup>, reals only, rank (m+1)<sup>d_qk</sup>."),
    ]),
    ("Measurements", [
        ("train", "forward and backward of sum(y · r), gradients for the weights and the input."),
        ("infer", "the forward pass under no_grad."),
        ("compile", "first step minus a timed step, for compiled implementations."),
        ("memory", "peak allocation during a step above the weights and the input."),
        ("error", "‖y − y<sub>64</sub>‖ / ‖y<sub>64</sub>‖ against a float64 eager evaluation."),
    ]),
]


def build(df: pd.DataFrame, meta: dict, source: str, stat: str,
          fragment: bool) -> str:
    impls = implementations(df, meta)
    session = meta.get("sessions", [{}])[-1]
    when = session.get("date", "")[:10]
    head = " · ".join(x for x in [
        html.escape(session.get("device", "")),
        f"torch {html.escape(session.get('torch', ''))}",
        f"triton {html.escape(session['triton'])}" if session.get("triton") else "",
        f"elissabeth {html.escape(session.get('elissabeth_commit', ''))}"
        + (" (modified)" if session.get("elissabeth_dirty") else ""),
        when, f"{int(df.ok.sum())} cells",
    ] if x)
    body = "".join([
        lead(meta, df, stat), kpis(df),
        glossary(GLOSSARY),
        length_section(df), depth_section(df),
        implementation_section(df, impls, meta), split_section(df, impls),
        memory_section(df, meta), rank_section(df), settings_section(df),
        scan_section(df), compile_section(df, impls),
        accuracy_section(df, impls), failures(df),
        f"<footer>Generated by <code>plot_benchmark.py</code> from"
        f" <code>{html.escape(source)}</code> ({'best' if stat == 'min' else 'median'}"
        f" step times). The GPU was shared with other jobs while this ran, so"
        f" absolute times carry their noise; ratios within a chart are more"
        f" reliable. Interactive: hover for values, click legend entries to hide"
        f" series, drag to zoom.</footer>",
    ])
    return page("LISS Layer Benchmark", "LISS layer benchmark", head, body,
                fragment=fragment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--fragment", type=Path, default=None,
                        help="also write the page without its html/head/body"
                             " skeleton (for publishing as an artifact)")
    parser.add_argument("--stat", choices=["min", "median"], default="min")
    args = parser.parse_args()
    df = load(args.csv, args.stat)
    meta_path = args.csv.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    out = args.out or args.csv.with_suffix(".html")
    out.write_text(build(df, meta, args.csv.name, args.stat, False))
    print(f"Wrote {out}")
    if args.fragment:
        # an artifact viewer cannot save files, so no "download as png"
        PLOT_CONFIG["modeBarButtonsToRemove"].append("toImage")
        args.fragment.write_text(build(df, meta, args.csv.name, args.stat, True))
        print(f"Wrote {args.fragment}")


if __name__ == "__main__":
    main()
