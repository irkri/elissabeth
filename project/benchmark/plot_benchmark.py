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
                    compact, eq, figure, glossary, log_ticks, m, md, page, ramp,
                    reading, sci, section, style, table, tiles)

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
    """A labelled tick at every length measured, upright labels when there
    are many (five panels side by side leave each one too narrow for them
    horizontally, and plotly's own tilt differs from panel to panel)."""
    vals = [v for v in values if v == v]
    return dict(**log_ticks(vals), **({"tickangle": -90} if len(set(vals)) >= 6 else {}))


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
            own = sub[(sub[panel] == pn) & sub[series].isin(order)]
            if ref is not None and not ref.empty and not own.empty:
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


def shared_gpu(meta: dict) -> bool:
    """Whether the GPU ran other jobs: the local runs cap their memory for
    the others' sake, the cluster job owns its GPU and sets no cap."""
    return bool(memory_caps(meta))


def device_memory(meta: dict) -> str:
    gb = meta.get("sessions", [{}])[-1].get("device_memory_gb")
    return f"{gb:.0f} GiB" if gb else "the GPU's memory"


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
    paths = "" if "triton" not in set(df.impl) else (
        "<p>The library runs these scans on one of two paths. On the PyTorch"
        " path (<code>scan: torch</code>, eager or compiled) every scan is a"
        " PyTorch scan over a stored state. On the Triton path"
        " (<code>scan: triton</code>; reals, log and arctic) one fused kernel per"
        " scan computes the key factor, the decayed scan and the query"
        " contraction, chunked over time, and keeps the " + m("R") + "-wide state"
        " in registers: a level stores " + m("(B, T, N, d_v)") + " tensors instead"
        " of " + m("(B, T, N, R, d_v)") + " ones.</p>")
    load = (", so they shared whatever load other jobs put on the GPU."
            if shared_gpu(meta) else ". The GPU ran nothing else.")
    scans = md(
        r"\begin{aligned}"
        r"S_1(t) &= \bigoplus_{s \le t} \lambda_1^{\,t-s} \otimes \psi_1(s) \otimes v_1(s) \\"
        r"S_l(t) &= \bigoplus_{s \le t} \lambda_l^{\,t-s} \otimes \psi_l(s) \otimes"
        r" \bigl\langle \varphi_{l-1}(s),\, S_{l-1}(s-1) \bigr\rangle \otimes v_l(s) \\"
        r"\mathrm{ISS}^{(p)}(t) &= \bigl\langle \varphi_p(t),\, S_p(t) \bigr\rangle"
        r" = \bigoplus_r \varphi_{p,r}(t) \otimes S_{p,r}(t)"
        r"\end{aligned}")
    return f"""<div class="lead">
<p>A LISS level of depth {m("p")} sums, for every head and position {m("t")},
over all index tuples {m(r"t_1 < \dots < t_p \le t")}
the semiring product of the values at those indices and of the kernels between
neighbouring indices. Because every kernel factorises into a query feature
{m(r"\varphi")} at the later index and a key feature {m(r"\psi")} at the earlier one,
times a decay {m(r"\lambda")}, the sum is not evaluated over the {m("O(T^p)")}
tuples but as {m("p")} scans:</p>
{scans}
<p>Each scan carries a state of shape {m("(B, T, N, R, d_v)")}: batch, time,
heads, kernel rank and value width. The work of a level is therefore
{m(r"p \cdot B \cdot T \cdot N \cdot R \cdot d_v")} state entries, linear in
both the length and the depth, and the backward pass keeps every state for the
gradient. The four semirings differ only in {m(r"\oplus")} and {m(r"\otimes")}: a sum and
a product (<code>reals</code>), logsumexp and a sum (<code>log</code>), a maximum and a
sum (<code>arctic</code>), a maximum and a product (<code>bayesian</code>).</p>{paths}
<p>Each cell builds one causal LISS layer ({m(f"B = {B}")}, {m(f"N = {N}")} heads,
{m(f"d_v = {dv}")}, model width {d}, context length {m("T")}) and times a
training step, the forward and backward pass of the loss
{m(r"\mathcal{L} = \sum_{b,t,i} y_{bti}\, r_{bti}")} for a fixed random {m("r")}
(<b>train</b>), and the forward pass alone under
<code>no_grad</code> (<b>infer</b>). Times are {which}; the other is in each
point's hover. {together} were timed in turns, one step each{load}
Memory is the peak a step allocates on top of the weights and the
input. Every cell's output and input gradient are compared with a float64
evaluation of the same weights.{cap_text}</p></div>"""


def kpis(df: pd.DataFrame) -> str:
    items = []
    L = df[(df.experiment == "length") & df.ok & (df.task == "train")]
    piv = L.pivot_table(index=["setup", "p", "T"], columns="impl",
                        values="total", aggfunc="min")
    if {"compiled", "triton"} <= set(piv.columns):
        long = piv[piv.index.get_level_values("T") >= 16384]
        sp = (long["compiled"] / long["triton"]).dropna()
        if not sp.empty:
            items.append(("Triton over compiled", f"{sp.median():.1f}×",
                          f"median training speed-up from {eq('T', '16k')}, {sp.min():.1f}–"
                          f"{sp.max():.1f}× over {len(sp)} cells"))
        mem = L.pivot_table(index=["setup", "p", "T"], columns="impl",
                            values="peak_mb", aggfunc="min")
        m = (mem["triton"] / mem["compiled"]).dropna().groupby(level="setup").median()
        if not m.empty:
            lo, hi = m.idxmin(), m.idxmax()
            items.append(("Triton's memory", f"{m[lo]:.2f}×",
                          f"of compiled's in training, {lo}; {m[hi]:.2f}× in {hi}"))
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
                              f"compiled training step at {eq('T', compact(Tref))}"
                              f" (work ratio {hi / lo:g}×)"))
    A = df[(df.experiment == "attention") & df.ok & (df.task == "train")
           & (df.impl == "compiled")]
    fastest = "triton" if "triton" in set(L.impl) else "compiled"
    R = L[(L.impl == fastest) & (L.setup == "reals") & (L["p"] == 2)]
    if not A.empty and not R.empty:
        common = sorted(set(A["T"]) & set(R["T"]))
        if common:
            T = common[-1]
            ratio = (A[A["T"] == T]["total"].min()
                     / R[R["T"] == T]["total"].min())
            items.append(("Attention against LISS p = 2",
                          f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×",
                          f"softmax attention's training step over a reals"
                          f" level's ({fastest}), {eq('T', compact(T))}"))
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
    if not A.empty:     # attention has no triton: compare with its compiled step
        A = pd.concat([A] + [
            A[(A.impl == "compiled") & (A.task == v.split(" · ")[1])].assign(view=v)
            for v in views if v not in set(A.view)])
    fig, vis = panel_lines(
        L, panel="setup", panels=panels, x="T", y="total", series="p",
        order=depths, colors=colors, label=lambda p: f"p = {p:g}",
        view="view", views=views, refs=A, ref_name="softmax attention",
        x_title="length T", y_title="time per step [ms]",
        hover="T = %{x:,}<br>%{y:.3g} ms", height=400,
    )
    fig.update_xaxes(**t_ticks(L["T"].unique()))
    # the reading: exponent of T over the last two doublings, per implementation
    def slopes(impl: str) -> list[float]:
        out = []
        C = L[(L.impl == impl) & (L.task == "train")]
        for _, g in C.groupby(["setup", "p"]):
            g = g.sort_values("T")
            if len(g) >= 3:
                a, b = g.iloc[-3], g.iloc[-1]
                out.append(math.log(b["total"] / a["total"]) / math.log(b["T"] / a["T"]))
        return out
    text = ""
    for impl in [i for i in ("compiled", "triton") if i in set(L.impl)]:
        sl = slopes(impl)
        if sl:
            text += (f"{' The ' if text else 'Over the longest lengths the '}"
                     f"<code>{impl}</code> training step grows like"
                     f" {m('T^{' + format(np.median(sl), '.2f') + '}')} (median over"
                     f" semirings and depths, range {min(sl):.2f}–{max(sl):.2f})"
                     + ("" if text else ": linear, as the recursion promises") + ".")
    if text:
        text += (" Below a few thousand positions the step is dominated by launch"
                 " overhead and barely depends on T.")
    att = ""
    if not A.empty:
        att = (" The dotted line in every panel is softmax attention (PyTorch's"
               " scaled-dot-product kernel, causal, same width and heads) in the"
               " same implementation and task, compiled in the Triton views, for"
               " scale.")
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
    text = ""
    for impl in [i for i in ("compiled", "triton") if i in set(L.impl)]:
        C = L[(L.impl == impl) & (L.task == "train") & (L["T"] == pick[-1])]
        per = [(setup, np.polyfit(g["p"], g["total"], 1)[0])
               for setup, g in C.groupby("setup") if g["p"].nunique() >= 2]
        per = sorted(per, key=lambda sa: SETUPS.index(sa[0]) if sa[0] in SETUPS else 99)
        if per:
            parts = ", ".join(f"{s} {a:.1f} ms" for s, a in per)
            text += (f"At {eq('T', f'{pick[-1]:,}')} every extra level adds a near-constant"
                     f" time to the <code>{impl}</code> training step: {parts} per"
                     " level (least-squares slope over p). " if not text else
                     f"With <code>{impl}</code>: {parts}.")
    intro = ("<p>The same cells against the depth " + m("p") + ", at a short, a"
             " middle and a long length. The work grows linearly in " + m("p") + ", and so"
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
        fig.update_yaxes(**log_ticks([0.25, 0.5, 1, 2, 4, 8, 16, 32]))
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
        if {"compiled", "triton"} <= set(rows.impl):
            comp = rows[rows.impl == "compiled"].set_index(["setup", "p", "T"])["total"]
            tri = rows[rows.impl == "triton"].set_index(["setup", "p", "T"])["total"]
            gain = (comp / tri).dropna()
            per_T = gain.groupby(level="T").median()
            short = rows[(rows.impl == "triton") & (rows["T"] <= 1024)]
            floor = short.groupby("p")["total"].median()
            step = np.polyfit(floor.index, floor.values, 1) if len(floor) > 1 else None
            text += (" <code>triton</code> against <code>compiled</code>, median"
                     " over semirings and depths: " + ", ".join(
                         f"{v:.1f}× at {eq('T', compact(T))}" for T, v in per_T.items())
                     + f" (up to {gain.max():.0f}×, in the log semiring, whose PyTorch"
                     " backward runs two reverse logcumsumexps per scan).")
            if step is not None:
                text += (" Up to 1k positions the Triton step does not depend on T:"
                         f" about {step[1]:.1f} ms plus {step[0]:.1f} ms per level,"
                         " the cost of launching its kernels, which is why it is"
                         " slower than <code>compiled</code> at the shortest"
                         " lengths.")
    intro = (f"<p>Every implementation evaluates the same layer with the same"
             f" weights:</p>"
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
    share = (d["bwd"] / (d["fwd"] + d["bwd"])).groupby(d["impl"]).agg(["min", "max"])
    text = ("The backward's share of the training step: " + ", ".join(
        f"<code>{i}</code> {share.loc[i, 'min']:.0%}–{share.loc[i, 'max']:.0%}"
        for i in views if i in share.index) + ".") if not share.empty else ""
    if {"compiled", "triton"} <= set(d.impl):
        b = d.pivot_table(index="setup", columns="impl", values="bwd")
        f = d.pivot_table(index="setup", columns="impl", values="fwd")
        rb, rf = (b["compiled"] / b["triton"]).dropna(), (f["compiled"] / f["triton"]).dropna()
        if not rb.empty:
            text += (f" <code>triton</code> takes the forward {rf.min():.0f}–{rf.max():.0f}×"
                     f" and the backward {rb.min():.0f}–{rb.max():.0f}× faster than"
                     f" <code>compiled</code> (the backward most in {rb.idxmax()}).")
    intro = (f"<p>The training step split into its forward and backward pass, at"
             f" {eq('T', f'{T:,}')} and {eq('p', f'{p:g}')}. The backward of a scan is a reverse"
             f" scan of the same length (a reverse cumulative sum in the reals and"
             f" the log semiring, a scatter to the running argmax in the max"
             f" semirings), and the contractions and value products have"
             f" backwards of their own.</p>")
    return section("Forward against backward", intro,
                   figure(fig, vis, "Implementation"),
                   reading(text) if text else "", anchor="split")


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
    text = ""
    for impl in [i for i in ("compiled", "triton") if i in set(L.impl)]:
        C = L[(L.impl == impl) & (L.task == "train")].copy()
        C["bytes_per_entry"] = C["peak_mb"] * 2**20 / C["work"]
        per = C.groupby("setup")["bytes_per_entry"].median()
        parts = ", ".join(f"{s} {v:.0f}" for s, v in per.reindex(
            present(per.index, SETUPS)).items())
        text += ("Bytes of activation memory per state entry and level in the"
                 f" <code>{impl}</code> training step (median over cells): {parts}."
                 " A float32 state entry is 4 bytes, so the backward keeps several"
                 " tensors of the state's size per level." if not text else
                 f" With <code>{impl}</code>: {parts}; it stores no {m('R')}-wide"
                 " state, so the gap grows with the rank.")
    oom = df[(df.experiment == "length") & (df.status == "OOM")]
    if not oom.empty:
        text += (f" {len(oom)} cells did not fit"
                 + (f" in the {cap:g} GiB cap" if cap else f" in {device_memory(meta)}")
                 + "; their lines stop where that happens.")
    intro = ("<p>Peak memory of a step above the weights and the input, against"
             " the length. Training keeps every level's state for the backward"
             " pass, so memory grows with " + m(r"p \cdot T") + "; the forward pass"
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
    text = ""
    for impl in [i for i in ("compiled", "triton") if i in set(R.impl)]:
        C = R[(R.impl == impl) & (R.family != "reals · cosine")]
        g = C.groupby("rank")[["total", "peak_mb"]].median()
        if len(g) > 1:
            lo, hi = g.index.min(), g.index.max()
            text += (f"{' ' if text else ''}From rank {lo:g} to {hi:g} (exponential"
                     f" kernel) the <code>{impl}</code> step grows"
                     f" {g.total[hi] / g.total[lo]:.1f}× and its memory"
                     f" {g.peak_mb[hi] / g.peak_mb[lo]:.1f}× (median over semirings)"
                     + (f", while the work grows {hi / lo:g}×." if not text else "."))
    intro = (f"<p>The kernel rank {m('R')} is the number of separable terms a kernel"
             f" carries through the scans: {m('R = d_{qk}')} for the exponential"
             f" kernel and {m('R = (m+1)^{d_{qk}}')} for the cosine kernel of exponent"
             f" {m('m')}. The state, and with it the work, is {m('R')} times as large."
             f" {eq('T', f'{T:,}')}, {eq('p', p)}, training step.</p>")
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
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08,
                      uniformtext=dict(minsize=11, mode="show"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="time relative to the base config [×]")
    fig.add_vline(x=1, line=dict(color=NEUTRAL, width=1, dash="dot"))
    style(fig, 120 + 34 * len(order))
    fig.update_layout(margin=dict(l=190, r=70))
    T, p = int(S["T"].iloc[0]), int(S["p"].dropna().iloc[0]) if S["p"].notna().any() else 3
    text = ""
    ok = S[S.ok]
    if {"compiled", "triton"} <= set(ok.impl):
        piv = ok.pivot_table(index="variant", columns=["impl", "semiring"], values="total")
        rel = piv / piv.loc["base"]
        own = (rel["triton"] / rel["compiled"]).min(axis=1)
        wide = own[own > 1.5].sort_values(ascending=False).index.tolist()
        ahead = (piv["compiled"] / piv["triton"])
        if wide:
            text = ("Each implementation is measured against its own base. Options"
                    " that widen the state cost <code>triton</code> more than"
                    " <code>compiled</code>: " + ", ".join(
                        f"{v} {rel['triton'].loc[v].min():.1f}–{rel['triton'].loc[v].max():.1f}×"
                        f" its base against {rel['compiled'].loc[v].min():.1f}–"
                        f"{rel['compiled'].loc[v].max():.1f}×" for v in wide)
                    + ". The PyTorch scans are bound by their sequential loop, so"
                    " more columns run alongside almost for free, while the fused"
                    " kernel is bound by its arithmetic. In absolute time"
                    " <code>triton</code> stays ahead: " + ", ".join(
                        f"{v} {ahead.loc[v].min():.1f}–{ahead.loc[v].max():.1f}×"
                        for v in wide)
                    + f" faster than <code>compiled</code>, against"
                    f" {ahead.loc['base'].min():.0f}–{ahead.loc['base'].max():.0f}×"
                    " for the base.")
    intro = (f"<p>One option of <code>LISSConfig</code> changed at a time against a"
             f" base layer (decay × exponential kernel, {eq('T', f'{T:,}')}, {eq('p', p)}), in the"
             f" reals and the arctic semiring. A bar of 2× doubles the training step."
             f" Normalisation is defined for the reals and the log semiring only,"
             f" so the arctic rows have none. The hover has the absolute time and"
             f" memory.</p>")
    return section("Layer options", intro, figure(fig, vis, "Implementation"),
                   reading(text) if text else "", anchor="settings")


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
                    f"{v:.1f}× at {eq('T', compact(T))}" for T, v in per_T.items())
                + (". Below one it is the faster scan per step, from"
                   f" {eq('T', compact(per_T[per_T < 1].index.min()))} on: its rounds"
                   " are elementwise operations Inductor fuses, while the cumsum is"
                   " ATen's sequential time-axis scan behind the custom op"
                   if (per_T < 1).any() else
                   ". It is the slower scan per step at every length measured")
                + ". It pays elsewhere: compiling takes"
                f" up to {cmp['hillis'].max() / 60:.0f} min against"
                f" {cmp['compiled'].max():.0f} s, it needs"
                f" {(mem['hillis'] / mem['compiled']).dropna().median():.1f}× the"
                " memory"
                + (f", and it ran out of memory at {eq('T', compact(oom['T'].min()))}"
                   if not oom.empty else "") + ".")
    if {"compiled", "triton"} <= set(piv.columns):
        reals = piv[piv.index.get_level_values("semiring") == "reals"].droplevel(0)
        r = (reals["triton"] / reals["compiled"]).dropna()
        if not r.empty:
            cmp_t = S[S.impl == "triton"]["compile_s"].max()
            text += (" The fused Triton scan, which applies the decay step by"
                     " step and has neither bound nor fallback, over the rescaled"
                     " cumsum's time: " + ", ".join(
                         f"{v:.1f}× at {eq('T', compact(T))}" for T, v in r.items()) + ".")
            if "hillis" in reals and np.isfinite(reals["hillis"].iloc[-1]):
                T = reals.index[-1]
                text += (f" At {eq('T', compact(T))} it takes"
                         f" {reals['triton'].iloc[-1] / reals['hillis'].iloc[-1]:.1f}×"
                         f" Hillis–Steele's time after a {cmp_t:.0f} s compile.")
    intro = ("<p>On the PyTorch path a decayed scan in the reals or the bayesian"
             " semiring has two forms. While the decay rate " + m(r"\beta")
             + " stays small enough, " + m(r"\beta\,(T-1) \le 40") + ", the library"
             " rescales, " + m(r"S_t = e^{-\beta t} \sum_{s \le t} e^{\beta s} u_s")
             + ", one ordinary scan. Past that bound " + m(r"e^{\beta t}") + " would"
             " overflow, and it switches to a Hillis–Steele scan: " + m(r"\log_2 T")
             + " rounds of shifted additions, exact but " + m(r"O(T \log T)") + "."
             " The bound is met by a strong decay or by a length well past the"
             " context length, so a model trained at one length can take the slow"
             " path at another. <code>scan: triton</code> multiplies by the decay"
             " at every step instead, in " + m("O(T)") + " at any rate; of these two"
             " semirings it covers the reals. All are timed here under compile, at "
             + m("p = 3") + ".</p>")
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
    per = g.groupby("impl")["compile_s"].agg(["min", "max"])
    text = ("Median compile time over the lengths, from the shallowest to the"
            " deepest configuration: " + ", ".join(
                f"<code>{i}</code> {per.loc[i, 'min']:.0f}–{per.loc[i, 'max']:.0f} s"
                for i in views if i in per.index) + ".")
    intro = ("<p>Wall time of the first training step minus a timed step: tracing,"
             " Inductor's code generation for the forward and the backward graph,"
             " and the first run, median over the lengths. The graph has one"
             " block of operations per level, so the compile time grows with"
             " " + m("p") + "; it is paid once per shape.</p>")
    return section("Compile time", intro, figure(fig, vis, "Implementation"),
                   reading(text), anchor="compile")


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
                f" {eq('T', compact(g['T'].min()))} to {sci(hi)} at"
                f" {eq('T', compact(g['T'].max()))} (compiled, median over semirings"
                f" and depths): PyTorch's CUDA scans along a non-innermost"
                f" dimension accumulate one position after another, so rounding"
                f" errors add up along the sequence. The stability report has the"
                f" mechanism.")
    t = E[(E.impl == "triton") & (E.vq == "output")]
    if not t.empty and text:
        text += (f" <code>triton</code> stays at {sci(t[t['T'] == t['T'].max()]['err'].median())}"
                 f" at {eq('T', compact(t['T'].max()))}, and its largest output error over"
                 f" all cells is {sci(t['err'].max())} against"
                 f" {sci(g['err'].max())}: a chunked scan rounds through one chunk"
                 " and the chain of chunk states, not through every position.")
    G = E[(E.vq == "input gradient") & E.setup.isin(["arctic", "bayesian"])]
    if not G.empty and G["err"].max() > 1e-4:
        text += (" The large gradient errors in the max semirings are not"
                 " drift: where two tuples are nearly tied, float32 and float64"
                 " pick different maximisers, and the gradient goes to a"
                 " different position.")
    intro = ("<p>Relative error of the output and of the input gradient against a"
             " float64 eager evaluation of the same weights and input"
             " (" + m(r"\lVert y - y_{64} \rVert / \lVert y_{64} \rVert") + "). This is the check that a"
             " faster implementation computes the same layer; cells where the"
             " float64 reference did not fit in memory are missing.</p>")
    return section("Accuracy of each implementation", intro,
                   figure(fig, vis, "Quantity · depth"),
                   reading(text) if text else "", anchor="accuracy")


def failures(df: pd.DataFrame, meta: dict) -> str:
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
        "<p>Out of memory means the step did not fit "
        + ("the run's memory cap" if memory_caps(meta) else device_memory(meta))
        + "; a failure is an exception, usually from the compiler.</p>",
        table(["experiment", "semiring", "option", "impl", "task", "p", "T",
               "status"], rows),
        anchor="failures",
    )


GLOSSARY = [
    ("The layer", [
        (m("p"), "the depth of a level: the number of indices of the iterated sum."),
        (m("T"), "the length of the sequence; the context length is set to " + m("T") + "."),
        (m("R"), "the kernel rank: separable terms a kernel carries through the scans."),
        (m("N,\\ d_v"), "heads (<code>n_is</code>) and the width of the value vectors."),
        ("work", "state entries times scans, " + m(r"p \cdot B \cdot T \cdot N \cdot R \cdot d_v")
         + " per level."),
    ]),
    ("Semirings", [
        ("reals", m(r"\oplus = +,\ \otimes = \times") + ": decayed, gated linear attention chains."),
        ("log", m(r"a \oplus b = \log(e^a + e^b),\ \otimes = +") + ": the reals of positive"
         " numbers, stored as logarithms."),
        ("arctic", m(r"\oplus = \max,\ \otimes = +") + ": the best tuple, max-plus."),
        ("bayesian", m(r"\oplus = \max,\ \otimes = \times") + ": the best tuple of products"
         " (Viterbi)."),
    ]),
    ("Kernels", [
        ("decay", m(r"\lambda^{t'-t}") + ", a per-head, per-pair rate folded into each scan."),
        ("exponential", m(r"\exp\bigl(q(x_{t'}) - k(x_t)\bigr)") + ", rank " + m("d_{qk}") + "."),
        ("cosine", m(r"\prod_j \cos^m (q_j - k_j)") + ", reals only, rank "
         + m("(m+1)^{d_{qk}}") + "."),
    ]),
    ("Measurements", [
        ("train", "forward and backward of " + m(r"\mathcal{L} = \sum y \cdot r")
         + ", gradients for the weights and the input."),
        ("infer", "the forward pass under <code>no_grad</code>."),
        ("compile", "first step minus a timed step, for compiled implementations."),
        ("memory", "peak allocation during a step above the weights and the input."),
        ("error", m(r"\lVert y - y_{64} \rVert / \lVert y_{64} \rVert")
         + " against a float64 eager evaluation."),
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
        accuracy_section(df, impls), failures(df, meta),
        f"<footer>Generated by <code>plot_benchmark.py</code> from"
        f" <code>{html.escape(source)}</code> ({'best' if stat == 'min' else 'median'}"
        f" step times)."
        + (" The GPU was shared with other jobs while this ran, so absolute times"
           " carry their noise; ratios within a chart are more reliable."
           if shared_gpu(meta) else "")
        + " Interactive: hover for values, click legend entries to hide"
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
