"""Triton kernels for the scans of a LISS level, fused with their factors.

One pair ``l`` of a level reads its input ``u_s`` (``(B, T, N, D)`` with
``D = d_v * w``), the key and query features ``k_s, q_t`` (``(B, T, N, R)``)
and a decay rate per head, and returns the contraction

    y_t = <q_t, S_{t-1}>   (inner pairs, ``inclusive=False``)
    y_t = <q_t, S_t>       (the last pair, ``inclusive=True``)
    S_t = lambda (x) S_{t-1} (+) k_t (x) u_t       (an R x D state)

in the reals (``+``, ``*``), the log semiring (logsumexp, ``+``) or the
arctic semiring (``max``, ``+``). The ``R x D`` state lives in registers
only, so a level keeps ``(B, T, N, D)`` tensors where the PyTorch path
keeps ``(B, T, N, R, D)`` ones.

The input of an inner pair is the previous contraction times the values,
``u = y_{l-1} (x) v_l``. A kernel can take the two factors instead of
``u`` and multiply them as it loads them, which is how :func:`level_chain`
runs a whole level of vector values: one op for all ``p`` pairs, no
``u`` tensor in memory, the gradient of every pair's values written in
place into one ``(B, T, N, p, D)`` buffer, and in the arctic semiring
nothing saved for the backward pass but the argmax bits.
:func:`level_scan` is one pair with ``u`` given (matrix values and the
count normalisations, which act between the pairs).

Every scan is the three-phase chunked scan of ``lru-torch``: each program
scans one chunk of time and stores its end state (phase 1), one program
per sequence combines the chunk states in order (phase 2), and each chunk
is rescanned from its incoming state (phase 3). The decay across a chunk
is exact (``exp(-rate * length)``, an offset in the log domain), so the
order of rounding is ``chunk + chunks`` steps deep instead of ``T``.

A program holds the states of ``BN`` heads of one sequence, a
``(BN, R, D)`` tile, so a step loads all of its heads' inputs at once.
:func:`_blocks` chooses ``BN`` from the shape: several heads pay at value
widths that fill their tile, one head is better where padding would
waste most of it.

Backward passes (all with the same three phases):

- **reals**: the adjoint state ``G_t = lambda G_{t+1} + q_t dy_t^T`` runs
  backwards and gives ``du`` and ``dk``; ``dq_t = S dy_t`` rescans the
  state forwards from the chunk states saved by the forward pass.
- **log**: the gradient of an input is its posterior times ``dy``,
  ``exp(x_s) sum_t lambda^{t-s} exp(q_t - y_t) dy_t``. The sum runs
  backwards in a rescaled form, ``e^M H`` with a running maximum ``M``
  (online softmax), so neither factor overflows and no state is needed;
  ``dq`` rescans the state forwards like the reals.
- **arctic**: the gradient follows the argmaxes. The forward pass records,
  per chunk of 64 steps, one bit per state entry and step for "took the new
  value" and one for "the contraction took this feature" (two ``int64``
  per entry and chunk), and the backward pass is a reverse scan gated by
  them.

The decay rate's gradient is a sum over pairs of index distances, which
``rate_gradient`` writes in terms of the per-position gradients the
kernels return.
"""
import math

import torch
import triton
import triton.language as tl

SEMIRINGS = {"reals": 0, "log": 1, "arctic": 2}
"""The semirings the kernels implement, and their code in the kernels."""

ARCTIC_CHUNK = 64
"""Steps per chunk of the arctic scan: the bits of one ``int64``."""


# --------------------------------------------------------------------------
# the semiring arithmetic of a state tile (BN heads, BR features, BD values)
# --------------------------------------------------------------------------
@triton.jit
def _input(prev_ptr, v_ptr, out, v_off, mask, SR: tl.constexpr,
           HAS_PREV: tl.constexpr, ACC: tl.constexpr):
    """``u = y_prev (x) v`` at one step, ``(BN, BD)``; ``u = v`` for the
    first pair (or when ``u`` is given)."""
    v = tl.load(v_ptr + v_off, mask=mask, other=0.0).to(ACC)
    if HAS_PREV:
        prev = tl.load(prev_ptr + out, mask=mask, other=0.0).to(ACC)
        if SR == 0:
            return prev * v
        return prev + v
    return v


@triton.jit
def _outer(k, u, SR: tl.constexpr):
    """The scan input ``k_r (x) u_d``, ``(BN, BR, BD)``."""
    if SR == 0:
        return k[:, :, None] * u[:, None, :]
    return k[:, :, None] + u[:, None, :]


@triton.jit
def _accumulate(S, x, rate, lam, SR: tl.constexpr):
    """``lambda (x) S (+) x``: one step of the scan."""
    if SR == 0:
        return lam * S + x
    carried = S - rate
    if SR == 1:
        high = tl.maximum(carried, x)
        return high + tl.log(1.0 + tl.exp(-tl.abs(carried - x)))
    return tl.maximum(carried, x)


@triton.jit
def _contract(S, q, SR: tl.constexpr):
    """``<q, S>``: the semiring sum over the features, ``(BN, BD)``."""
    if SR == 0:
        return tl.sum(q[:, :, None] * S, axis=1)
    z = q[:, :, None] + S
    high = tl.max(z, axis=1)
    if SR == 1:
        return high + tl.log(tl.sum(tl.exp(z - high[:, None, :]), axis=1))
    return high


@triton.jit
def _factor(ptr, base, t, stride, mask, HAS: tl.constexpr, ONE: tl.constexpr,
            PAD: tl.constexpr, ACC: tl.constexpr):
    """A key or query tile ``(BN, BR)`` at step ``t``; without the factor,
    the semiring's one (``PAD`` on the padded features)."""
    if HAS:
        return tl.load(ptr + base + t * stride, mask=mask, other=PAD).to(ACC)
    return tl.where(mask, ONE, PAD).to(ACC)


@triton.jit
def _plane(b, B, N, C, R, D):
    """The size of one ``(B * N, C, R, D)`` plane of a chunk buffer, in
    int64 like the sequence index ``b`` (the scalars are int32)."""
    return (b * 0 + B) * N * C * R * D


# --------------------------------------------------------------------------
# forward: phases 1 and 3, and the forward rescan of the backward (dq)
# --------------------------------------------------------------------------
@triton.jit
def _forward_kernel(
    prev_ptr, v_ptr, k_ptr, q_ptr, rate_ptr, y_ptr, dy_ptr, dq_ptr,
    start_ptr, end_ptr, bits_ptr,
    B, T, N, D, R, C, s_rate,
    sv_b, sv_t, sv_n, sv_d,
    sk_b, sk_t, sk_n, sk_r,
    sq_b, sq_t, sq_n, sq_r,
    sy_b, sy_t, sy_n, sy_d,
    so_j, so_b, so_t, so_n, so_r,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BN: tl.constexpr,
    BR: tl.constexpr, BD: tl.constexpr, SR: tl.constexpr,
    INCLUSIVE: tl.constexpr, HAS_PREV: tl.constexpr, HAS_K: tl.constexpr,
    HAS_Q: tl.constexpr, HAS_RATE: tl.constexpr, EMPTY: tl.constexpr,
    ONE: tl.constexpr, QPAD: tl.constexpr, ACC: tl.constexpr,
):
    """MODE 0: the end state of the chunk (phase 1). MODE 1: the outputs
    ``y`` from the chunk's start state (phase 3), and in the arctic
    semiring the argmax bits. MODE 2: ``dq = d<q, S>/dq`` from the start
    state (backward, reals and log; ``dy`` strided)."""
    pid = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    n_mask = n < N
    nr = n_mask[:, None] & (r < R)[None, :]
    nd = n_mask[:, None] & (d < D)[None, :]
    nrd = nr[:, :, None] & (d < D)[None, None, :]
    state = (((b * N + n[:, None, None]) * C + c) * R
             + r[None, :, None]) * D + d[None, None, :]
    if HAS_RATE:
        rate = tl.load(rate_ptr + n * s_rate, mask=n_mask,
                       other=0.0).to(ACC)[:, None, None]
    else:
        rate = tl.zeros((BN, 1, 1), ACC)
    lam = tl.exp(-rate)
    if MODE == 0:
        S = tl.full((BN, BR, BD), EMPTY, ACC)
    else:
        S = tl.load(start_ptr + state, mask=nrd, other=EMPTY).to(ACC)
    if SR == 2:
        took = tl.zeros((BN, BR, BD), tl.int64)
        path = tl.zeros((BN, BR, BD), tl.int64)
    v_base = b * sv_b + n[:, None] * sv_n + d[None, :] * sv_d
    k_base = b * sk_b + n[:, None] * sk_n + r[None, :] * sk_r
    q_base = b * sq_b + n[:, None] * sq_n + r[None, :] * sq_r
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for tau in range(0, length):
        t = (t0 + tau).to(tl.int64)
        out = ((b * T + t) * N + n[:, None]) * D + d[None, :]
        u = _input(prev_ptr, v_ptr, out, v_base + t * sv_t, nd, SR, HAS_PREV,
                   ACC)
        k = _factor(k_ptr, k_base, t, sk_t, nr, HAS_K, ONE, EMPTY, ACC)
        x = _outer(k, u, SR)
        if MODE != 0:
            q = _factor(q_ptr, q_base, t, sq_t, nr, HAS_Q, ONE, QPAD, ACC)
        if MODE != 0 and not INCLUSIVE:
            read = S
        S_new = _accumulate(S, x, rate, lam, SR)
        if SR == 2 and MODE == 1:
            took = took | ((x >= S - rate).to(tl.int64) << tau.to(tl.int64))
        S = S_new
        if MODE != 0 and INCLUSIVE:
            read = S
        if MODE == 1:
            tl.store(y_ptr + out, _contract(read, q, SR), mask=nd)
            if SR == 2:
                best = tl.argmax(q[:, :, None] + read, axis=1)
                hit = r[None, :, None] == best[:, None, :]
                path = path | (hit.to(tl.int64) << tau.to(tl.int64))
        if MODE == 2:
            dy = tl.load(dy_ptr + b * sy_b + t * sy_t + n[:, None] * sy_n
                         + d[None, :] * sy_d, mask=nd, other=0.0).to(ACC)
            if SR == 0:
                dq = tl.sum(read * dy[:, None, :], axis=2)
            else:
                y = tl.load(y_ptr + out, mask=nd, other=0.0).to(ACC)
                weight = tl.exp(tl.minimum(
                    q[:, :, None] + read - y[:, None, :], 0.0))
                weight = tl.where(dy[:, None, :] != 0,
                                  weight * dy[:, None, :], 0.0)
                dq = tl.sum(weight, axis=2)
            dq_off = j * so_j + b * so_b + t * so_t + n[:, None] * so_n \
                + r[None, :] * so_r
            tl.store(dq_ptr + dq_off, dq, mask=nr)
    if MODE == 0:
        tl.store(end_ptr + state, S, mask=nrd)
    if SR == 2 and MODE == 1:
        plane = _plane(b, B, N, C, R, D)
        tl.store(bits_ptr + state, took, mask=nrd)
        tl.store(bits_ptr + plane + state, path, mask=nrd)


@triton.jit
def _forward_prefix_kernel(
    end_ptr, start_ptr, rate_ptr,
    N, D, R, C, s_rate,
    CHUNK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    BD: tl.constexpr, SR: tl.constexpr, HAS_RATE: tl.constexpr,
    EMPTY: tl.constexpr, ACC: tl.constexpr, U: tl.constexpr,
):
    """Phase 2: the state entering every chunk, from the chunks' end
    states, in order. Every chunk but the last is full. The end states of
    ``U`` chunks are loaded at once: they do not depend on the chain, whose
    every step would otherwise wait for its load."""
    pid = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    n_mask = n < N
    nrd = n_mask[:, None, None] & (r < R)[None, :, None] \
        & (d < D)[None, None, :]
    if HAS_RATE:
        rate = tl.load(rate_ptr + n * s_rate, mask=n_mask,
                       other=0.0).to(ACC)[:, None, None] * CHUNK
    else:
        rate = tl.zeros((BN, 1, 1), ACC)
    lam = tl.exp(-rate)
    P = tl.full((BN, BR, BD), EMPTY, ACC)
    base = (((b * N + n[:, None, None]) * C) * R + r[None, :, None]) * D \
        + d[None, None, :]
    for c0 in range(0, C, U):
        E = ()
        for i in tl.static_range(U):
            E = E + (tl.load(end_ptr + base + (c0 + i) * R * D,
                             mask=nrd & (c0 + i < C), other=EMPTY).to(ACC),)
        for i in tl.static_range(U):
            tl.store(start_ptr + base + (c0 + i) * R * D, P,
                     mask=nrd & (c0 + i < C))
            P = _accumulate(P, E[i], rate, lam, SR)


# --------------------------------------------------------------------------
# backward, reals and log: the reverse scan of the adjoint
# --------------------------------------------------------------------------
@triton.jit
def _reverse_kernel(
    dy_ptr, y_ptr, prev_ptr, v_ptr, k_ptr, q_ptr, rate_ptr,
    dv_ptr, dprev_ptr, dk_ptr, start_ptr, end_ptr,
    B, T, N, D, R, C, s_rate,
    sv_b, sv_t, sv_n, sv_d,
    sk_b, sk_t, sk_n, sk_r,
    sq_b, sq_t, sq_n, sq_r,
    sy_b, sy_t, sy_n, sy_d,
    sg_b, sg_t, sg_n, sg_d,
    so_j, so_b, so_t, so_n, so_r,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BN: tl.constexpr,
    BR: tl.constexpr, BD: tl.constexpr, SR: tl.constexpr,
    INCLUSIVE: tl.constexpr, HAS_PREV: tl.constexpr, HAS_K: tl.constexpr,
    HAS_Q: tl.constexpr, HAS_RATE: tl.constexpr, ZERO: tl.constexpr,
    ONE: tl.constexpr, QPAD: tl.constexpr, ACC: tl.constexpr,
):
    """The adjoint ``H_s = sum_{t >= s} lambda^{t-s} a_t`` backwards over
    one chunk: ``a_t = q_t dy_t^T`` (reals), or ``exp(q_t - y_t) dy_t``
    held as ``e^M H`` (log). MODE 0 stores the chunk's own sum (phase 1);
    MODE 1 starts from the sum entering it from later chunks and writes
    the gradients of the values (``dv``, strided), of the previous
    contraction (``dprev``, reals only: in the log domain it is ``dv``)
    and of the keys (phase 3)."""
    pid = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    n_mask = n < N
    nr = n_mask[:, None] & (r < R)[None, :]
    nd = n_mask[:, None] & (d < D)[None, :]
    nrd = nr[:, :, None] & (d < D)[None, None, :]
    state = (((b * N + n[:, None, None]) * C + c) * R
             + r[None, :, None]) * D + d[None, None, :]
    plane = _plane(b, B, N, C, R, D)
    if HAS_RATE:
        rate = tl.load(rate_ptr + n * s_rate, mask=n_mask,
                       other=0.0).to(ACC)[:, None, None]
    else:
        rate = tl.zeros((BN, 1, 1), ACC)
    lam = tl.exp(-rate)
    H = tl.zeros((BN, BR, BD), ACC)
    M = tl.full((BN, BR, BD), ZERO, ACC)
    if MODE == 1:
        H = tl.load(start_ptr + state, mask=nrd, other=0.0).to(ACC)
        if SR == 1:
            M = tl.load(start_ptr + plane + state, mask=nrd,
                        other=ZERO).to(ACC)
    v_base = b * sv_b + n[:, None] * sv_n + d[None, :] * sv_d
    k_base = b * sk_b + n[:, None] * sk_n + r[None, :] * sk_r
    q_base = b * sq_b + n[:, None] * sq_n + r[None, :] * sq_r
    y_base = b * sy_b + n[:, None] * sy_n + d[None, :] * sy_d
    g_base = b * sg_b + n[:, None] * sg_n + d[None, :] * sg_d
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for i in range(0, length):
        t = (t0 + length - 1 - i).to(tl.int64)
        out = ((b * T + t) * N + n[:, None]) * D + d[None, :]
        dy = tl.load(dy_ptr + y_base + t * sy_t, mask=nd, other=0.0).to(ACC)
        q = _factor(q_ptr, q_base, t, sq_t, nr, HAS_Q, ONE, QPAD, ACC)
        if MODE == 1 and not INCLUSIVE:
            H_read = H
            M_read = M
        # add a_t
        if SR == 0:
            H = lam * H + q[:, :, None] * dy[:, None, :]
        else:
            y = tl.load(y_ptr + out, mask=nd, other=0.0).to(ACC)
            # log|a_t|; -inf where dy = 0, which adds nothing
            level = q[:, :, None] - y[:, None, :] \
                + tl.log(tl.abs(dy))[:, None, :]
            carried = M - rate
            high = tl.maximum(carried, level)
            sign = tl.where(dy > 0, 1.0, -1.0)
            H = H * tl.exp(carried - high) \
                + sign[:, None, :] * tl.exp(level - high)
            M = high
        if MODE == 1:
            if INCLUSIVE:
                H_read = H
                M_read = M
            v = tl.load(v_ptr + v_base + t * sv_t, mask=nd,
                        other=0.0).to(ACC)
            u = v
            if HAS_PREV:
                prev = tl.load(prev_ptr + out, mask=nd, other=0.0).to(ACC)
                if SR == 0:
                    u = prev * v
                else:
                    u = prev + v
            k = _factor(k_ptr, k_base, t, sk_t, nr, HAS_K, ONE, 0.0, ACC)
            if SR == 0:
                du = tl.sum(k[:, :, None] * H_read, axis=1)
                dk = tl.sum(H_read * u[:, None, :], axis=2)
            else:
                e = M_read + u[:, None, :] + k[:, :, None]
                g = tl.exp(tl.minimum(e, 80.0)) * H_read
                g = tl.where(nrd, g, 0.0)
                du = tl.sum(g, axis=1)
                dk = tl.sum(g, axis=2)
            dv = du
            if HAS_PREV and SR == 0:
                dv = du * prev
                tl.store(dprev_ptr + out, du * v, mask=nd)
            tl.store(dv_ptr + g_base + t * sg_t, dv, mask=nd)
            if HAS_K:
                dk_off = j * so_j + b * so_b + t * so_t + n[:, None] * so_n \
                    + r[None, :] * so_r
                tl.store(dk_ptr + dk_off, dk, mask=nr)
    if MODE == 0:
        tl.store(end_ptr + state, H, mask=nrd)
        if SR == 1:
            tl.store(end_ptr + plane + state, M, mask=nrd)


@triton.jit
def _reverse_prefix_kernel(
    end_ptr, start_ptr, rate_ptr,
    B, T, N, D, R, C, s_rate,
    CHUNK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
    BD: tl.constexpr, SR: tl.constexpr, HAS_RATE: tl.constexpr,
    ZERO: tl.constexpr, ACC: tl.constexpr, U: tl.constexpr,
):
    """Phase 2 backwards: the adjoint entering every chunk from the later
    ones, ``P_{c-1} = E_c (+) lambda^{len(c)} P_c``, ``U`` chunks' loads at
    once."""
    pid = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    n_mask = n < N
    nrd = n_mask[:, None, None] & (r < R)[None, :, None] \
        & (d < D)[None, None, :]
    plane = _plane(b, B, N, C, R, D)
    if HAS_RATE:
        rate = tl.load(rate_ptr + n * s_rate, mask=n_mask,
                       other=0.0).to(ACC)[:, None, None]
    else:
        rate = tl.zeros((BN, 1, 1), ACC)
    H = tl.zeros((BN, BR, BD), ACC)
    M = tl.full((BN, BR, BD), ZERO, ACC)
    base = (((b * N + n[:, None, None]) * C) * R + r[None, :, None]) * D \
        + d[None, None, :]
    for i0 in range(0, C, U):
        E = ()
        E_M = ()
        for i in tl.static_range(U):
            live = nrd & (i0 + i < C)
            state = base + (C - 1 - i0 - i) * R * D
            E = E + (tl.load(end_ptr + state, mask=live, other=0.0).to(ACC),)
            if SR == 1:
                E_M = E_M + (tl.load(end_ptr + plane + state, mask=live,
                                     other=ZERO).to(ACC),)
        for i in tl.static_range(U):
            c = C - 1 - i0 - i
            live = nrd & (i0 + i < C)
            state = base + c * R * D
            tl.store(start_ptr + state, H, mask=live)
            if SR == 1:
                tl.store(start_ptr + plane + state, M, mask=live)
            length = tl.minimum(CHUNK, T - c * CHUNK)
            if SR == 0:
                H = tl.exp(-rate * length) * H + E[i]
            else:
                carried = M - rate * length
                high = tl.maximum(carried, E_M[i])
                H = H * tl.exp(carried - high) + E[i] * tl.exp(E_M[i] - high)
                M = high


# --------------------------------------------------------------------------
# backward, arctic: the reverse scan gated by the argmax bits
# --------------------------------------------------------------------------
@triton.jit
def _arctic_reverse_kernel(
    dy_ptr, bits_ptr, dv_ptr, dk_ptr, dq_ptr, drate_ptr, start_ptr, end_ptr,
    B, T, N, D, R, C,
    sy_b, sy_t, sy_n, sy_d,
    sg_b, sg_t, sg_n, sg_d,
    so_j, so_b, so_t, so_n, so_r,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BN: tl.constexpr,
    BR: tl.constexpr, BD: tl.constexpr, INCLUSIVE: tl.constexpr,
    HAS_K: tl.constexpr, HAS_Q: tl.constexpr, HAS_RATE: tl.constexpr,
    ACC: tl.constexpr,
):
    """The gradient ``K`` reaching the state from later steps, backwards
    over one chunk: a step that took its new value passes ``K`` to its
    input, one that carried the old state passes it on. MODE 0 stores the
    chunk's own ``K`` and whether it passes everything through (phase 1);
    MODE 1 starts from the ``K`` entering it and writes ``du`` (into
    ``dv``, strided: in max-plus it is also the gradient of the values and
    of the previous contraction), ``dk`` and ``dq`` (phase 3), and the
    chunk's part of the rate gradient: a step that carried its state took
    ``-rate`` with it, so ``dL/d rate`` is minus every gradient carried
    (per head, ``(NDB, B, N, C)``)."""
    pid = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    n_mask = n < N
    nr = n_mask[:, None] & (r < R)[None, :]
    nd = n_mask[:, None] & (d < D)[None, :]
    nrd = nr[:, :, None] & (d < D)[None, None, :]
    state = (((b * N + n[:, None, None]) * C + c) * R
             + r[None, :, None]) * D + d[None, None, :]
    plane = _plane(b, B, N, C, R, D)
    took = tl.load(bits_ptr + state, mask=nrd, other=0)
    path = tl.load(bits_ptr + plane + state, mask=nrd, other=0)
    if MODE == 1:
        K = tl.load(start_ptr + state, mask=nrd, other=0.0).to(ACC)
    else:
        K = tl.zeros((BN, BR, BD), ACC)
    y_base = b * sy_b + n[:, None] * sy_n + d[None, :] * sy_d
    g_base = b * sg_b + n[:, None] * sg_n + d[None, :] * sg_d
    carried = tl.zeros((BN, BR, BD), ACC)
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for i in range(0, length):
        tau = length - 1 - i
        t = (t0 + tau).to(tl.int64)
        dy = tl.load(dy_ptr + y_base + t * sy_t, mask=nd, other=0.0).to(ACC)
        new = ((took >> tau) & 1).to(ACC)
        hit = ((path >> tau) & 1).to(ACC) * dy[:, None, :]
        if INCLUSIVE:
            G = K + hit
        else:
            G = K
        dx = new * G
        K = G - dx
        if MODE == 1 and HAS_RATE:
            carried += K
        if not INCLUSIVE:
            K = K + hit
        if MODE == 1:
            tl.store(dv_ptr + g_base + t * sg_t, tl.sum(dx, axis=1), mask=nd)
            off = j * so_j + b * so_b + t * so_t + n[:, None] * so_n \
                + r[None, :] * so_r
            if HAS_K:
                tl.store(dk_ptr + off, tl.sum(dx, axis=2), mask=nr)
            if HAS_Q:
                tl.store(dq_ptr + off, tl.sum(hit, axis=2), mask=nr)
    if MODE == 0:
        tl.store(end_ptr + state, K, mask=nrd)
        tl.store(end_ptr + plane + state, (took == 0).to(ACC), mask=nrd)
    if MODE == 1 and HAS_RATE:
        drate = -tl.sum(tl.sum(tl.where(nrd, carried, 0.0), axis=2), axis=1)
        tl.store(drate_ptr + ((j * B + b) * N + n) * C + c, drate,
                 mask=n_mask)


@triton.jit
def _arctic_reverse_prefix_kernel(
    end_ptr, start_ptr,
    B, N, D, R, C,
    BN: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
    ACC: tl.constexpr, U: tl.constexpr,
):
    """Phase 2 backwards: ``K_{c-1} = local_c + through_c * K_c``, ``U``
    chunks' loads at once."""
    pid = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    NB = tl.cdiv(N, BN)
    b = pid // NB
    n = (pid % NB) * BN + tl.arange(0, BN)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    nrd = (n < N)[:, None, None] & (r < R)[None, :, None] \
        & (d < D)[None, None, :]
    plane = _plane(b, B, N, C, R, D)
    K = tl.zeros((BN, BR, BD), ACC)
    base = (((b * N + n[:, None, None]) * C) * R + r[None, :, None]) * D \
        + d[None, None, :]
    for i0 in range(0, C, U):
        local = ()
        through = ()
        for i in tl.static_range(U):
            live = nrd & (i0 + i < C)
            state = base + (C - 1 - i0 - i) * R * D
            local = local + (tl.load(end_ptr + state, mask=live,
                                     other=0.0).to(ACC),)
            through = through + (tl.load(end_ptr + plane + state, mask=live,
                                         other=0.0).to(ACC),)
        for i in tl.static_range(U):
            tl.store(start_ptr + base + (C - 1 - i0 - i) * R * D, K,
                     mask=nrd & (i0 + i < C))
            K = local[i] + through[i] * K


# --------------------------------------------------------------------------
# launching
# --------------------------------------------------------------------------
def _chunk(T: int, semiring: str) -> int:
    if semiring == "arctic":
        return ARCTIC_CHUNK
    if T <= 8192:
        return 64
    return 128 if T <= 32768 else 256


def _blocks(
    N: int,
    R: int,
    D: int,
    semiring: str,
    backward: bool,
) -> tuple[int, int, int, int]:
    """``(BN, BR, BD, num_warps)``: ``BN`` heads, the whole ``R`` and up to
    ``BD`` values per program (a tile of at most 2048 entries).

    Measured on a 3090 (B=8, T=20,000, every ``BN`` and warp count for
    ``N`` 4-16, ``R`` 1-8, ``D`` 6-64): a program is best at about 128
    head x value lanes, whatever ``R``; the arctic scans want at most 256
    tile entries forward and 128 backward; and in the log domain and
    arctic one head per program is best when padding would leave over 40 %
    of the lanes empty (``D = 17`` in 32); the reals then want at most 256
    entries. Against one head per program the rule gains 1.19x (arctic),
    1.41x (reals) and 1.47x (log), within 3-5 % of the best choice per
    shape, and up to 2-3x at narrow values with many heads; no shape is
    slower. Splitting a padded width into narrower value blocks (17 as 3 x
    8) does not pay: padding stays a cost, 1.2-2x the time of a width of
    16.
    """
    BR = triton.next_power_of_2(R)
    BD = min(triton.next_power_of_2(D), max(1, 2048 // BR))
    padded = D < 0.6 * BD
    lanes = BD if padded and semiring != "reals" else 128
    tile = 2048
    if semiring == "arctic":
        tile = 128 if backward else 256
    elif padded:
        tile = 256
    BN = 1
    while (2 * BN <= triton.next_power_of_2(N) and 2 * BN * BD <= lanes
           and 2 * BN * BR * BD <= tile):
        BN *= 2
    size = BN * BR * BD
    small = {"log": 128, "reals": 256}.get(semiring, 512)
    warps = 1 if size <= small else 2 if size <= 4 * small else 4
    return BN, BR, BD, warps


def _rank(key: torch.Tensor | None, query: torch.Tensor | None) -> int:
    for a in (key, query):
        if a is not None:
            return a.shape[-1]
    return 1


def _zero(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min / 8


def _pads(SR: int, zero: float) -> tuple[float, float, float]:
    """The empty state, the semiring's one, and the query of a padded
    feature: ``(0, 1, 0)`` in the reals, ``(zero, 0, -inf)`` in the log
    domain."""
    return (0.0, 1.0, 0.0) if SR == 0 else (zero, 0.0, -math.inf)


def _acc(*tensors: torch.Tensor | None) -> tuple[torch.dtype, object]:
    dtype = torch.float32
    for tensor in tensors:
        if tensor is not None and tensor.dtype == torch.float64:
            dtype = torch.float64
    return dtype, (tl.float64 if dtype == torch.float64 else tl.float32)


def _strides(a: torch.Tensor | None, k: int = 4) -> tuple[int, ...]:
    return (0,) * k if a is None else tuple(a.stride())  # type: ignore


class _Pair:
    """The geometry of one pair's scans: shapes, chunks, tiles, flags."""

    def __init__(
        self,
        v: torch.Tensor,
        key: torch.Tensor | None,
        query: torch.Tensor | None,
        rate: torch.Tensor | None,
        semiring: str,
        inclusive: bool,
        dtype: torch.dtype | None = None,
        rank: int | None = None,
        backward: bool = False,
    ) -> None:
        self.B, self.T, self.N, self.D = v.shape
        self.R = rank if rank is not None else _rank(key, query)
        if dtype is None:
            dtype = _acc(v, key, query)[0]
        self.dtype = dtype
        self.ACC = tl.float64 if dtype == torch.float64 else tl.float32
        self.semiring = semiring
        self.SR = SEMIRINGS[semiring]
        self.CHUNK = _chunk(self.T, semiring)
        self.C = triton.cdiv(self.T, self.CHUNK)
        self.BN, self.BR, self.BD, self.warps = _blocks(
            self.N, self.R, self.D, semiring, backward,
        )
        self.NB = triton.cdiv(self.N, self.BN)
        self.NDB = triton.cdiv(self.D, self.BD)
        # chunks per load batch of phase 2: ~64 registers of end states
        threads = 32 * self.warps
        tile = self.BN * self.BR * self.BD
        self.unroll = max(1, min(8, 64 * threads // tile))
        self.zero = _zero(dtype)
        self.empty, self.one, self.qpad = _pads(self.SR, self.zero)
        self.inclusive = inclusive
        self.has_rate = rate is not None
        self.s_rate = 0 if rate is None else rate.stride(0)

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.B * self.NB, self.C, self.NDB)

    @property
    def prefix_grid(self) -> tuple[int, int]:
        return (self.B * self.NB, self.NDB)

    def states(self, planes: int = 0) -> tuple[int, ...]:
        """The shape of a chunk buffer, with ``planes`` leading planes."""
        shape = (self.B * self.N, self.C, self.R, self.D)
        return ((planes,) if planes else ()) + shape

    def tiles(self) -> dict:
        return dict(CHUNK=self.CHUNK, BN=self.BN, BR=self.BR, BD=self.BD,
                    num_warps=self.warps)


def _forward(
    pair: _Pair,
    prev: torch.Tensor | None,
    v: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    y: torch.Tensor,
    saved: torch.Tensor,
) -> None:
    """Phases 1-3 of one pair into ``y`` (contiguous ``(B, T, N, D)``) and
    ``saved``: the chunk start states ``states()``, or in the arctic
    semiring the bits ``states(2)``, ``int64``."""
    SR = pair.SR
    arctic = SR == 2
    start = (torch.empty(pair.states(), device=v.device, dtype=pair.dtype)
             if arctic else saved)
    bits = saved if arctic else start
    common = dict(
        SR=SR, INCLUSIVE=pair.inclusive, HAS_PREV=prev is not None,
        HAS_K=key is not None, HAS_Q=query is not None,
        HAS_RATE=rate is not None, EMPTY=pair.empty, ONE=pair.one,
        QPAD=pair.qpad, ACC=pair.ACC, **pair.tiles(),
    )
    args = (
        pair.B, pair.T, pair.N, pair.D, pair.R, pair.C, pair.s_rate,
        *_strides(v), *_strides(key), *_strides(query), *(0,) * 9,
    )
    k = v if key is None else key
    q = v if query is None else query
    rt = v if rate is None else rate
    pv = v if prev is None else prev
    if pair.C > 1:
        end = torch.empty(pair.states(), device=v.device, dtype=pair.dtype)
        _forward_kernel[pair.grid](
            pv, v, k, q, rt, y, v, v, start, end, bits, *args,
            MODE=0, **common,
        )
        _forward_prefix_kernel[pair.prefix_grid](
            end, start, rt, pair.N, pair.D, pair.R, pair.C, pair.s_rate,
            SR=SR, HAS_RATE=rate is not None, EMPTY=pair.empty,
            ACC=pair.ACC, U=pair.unroll, **pair.tiles(),
        )
    else:
        start.fill_(pair.empty)
    _forward_kernel[pair.grid](
        pv, v, k, q, rt, y, v, v, start, v, bits, *args,
        MODE=1, **common,
    )


def _backward(
    pair: _Pair,
    dy: torch.Tensor,
    y: torch.Tensor | None,
    saved: torch.Tensor,
    prev: torch.Tensor | None,
    v: torch.Tensor | None,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    dv: torch.Tensor,
    dprev: torch.Tensor | None,
    dk: torch.Tensor | None,
    dq: torch.Tensor | None,
    has_key: bool,
    has_query: bool,
    drate: torch.Tensor | None = None,
) -> None:
    """The gradients of one pair, written into ``dv`` (``(B, T, N, D)``,
    any strides), ``dprev`` (reals with a previous pair; contiguous) and
    the ``(NDB, B, T, N, R)`` views ``dk``, ``dq`` (summed over their
    first axis by the caller). ``dy`` may have any strides. The arctic
    backward reads only ``dy`` and the bits, and writes the rate gradient
    per head and chunk into ``drate`` (``(NDB, B, N, C)``) if given."""
    SR = pair.SR
    device = dy.device
    grid = pair.grid
    out_strides = _strides(dk if dk is not None else dq, 5)
    if SR == 2:
        flags = dict(INCLUSIVE=pair.inclusive, HAS_K=dk is not None,
                     HAS_Q=dq is not None, HAS_RATE=drate is not None,
                     ACC=pair.ACC, **pair.tiles())
        args = (pair.B, pair.T, pair.N, pair.D, pair.R, pair.C,
                *_strides(dy), *_strides(dv), *out_strides)
        start = torch.zeros(pair.states(), device=device, dtype=pair.dtype)
        if pair.C > 1:
            end = torch.empty(pair.states(2), device=device,
                              dtype=pair.dtype)
            _arctic_reverse_kernel[grid](
                dy, saved, dv, dv, dv, dv, start, end, *args, MODE=0,
                **flags,
            )
            _arctic_reverse_prefix_kernel[pair.prefix_grid](
                end, start, pair.B, pair.N, pair.D, pair.R, pair.C,
                BN=pair.BN, BR=pair.BR, BD=pair.BD, ACC=pair.ACC,
                U=pair.unroll, num_warps=pair.warps,
            )
        _arctic_reverse_kernel[grid](
            dy, saved, dv, dv if dk is None else dk, dv if dq is None else dq,
            dv if drate is None else drate, start, dv, *args, MODE=1,
            **flags,
        )
        return
    assert v is not None
    common = dict(
        SR=SR, INCLUSIVE=pair.inclusive, HAS_PREV=prev is not None,
        HAS_K=has_key, HAS_Q=has_query, HAS_RATE=rate is not None,
        ONE=pair.one, QPAD=pair.qpad, ACC=pair.ACC, **pair.tiles(),
    )
    k = dv if key is None else key
    q = dv if query is None else query
    rt = dv if rate is None else rate
    pv = dv if prev is None else prev
    yy = dv if y is None else y
    if dq is not None:
        _forward_kernel[grid](
            pv, v, k, q, rt, yy, dy, dq, saved, dv, dv,
            pair.B, pair.T, pair.N, pair.D, pair.R, pair.C, pair.s_rate,
            *_strides(v), *_strides(key), *_strides(query), *_strides(dy),
            *_strides(dq, 5), MODE=2, EMPTY=pair.empty, **common,
        )
    planes = 1 if SR == 0 else 2
    start = torch.zeros(pair.states(planes), device=device, dtype=pair.dtype)
    if SR == 1:
        start[1] = pair.zero
    args = (
        pair.B, pair.T, pair.N, pair.D, pair.R, pair.C, pair.s_rate,
        *_strides(v), *_strides(key), *_strides(query), *_strides(dy),
        *_strides(dv), *out_strides,
    )
    if pair.C > 1:
        end = torch.empty_like(start)
        _reverse_kernel[grid](
            dy, yy, pv, v, k, q, rt, dv, dv, dv, dv, end, *args,
            MODE=0, ZERO=pair.zero, **common,
        )
        _reverse_prefix_kernel[pair.prefix_grid](
            end, start, rt, pair.B, pair.T, pair.N, pair.D, pair.R, pair.C,
            pair.s_rate, SR=SR, HAS_RATE=rate is not None, ZERO=pair.zero,
            ACC=pair.ACC, U=pair.unroll, **pair.tiles(),
        )
    _reverse_kernel[grid](
        dy, yy, pv, v, k, q, rt, dv, dv if dprev is None else dprev,
        dv if dk is None else dk, start, dv, *args,
        MODE=1, ZERO=pair.zero, **common,
    )


def _rate_sums(
    incoming: torch.Tensor,
    outgoing: torch.Tensor,
    delta: torch.Tensor | float,
) -> torch.Tensor:
    """``sum_s s incoming_s - sum_t (t - delta)^+ outgoing_t`` over batch
    and time, in float64: ``(B, T, ...) -> (...)``."""
    T = incoming.shape[1]
    t = torch.arange(T, device=incoming.device, dtype=torch.float64)
    t = t.view((1, T) + (1,) * (incoming.ndim - 2))
    # t = 0 of an inner pair reads the empty state: no pair ends there.
    end = (t - delta).clamp_min(0.0)
    return (
        (t * incoming.double()).sum((0, 1))
        - (end * outgoing.double()).sum((0, 1))
    )


def rate_gradient(
    semiring: str,
    inclusive: bool,
    u: torch.Tensor,
    du: torch.Tensor,
    dy: torch.Tensor,
    query: torch.Tensor | None,
    dq: torch.Tensor | None,
) -> torch.Tensor:
    """``dL/d rate`` per head, ``(N,)``, in the reals and the log semiring.

    The decay weighs a pair ``(t, s)`` by ``exp(-rate (t - delta - s))``, so
    ``dL/d rate = -sum_{t,s} (t - delta - s) c(t, s)`` over the pair
    contributions ``c``. Summed over ``s`` they are ``dy_t . y_t`` (reals)
    or ``dy_t`` (log domain), over ``t`` they are ``du_s . u_s`` or
    ``du_s``, and in the reals ``dy_t . y_t = q_t . dq_t``. The sums over
    time run in float64. Accurate to the error of the ``O(T)`` terms, the
    same as autograd through the PyTorch path's ``exp(rate * t)``. (The
    arctic kernels sum the carried gradient instead, which is exact.)
    """
    if semiring == "reals":
        assert dq is not None
        outgoing = (dq * query).sum(-1) if query is not None else dq[..., 0]
        incoming = (du * u).sum(-1)
    else:
        outgoing = dy.sum(-1)
        incoming = du.sum(-1)
    return _rate_sums(incoming, outgoing, 0.0 if inclusive else 1.0)


def _empty(like: torch.Tensor) -> torch.Tensor:
    return like.new_empty((0,))


def _arctic_rate(buffer: torch.Tensor) -> torch.Tensor:
    """The per-chunk rate gradients ``(NDB, B, N, C)`` summed per head."""
    return buffer.double().sum((0, 1, 3))


# --------------------------------------------------------------------------
# one pair: the custom op
# --------------------------------------------------------------------------
@torch.library.custom_op("elissabeth::level_scan", mutates_args=())
def _level_scan_op(
    u: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
    inclusive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(y, saved)``: the contraction, and what the backward pass reads
    (the chunks' start states, or the arctic argmax bits)."""
    with torch.cuda.device(u.device):
        pair = _Pair(u, key, query, rate, semiring, inclusive)
        y = torch.empty((pair.B, pair.T, pair.N, pair.D), device=u.device,
                        dtype=pair.dtype)
        saved = (torch.empty(pair.states(2), device=u.device,
                             dtype=torch.int64) if pair.SR == 2
                 else torch.empty(pair.states(), device=u.device,
                                  dtype=pair.dtype))
        _forward(pair, None, u, key, query, rate, y, saved)
        return y, saved


@_level_scan_op.register_fake
def _(u, key, query, rate, semiring, inclusive):
    B, T, N, D = u.shape
    dtype, _ = _acc(u, key, query)
    R = _rank(key, query)
    C = -(-T // _chunk(T, semiring))
    y = u.new_empty((B, T, N, D), dtype=dtype)
    if semiring == "arctic":
        return y, u.new_empty((2, B * N, C, R, D), dtype=torch.int64)
    return y, u.new_empty((B * N, C, R, D), dtype=dtype)


@torch.library.custom_op("elissabeth::level_scan_backward", mutates_args=())
def _level_scan_backward_op(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    saved: torch.Tensor,
    u: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
    inclusive: bool,
    need_dq: bool,
    need_rate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(du, dk, dq, drate)``, with an empty tensor for a gradient not
    asked for; ``drate`` (float64) comes from the kernels in the arctic
    semiring only."""
    with torch.cuda.device(u.device):
        pair = _Pair(u, key, query, rate, semiring, inclusive,
                     backward=True)
        dy = dy.to(pair.dtype)
        shape = (pair.NDB, pair.B, pair.T, pair.N, pair.R)
        du = torch.empty((pair.B, pair.T, pair.N, pair.D), device=u.device,
                         dtype=pair.dtype)
        dk = (torch.empty(shape, device=u.device, dtype=pair.dtype)
              if key is not None else None)
        dq = (torch.empty(shape, device=u.device, dtype=pair.dtype)
              if need_dq else None)
        arctic_rate = need_rate and pair.SR == 2
        drate = (torch.empty((pair.NDB, pair.B, pair.N, pair.C),
                             device=u.device, dtype=pair.dtype)
                 if arctic_rate else None)
        _backward(pair, dy, y, saved, None, u, key, query, rate, du, None,
                  dk, dq, key is not None, query is not None, drate)
    return (
        du.to(u.dtype),
        _empty(u) if dk is None else dk.sum(0).to(key.dtype),  # type: ignore
        _empty(u) if dq is None else dq.sum(0).to(
            query.dtype if query is not None else u.dtype),
        _empty(u) if drate is None else _arctic_rate(drate),
    )


@_level_scan_backward_op.register_fake
def _(dy, y, saved, u, key, query, rate, semiring, inclusive, need_dq,
      need_rate):
    B, T, N, D = u.shape
    R = _rank(key, query)
    du = torch.empty_like(u, memory_format=torch.contiguous_format)
    dk = (_empty(u) if key is None
          else key.new_empty((B, T, N, R)))
    dq = (_empty(u) if not need_dq
          else u.new_empty((B, T, N, R),
                           dtype=query.dtype if query is not None
                           else u.dtype))
    drate = (u.new_empty((N,), dtype=torch.float64)
             if need_rate and semiring == "arctic" else _empty(u))
    return du, dk, dq, drate


def _setup(ctx, inputs, output) -> None:
    u, key, query, rate, semiring, inclusive = inputs
    y, saved = output
    ctx.semiring = semiring
    ctx.inclusive = inclusive
    ctx.save_for_backward(
        u, key, query, rate, y if semiring == "log" else None, saved,
    )


def _scan_backward(ctx, dy: torch.Tensor, _: torch.Tensor):
    u, key, query, rate, y, saved = ctx.saved_tensors
    need_rate = rate is not None and ctx.needs_input_grad[3]
    need_dq = query is not None or (ctx.semiring == "reals" and need_rate)
    du, dk, dq, drate = _level_scan_backward_op(
        dy.contiguous(), y, saved, u, key, query, rate, ctx.semiring,
        ctx.inclusive, need_dq, need_rate,
    )
    if need_rate and ctx.semiring != "arctic":
        drate = rate_gradient(
            ctx.semiring, ctx.inclusive, u, du, dy, query, dq,
        )
    return (
        du,
        dk if key is not None else None,
        dq if query is not None else None,
        drate.to(rate.dtype) if need_rate else None,
        None,
        None,
    )


_level_scan_op.register_autograd(_scan_backward, setup_context=_setup)


def level_scan(
    u: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
    inclusive: bool,
) -> torch.Tensor:
    """One fused scan of a LISS level, ``(B, T, N, d_v, w)``.

    ``u`` is ``(B|1, T, N|1, d_v, w)``, ``key`` and ``query`` are
    ``(B|1, T|1, N|1, R)`` or ``None`` (rank 1, factor one), ``rate`` is
    ``(N,)`` or ``None``. Broadcast axes are expanded as views, and the
    gradients are summed back over them by autograd.
    """
    d_v, w = u.shape[-2:]
    B, T, N = torch.broadcast_shapes(
        u.shape[:3],
        *(a.shape[:3] for a in (key, query) if a is not None),
        *(((1, 1, rate.shape[0]),) if rate is not None else ()),
    )
    flat = u.flatten(-2).expand(B, T, N, d_v * w)
    R = _rank(key, query)
    if key is not None:
        key = key.expand(B, T, N, R)
    if query is not None:
        query = query.expand(B, T, N, R)
    if rate is not None:
        rate = rate.contiguous()
    y, _ = _level_scan_op(flat, key, query, rate, semiring, inclusive)
    return y.unflatten(-1, (d_v, w))


# --------------------------------------------------------------------------
# a whole level of vector values: the custom op
# --------------------------------------------------------------------------
def _pair_of(a: torch.Tensor | None, l: int) -> torch.Tensor | None:
    return None if a is None else a[:, :, :, l]


@torch.library.custom_op("elissabeth::level_chain", mutates_args=())
def _level_chain_op(
    v: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y, saved, ys)``: the level's output; per pair what its backward
    reads (chunk start states, or the arctic bits); and the inner pairs'
    outputs (reals and log; empty in the arctic semiring, whose backward
    needs neither them nor the values)."""
    B, T, N, P, D = v.shape
    with torch.cuda.device(v.device):
        first = _Pair(v[:, :, :, 0], _pair_of(key, 0), _pair_of(query, 0),
                      None, semiring, False, rank=_rank(key, query))
        arctic = first.SR == 2
        y = torch.empty((B, T, N, D), device=v.device, dtype=first.dtype)
        saved = (torch.empty((P,) + first.states(2), device=v.device,
                             dtype=torch.int64) if arctic
                 else torch.empty((P,) + first.states(), device=v.device,
                                  dtype=first.dtype))
        inner = 0 if arctic else P - 1
        ys = torch.empty((inner, B, T, N, D), device=v.device,
                         dtype=first.dtype)
        scratch = (torch.empty((min(P - 1, 2), B, T, N, D), device=v.device,
                               dtype=first.dtype) if arctic else ys)
        prev = None
        for l in range(P):
            rate_l = None if rate is None else rate[:, l]
            pair = _Pair(v[:, :, :, l], _pair_of(key, l), _pair_of(query, l),
                         rate_l, semiring, l == P - 1, dtype=first.dtype,
                         rank=first.R)
            out = y if l == P - 1 else scratch[l % 2 if arctic else l]
            _forward(pair, prev, v[:, :, :, l], _pair_of(key, l),
                     _pair_of(query, l), rate_l, out, saved[l])
            prev = out
        return y, saved, ys


@_level_chain_op.register_fake
def _(v, key, query, rate, semiring):
    B, T, N, P, D = v.shape
    dtype, _ = _acc(v, key, query)
    R = _rank(key, query)
    C = -(-T // _chunk(T, semiring))
    y = v.new_empty((B, T, N, D), dtype=dtype)
    if semiring == "arctic":
        return (y, v.new_empty((P, 2, B * N, C, R, D), dtype=torch.int64),
                v.new_empty((0, B, T, N, D), dtype=dtype))
    return (y, v.new_empty((P, B * N, C, R, D), dtype=dtype),
            v.new_empty((P - 1, B, T, N, D), dtype=dtype))


def _pairs_last(a: torch.Tensor) -> torch.Tensor:
    """A pair-major ``(P, B, T, N, X)`` buffer as ``(B, T, N, P, X)``."""
    return a.permute(1, 2, 3, 0, 4)


@torch.library.custom_op("elissabeth::level_chain_backward", mutates_args=())
def _level_chain_backward_op(
    dy: torch.Tensor,
    y: torch.Tensor | None,
    saved: torch.Tensor,
    ys: torch.Tensor,
    v: torch.Tensor | None,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
    rank: int,
    has_key: bool,
    has_query: bool,
    need_rate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dv, dk, dq, drate)`` of a level, with an empty tensor for a
    gradient not asked for. ``dv``, ``dk`` and ``dq`` are ``(B, T, N, p,
    X)`` views of pair-major buffers, so every pair writes (and the next
    one reads) a contiguous slice. ``drate`` comes from the arctic kernels;
    in the reals ``dq`` is also returned without a query, for the rate
    gradient. ``v``, ``key`` and ``query`` are only read in the reals and
    the log semiring."""
    B, T, N, D = dy.shape
    P = saved.shape[0]
    SR = SEMIRINGS[semiring]
    device = dy.device
    dtype = dy.dtype
    with torch.cuda.device(device):
        dv = torch.empty((P, B, T, N, D), device=device, dtype=dtype)
        probe = _Pair(dy, None, None, None, semiring, False, dtype=dtype,
                      rank=rank, backward=True)
        out = (P, probe.NDB, B, T, N, rank)
        dk = torch.empty(out, device=device, dtype=dtype) if has_key else None
        need_dq = has_query or (SR == 0 and need_rate)
        dq = torch.empty(out, device=device, dtype=dtype) if need_dq else None
        dprev = (torch.empty((min(P - 1, 2), B, T, N, D), device=device,
                             dtype=dtype) if SR == 0 and P > 1 else None)
        drate = (torch.empty((P, probe.NDB, B, N, probe.C), device=device,
                             dtype=dtype) if need_rate and SR == 2 else None)
        grad = dy
        for l in reversed(range(P)):
            rate_l = None if rate is None else rate[:, l]
            pair = _Pair(dy, None, None, rate_l, semiring, l == P - 1,
                         dtype=dtype, rank=rank, backward=True)
            dprev_l = (dprev[(l - 1) % len(dprev)]
                       if dprev is not None and l > 0 else None)
            _backward(
                pair, grad, (y if l == P - 1 else ys[l]) if SR == 1 else None,
                saved[l], ys[l - 1] if l > 0 and SR != 2 else None,
                None if v is None else v[:, :, :, l], _pair_of(key, l),
                _pair_of(query, l), rate_l, dv[l], dprev_l,
                None if dk is None else dk[l], None if dq is None else dq[l],
                has_key, has_query, None if drate is None else drate[l],
            )
            grad = dprev_l if SR == 0 else dv[l]
    return (
        _pairs_last(dv),
        _empty(dy) if dk is None else _pairs_last(dk.sum(1)),
        _empty(dy) if dq is None else _pairs_last(dq.sum(1)),
        (_empty(dy) if drate is None
         else drate.double().sum((1, 2, 4)).T.contiguous()),
    )


@_level_chain_backward_op.register_fake
def _(dy, y, saved, ys, v, key, query, rate, semiring, rank, has_key,
      has_query, need_rate):
    B, T, N, D = dy.shape
    P = saved.shape[0]
    out = (P, B, T, N, rank)
    need_dq = has_query or (semiring == "reals" and need_rate)
    return (
        _pairs_last(dy.new_empty((P, B, T, N, D))),
        _pairs_last(dy.new_empty(out)) if has_key else _empty(dy),
        _pairs_last(dy.new_empty(out)) if need_dq else _empty(dy),
        dy.new_empty((N, P), dtype=torch.float64)
        if need_rate and semiring == "arctic" else _empty(dy),
    )


def _chain_rate_gradient(
    semiring: str,
    v: torch.Tensor,
    dv: torch.Tensor,
    dy: torch.Tensor,
    query: torch.Tensor | None,
    dq: torch.Tensor,
) -> torch.Tensor:
    """:func:`rate_gradient` for every pair of a level at once, ``(N, p)``.
    In the reals ``du . u = dv . v`` (``u = y_prev v``, ``dv = du y_prev``);
    in the log semiring the gradient reaching pair ``l``'s output is the
    gradient of pair ``l + 1``'s values."""
    P = dv.shape[3]
    if semiring == "reals":
        incoming = (dv * v).sum(-1)
        outgoing = (dq * query).sum(-1) if query is not None else dq[..., 0]
    else:
        incoming = dv.sum(-1)
        outgoing = torch.cat((incoming[..., 1:], dy.sum(-1, keepdim=True)),
                             -1)
    delta = torch.ones(P, device=dv.device, dtype=torch.float64)
    delta[-1] = 0.0
    return _rate_sums(incoming, outgoing, delta)


def _chain_setup(ctx, inputs, output) -> None:
    v, key, query, rate, semiring = inputs
    y, saved, ys = output
    ctx.semiring = semiring
    ctx.rank = _rank(key, query)
    ctx.has_key = key is not None
    ctx.has_query = query is not None
    ctx.dtypes = tuple(None if a is None else a.dtype
                       for a in (v, key, query))
    if semiring == "arctic":
        ctx.save_for_backward(None, None, None, rate, None, saved, ys)
    else:
        ctx.save_for_backward(v, key, query, rate,
                              y if semiring == "log" else None, saved, ys)


def _chain_backward(ctx, dy: torch.Tensor, *_):
    v, key, query, rate, y, saved, ys = ctx.saved_tensors
    need_rate = rate is not None and ctx.needs_input_grad[3]
    dtype = dy.dtype if ctx.semiring == "arctic" else saved.dtype
    dy = dy.contiguous().to(dtype)
    dv, dk, dq, drate = _level_chain_backward_op(
        dy, y, saved, ys, v, key, query, rate, ctx.semiring, ctx.rank,
        ctx.has_key, ctx.has_query, need_rate,
    )
    if need_rate and ctx.semiring != "arctic":
        drate = _chain_rate_gradient(ctx.semiring, v, dv, dy, query, dq)
    v_dtype, k_dtype, q_dtype = ctx.dtypes
    return (
        dv.to(v_dtype),
        dk.to(k_dtype) if ctx.has_key else None,
        dq.to(q_dtype) if ctx.has_query else None,
        drate.to(rate.dtype) if need_rate else None,
        None,
    )


_level_chain_op.register_autograd(_chain_backward, setup_context=_chain_setup)


def level_chain(
    v: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
) -> torch.Tensor:
    """All ``p`` pairs of a level of vector values in one op,
    ``(B, T, N, d_v, 1)``: the inner pairs exclusive, the last inclusive,
    and ``u_l = y_{l-1} (x) v_l`` formed inside the kernels.

    ``v`` is ``(B|1, T, N|1, p|1, d_v, 1)``, ``key`` and ``query`` are
    ``(B|1, T|1, N|1, p, R)`` or ``None``, ``rate`` is ``(N, p)`` or
    ``None``. Broadcast axes are expanded as views, and the gradients are
    summed back over them by autograd.
    """
    p = max(a.shape[3] for a in (v, key, query) if a is not None)
    if rate is not None:
        p = max(p, rate.shape[1])
    d_v = v.shape[-2]
    B, T, N = torch.broadcast_shapes(
        v.shape[:3],
        *(a.shape[:3] for a in (key, query) if a is not None),
        *(((1, 1, rate.shape[0]),) if rate is not None else ()),
    )
    flat = v.flatten(-2).expand(B, T, N, p, d_v)
    R = _rank(key, query)
    if key is not None:
        key = key.expand(B, T, N, p, R)
    if query is not None:
        query = query.expand(B, T, N, p, R)
    if rate is not None:
        rate = rate.expand(N, p).contiguous()
    y, _, _ = _level_chain_op(flat, key, query, rate, semiring)
    return y.unsqueeze(-1)
