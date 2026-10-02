"""Triton kernels for one scan of a LISS level, fused with its factors.

One pair ``l`` of a level reads the level input ``u_s`` (the previous
contraction times the values, ``(B, T, N, D)`` with ``D = d_v * w``), the
key and query features ``k_s, q_t`` (``(B, T, N, R)``) and a decay rate per
head, and returns the contraction

    y_t = <q_t, S_{t-1}>   (inner pairs, ``inclusive=False``)
    y_t = <q_t, S_t>       (the last pair, ``inclusive=True``)
    S_t = lambda (x) S_{t-1} (+) k_t (x) u_t       (an R x D state)

in the reals (``+``, ``*``), the log semiring (logsumexp, ``+``) or the
arctic semiring (``max``, ``+``). The ``R x D`` state lives in registers
only, so a level keeps ``(B, T, N, D)`` tensors where the PyTorch path
keeps ``(B, T, N, R, D)`` ones.

Every scan is the three-phase chunked scan of ``lru-torch``: each program
scans one chunk of time and stores its end state (phase 1), one program
per sequence combines the chunk states in order (phase 2), and each chunk
is rescanned from its incoming state (phase 3). The decay across a chunk
is exact (``exp(-rate * length)``, an offset in the log domain), so the
order of rounding is ``chunk + chunks`` steps deep instead of ``T``.

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
# the semiring arithmetic of a state tile
# --------------------------------------------------------------------------
@triton.jit
def _outer(k, u, SR: tl.constexpr, HAS_K: tl.constexpr, BR: tl.constexpr):
    """The scan input ``k_r (x) u_d`` as a ``(BR, BD)`` tile."""
    if HAS_K:
        if SR == 0:
            return k[:, None] * u[None, :]
        return k[:, None] + u[None, :]
    return tl.broadcast_to(u[None, :], (BR, u.shape[0]))


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
def _contract(S, q, SR: tl.constexpr, HAS_Q: tl.constexpr):
    """``<q, S>``: the semiring sum over the features, ``(BD,)``."""
    if not HAS_Q:
        return tl.sum(S, axis=0)
    if SR == 0:
        return tl.sum(q[:, None] * S, axis=0)
    z = q[:, None] + S
    high = tl.max(z, axis=0)
    if SR == 1:
        return high + tl.log(tl.sum(tl.exp(z - high[None, :]), axis=0))
    return high


@triton.jit
def _plane(bn, B, N, C, R, D):
    """The size of one ``(B * N, C, R, D)`` plane of a chunk buffer, in
    int64 like the program index ``bn`` (the scalars are int32)."""
    return (bn * 0 + B) * N * C * R * D


# --------------------------------------------------------------------------
# forward: phases 1 and 3, and the forward rescan of the backward (dq)
# --------------------------------------------------------------------------
@triton.jit
def _forward_kernel(
    u_ptr, k_ptr, q_ptr, rate_ptr, y_ptr, dy_ptr, out_ptr, start_ptr,
    end_ptr, bits_ptr,
    B, T, N, D, R, C,
    su_b, su_t, su_n, su_d,
    sk_b, sk_t, sk_n, sk_r,
    sq_b, sq_t, sq_n, sq_r,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BR: tl.constexpr,
    BD: tl.constexpr, SR: tl.constexpr, INCLUSIVE: tl.constexpr,
    HAS_K: tl.constexpr, HAS_Q: tl.constexpr, HAS_RATE: tl.constexpr,
    EMPTY: tl.constexpr, QPAD: tl.constexpr, ACC: tl.constexpr,
):
    """MODE 0: the end state of the chunk (phase 1). MODE 1: the outputs
    ``y`` from the chunk's start state (phase 3), and in the arctic
    semiring the argmax bits. MODE 2: ``dq = d<q, S>/dq`` from the start
    state (backward, reals and log)."""
    bn = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    b = bn // N
    n = bn % N
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    r_mask = r < R
    d_mask = d < D
    rd_mask = r_mask[:, None] & d_mask[None, :]
    state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
    if HAS_RATE:
        rate = tl.load(rate_ptr + n).to(ACC)
    else:
        rate = tl.zeros((), ACC)
    lam = tl.exp(-rate)
    if MODE == 0:
        S = tl.full((BR, BD), EMPTY, ACC)
    else:
        S = tl.load(start_ptr + state, mask=rd_mask, other=EMPTY).to(ACC)
    if SR == 2:
        took = tl.zeros((BR, BD), tl.int64)
        path = tl.zeros((BR, BD), tl.int64)
    u_base = b * su_b + n * su_n + d * su_d
    k_base = b * sk_b + n * sk_n + r * sk_r
    q_base = b * sq_b + n * sq_n + r * sq_r
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for tau in range(0, length):
        t = (t0 + tau).to(tl.int64)
        u = tl.load(u_ptr + u_base + t * su_t, mask=d_mask, other=0.0)
        k = tl.zeros((BR,), ACC)
        if HAS_K:
            k = tl.load(k_ptr + k_base + t * sk_t, mask=r_mask,
                        other=EMPTY).to(ACC)
        x = _outer(k, u.to(ACC), SR, HAS_K, BR)
        q = tl.zeros((BR,), ACC)
        if HAS_Q:
            q = tl.load(q_ptr + q_base + t * sq_t, mask=r_mask,
                        other=QPAD).to(ACC)
        out = ((b * T + t) * N + n) * D + d
        if MODE != 0 and not INCLUSIVE:
            read = S
        S_new = _accumulate(S, x, rate, lam, SR)
        if SR == 2 and MODE == 1:
            took = took | ((x >= S - rate).to(tl.int64) << tau.to(tl.int64))
        S = S_new
        if MODE != 0 and INCLUSIVE:
            read = S
        if MODE == 1:
            y = _contract(read, q, SR, HAS_Q)
            tl.store(y_ptr + out, y, mask=d_mask)
            if SR == 2:
                if HAS_Q:
                    best = tl.argmax(q[:, None] + read, axis=0)
                    hit = r[:, None] == best[None, :]
                else:
                    hit = tl.full((BR, BD), 1, tl.int1)
                path = path | (hit.to(tl.int64) << tau.to(tl.int64))
        if MODE == 2:
            dy = tl.load(dy_ptr + out, mask=d_mask, other=0.0).to(ACC)
            if SR == 0:
                dq = tl.sum(read * dy[None, :], axis=1)
            else:
                y = tl.load(y_ptr + out, mask=d_mask, other=0.0).to(ACC)
                weight = tl.exp(tl.minimum(q[:, None] + read - y[None, :], 0.0))
                weight = tl.where(dy[None, :] != 0, weight * dy[None, :], 0.0)
                dq = tl.sum(weight, axis=1)
            dq_off = (((j * B + b) * T + t) * N + n) * R + r
            tl.store(out_ptr + dq_off, dq, mask=r_mask)
    if MODE == 0:
        tl.store(end_ptr + state, S, mask=rd_mask)
    if SR == 2 and MODE == 1:
        plane = _plane(bn, B, N, C, R, D)
        tl.store(bits_ptr + state, took, mask=rd_mask)
        tl.store(bits_ptr + plane + state, path, mask=rd_mask)


@triton.jit
def _forward_prefix_kernel(
    end_ptr, start_ptr, rate_ptr,
    N, D, R, C,
    CHUNK: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
    SR: tl.constexpr, HAS_RATE: tl.constexpr, EMPTY: tl.constexpr,
    ACC: tl.constexpr,
):
    """Phase 2: the state entering every chunk, from the chunks' end
    states, in order. Every chunk but the last is full."""
    bn = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    n = bn % N
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    rd_mask = (r < R)[:, None] & (d < D)[None, :]
    if HAS_RATE:
        rate = tl.load(rate_ptr + n).to(ACC) * CHUNK
    else:
        rate = tl.zeros((), ACC)
    lam = tl.exp(-rate)
    P = tl.full((BR, BD), EMPTY, ACC)
    for c in range(0, C):
        state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
        tl.store(start_ptr + state, P, mask=rd_mask)
        E = tl.load(end_ptr + state, mask=rd_mask, other=EMPTY).to(ACC)
        P = _accumulate(P, E, rate, lam, SR)


# --------------------------------------------------------------------------
# backward, reals and log: the reverse scan of the adjoint
# --------------------------------------------------------------------------
@triton.jit
def _reverse_kernel(
    dy_ptr, y_ptr, u_ptr, k_ptr, q_ptr, rate_ptr, du_ptr, dk_ptr,
    start_ptr, end_ptr,
    B, T, N, D, R, C,
    su_b, su_t, su_n, su_d,
    sk_b, sk_t, sk_n, sk_r,
    sq_b, sq_t, sq_n, sq_r,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BR: tl.constexpr,
    BD: tl.constexpr, SR: tl.constexpr, INCLUSIVE: tl.constexpr,
    HAS_K: tl.constexpr, HAS_Q: tl.constexpr, HAS_RATE: tl.constexpr,
    ZERO: tl.constexpr, QPAD: tl.constexpr, ACC: tl.constexpr,
):
    """The adjoint ``H_s = sum_{t >= s} lambda^{t-s} a_t`` backwards over
    one chunk: ``a_t = q_t dy_t^T`` (reals), or ``exp(q_t - y_t) dy_t``
    held as ``e^M H`` (log). MODE 0 stores the chunk's own sum (phase 1);
    MODE 1 starts from the sum entering it from later chunks and writes
    ``du`` and ``dk`` (phase 3)."""
    bn = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    b = bn // N
    n = bn % N
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    r_mask = r < R
    d_mask = d < D
    rd_mask = r_mask[:, None] & d_mask[None, :]
    state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
    plane = _plane(bn, B, N, C, R, D)
    if HAS_RATE:
        rate = tl.load(rate_ptr + n).to(ACC)
    else:
        rate = tl.zeros((), ACC)
    lam = tl.exp(-rate)
    H = tl.zeros((BR, BD), ACC)
    M = tl.full((BR, BD), ZERO, ACC)
    if MODE == 1:
        H = tl.load(start_ptr + state, mask=rd_mask, other=0.0).to(ACC)
        if SR == 1:
            M = tl.load(start_ptr + plane + state, mask=rd_mask,
                        other=ZERO).to(ACC)
    u_base = b * su_b + n * su_n + d * su_d
    k_base = b * sk_b + n * sk_n + r * sk_r
    q_base = b * sq_b + n * sq_n + r * sq_r
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for i in range(0, length):
        t = (t0 + length - 1 - i).to(tl.int64)
        out = ((b * T + t) * N + n) * D + d
        dy = tl.load(dy_ptr + out, mask=d_mask, other=0.0).to(ACC)
        q = tl.zeros((BR,), ACC)
        if HAS_Q:
            q = tl.load(q_ptr + q_base + t * sq_t, mask=r_mask,
                        other=QPAD).to(ACC)
        if MODE == 1:
            u = tl.load(u_ptr + u_base + t * su_t, mask=d_mask,
                        other=0.0).to(ACC)
            k = tl.zeros((BR,), ACC)
            if HAS_K:
                k = tl.load(k_ptr + k_base + t * sk_t, mask=r_mask,
                            other=0.0).to(ACC)
        if MODE == 1 and not INCLUSIVE:
            H_read = H
            M_read = M
        # add a_t
        if SR == 0:
            if HAS_Q:
                H = lam * H + q[:, None] * dy[None, :]
            else:
                H = lam * H + dy[None, :]
        else:
            y = tl.load(y_ptr + out, mask=d_mask, other=0.0).to(ACC)
            # log|a_t|; -inf where dy = 0, which adds nothing
            level = q[:, None] - y[None, :] + tl.log(tl.abs(dy))[None, :]
            carried = M - rate
            high = tl.maximum(carried, level)
            sign = tl.where(dy > 0, 1.0, -1.0)
            H = H * tl.exp(carried - high) \
                + sign[None, :] * tl.exp(level - high)
            M = high
        if MODE == 1:
            if INCLUSIVE:
                H_read = H
                M_read = M
            if SR == 0:
                if HAS_K:
                    du = tl.sum(k[:, None] * H_read, axis=0)
                else:
                    du = tl.sum(H_read, axis=0)
                dk = tl.sum(H_read * u[None, :], axis=1)
            else:
                e = M_read + u[None, :]
                if HAS_K:
                    e = e + k[:, None]
                g = tl.exp(tl.minimum(e, 80.0)) * H_read
                g = tl.where(rd_mask, g, 0.0)
                du = tl.sum(g, axis=0)
                dk = tl.sum(g, axis=1)
            tl.store(du_ptr + out, du, mask=d_mask)
            if HAS_K:
                dk_off = (((j * B + b) * T + t) * N + n) * R + r
                tl.store(dk_ptr + dk_off, dk, mask=r_mask)
    if MODE == 0:
        tl.store(end_ptr + state, H, mask=rd_mask)
        if SR == 1:
            tl.store(end_ptr + plane + state, M, mask=rd_mask)


@triton.jit
def _reverse_prefix_kernel(
    end_ptr, start_ptr, rate_ptr,
    B, T, N, D, R, C,
    CHUNK: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr,
    SR: tl.constexpr, HAS_RATE: tl.constexpr, ZERO: tl.constexpr,
    ACC: tl.constexpr,
):
    """Phase 2 backwards: the adjoint entering every chunk from the later
    ones, ``P_{c-1} = E_c (+) lambda^{len(c)} P_c``."""
    bn = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    n = bn % N
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    rd_mask = (r < R)[:, None] & (d < D)[None, :]
    plane = _plane(bn, B, N, C, R, D)
    if HAS_RATE:
        rate = tl.load(rate_ptr + n).to(ACC)
    else:
        rate = tl.zeros((), ACC)
    H = tl.zeros((BR, BD), ACC)
    M = tl.full((BR, BD), ZERO, ACC)
    for i in range(0, C):
        c = C - 1 - i
        state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
        tl.store(start_ptr + state, H, mask=rd_mask)
        if SR == 1:
            tl.store(start_ptr + plane + state, M, mask=rd_mask)
        length = tl.minimum(CHUNK, T - c * CHUNK)
        E = tl.load(end_ptr + state, mask=rd_mask, other=0.0).to(ACC)
        if SR == 0:
            H = tl.exp(-rate * length) * H + E
        else:
            E_M = tl.load(end_ptr + plane + state, mask=rd_mask,
                          other=ZERO).to(ACC)
            carried = M - rate * length
            high = tl.maximum(carried, E_M)
            H = H * tl.exp(carried - high) + E * tl.exp(E_M - high)
            M = high


# --------------------------------------------------------------------------
# backward, arctic: the reverse scan gated by the argmax bits
# --------------------------------------------------------------------------
@triton.jit
def _arctic_reverse_kernel(
    dy_ptr, bits_ptr, du_ptr, dk_ptr, dq_ptr, start_ptr, end_ptr,
    B, T, N, D, R, C,
    MODE: tl.constexpr, CHUNK: tl.constexpr, BR: tl.constexpr,
    BD: tl.constexpr, INCLUSIVE: tl.constexpr, HAS_K: tl.constexpr,
    HAS_Q: tl.constexpr, ACC: tl.constexpr,
):
    """The gradient ``K`` reaching the state from later steps, backwards
    over one chunk: a step that took its new value passes ``K`` to its
    input, one that carried the old state passes it on. MODE 0 stores the
    chunk's own ``K`` and whether it passes everything through (phase 1);
    MODE 1 starts from the ``K`` entering it and writes ``du``, ``dk`` and
    ``dq`` (phase 3)."""
    bn = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    j = tl.program_id(2)
    b = bn // N
    n = bn % N
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    r_mask = r < R
    d_mask = d < D
    rd_mask = r_mask[:, None] & d_mask[None, :]
    state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
    plane = _plane(bn, B, N, C, R, D)
    took = tl.load(bits_ptr + state, mask=rd_mask, other=0)
    path = tl.load(bits_ptr + plane + state, mask=rd_mask, other=0)
    if MODE == 1:
        K = tl.load(start_ptr + state, mask=rd_mask, other=0.0).to(ACC)
    else:
        K = tl.zeros((BR, BD), ACC)
    t0 = c * CHUNK
    length = tl.minimum(CHUNK, T - t0)
    for i in range(0, length):
        tau = length - 1 - i
        t = (t0 + tau).to(tl.int64)
        out = ((b * T + t) * N + n) * D + d
        dy = tl.load(dy_ptr + out, mask=d_mask, other=0.0).to(ACC)
        new = ((took >> tau) & 1).to(ACC)
        hit = ((path >> tau) & 1).to(ACC) * dy[None, :]
        if INCLUSIVE:
            G = K + hit
        else:
            G = K
        dx = new * G
        K = G - dx
        if not INCLUSIVE:
            K = K + hit
        if MODE == 1:
            tl.store(du_ptr + out, tl.sum(dx, axis=0), mask=d_mask)
            dk_off = (((j * B + b) * T + t) * N + n) * R + r
            if HAS_K:
                tl.store(dk_ptr + dk_off, tl.sum(dx, axis=1), mask=r_mask)
            if HAS_Q:
                tl.store(dq_ptr + dk_off, tl.sum(hit, axis=1), mask=r_mask)
    if MODE == 0:
        tl.store(end_ptr + state, K, mask=rd_mask)
        tl.store(end_ptr + plane + state, (took == 0).to(ACC), mask=rd_mask)


@triton.jit
def _arctic_reverse_prefix_kernel(
    end_ptr, start_ptr,
    B, N, D, R, C,
    BR: tl.constexpr, BD: tl.constexpr, ACC: tl.constexpr,
):
    """Phase 2 backwards: ``K_{c-1} = local_c + through_c * K_c``."""
    bn = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1)
    r = tl.arange(0, BR)
    d = j * BD + tl.arange(0, BD)
    rd_mask = (r < R)[:, None] & (d < D)[None, :]
    plane = _plane(bn, B, N, C, R, D)
    K = tl.zeros((BR, BD), ACC)
    for i in range(0, C):
        c = C - 1 - i
        state = ((bn * C + c) * R + r[:, None]) * D + d[None, :]
        tl.store(start_ptr + state, K, mask=rd_mask)
        local = tl.load(end_ptr + state, mask=rd_mask, other=0.0).to(ACC)
        through = tl.load(end_ptr + plane + state, mask=rd_mask,
                          other=0.0).to(ACC)
        K = local + through * K


# --------------------------------------------------------------------------
# launching
# --------------------------------------------------------------------------
def _chunk(T: int, semiring: str) -> int:
    if semiring == "arctic":
        return ARCTIC_CHUNK
    if T <= 8192:
        return 64
    return 128 if T <= 32768 else 256


def _blocks(R: int, D: int) -> tuple[int, int, int]:
    """``(BR, BD, num_warps)``: the whole ``R`` in one tile, ``D`` split
    so that a state tile holds at most 2048 entries. One warp per program
    is fastest up to 512 entries (measured on a 3090 for R = 1..128); the
    larger tiles of the log and arctic scans spill with fewer than four."""
    BR = triton.next_power_of_2(R)
    BD = min(triton.next_power_of_2(D), max(1, 2048 // BR))
    size = BR * BD
    warps = 1 if size <= 512 else 2 if size <= 1024 else 4
    return BR, BD, warps


def _rank(key: torch.Tensor | None, query: torch.Tensor | None) -> int:
    for a in (key, query):
        if a is not None:
            return a.shape[-1]
    return 1


def _zero(dtype: torch.dtype) -> float:
    return torch.finfo(dtype).min / 8


def _pads(SR: int, zero: float) -> tuple[float, float]:
    """The empty state and the query of a padded feature: ``(0, 0)`` in
    the reals, ``(zero, -inf)`` in the log domain."""
    return (0.0, 0.0) if SR == 0 else (zero, -math.inf)


def _acc(*tensors: torch.Tensor | None) -> tuple[torch.dtype, object]:
    dtype = torch.float32
    for tensor in tensors:
        if tensor is not None and tensor.dtype == torch.float64:
            dtype = torch.float64
    return dtype, (tl.float64 if dtype == torch.float64 else tl.float32)


def _strides(a: torch.Tensor | None) -> tuple[int, int, int, int]:
    return (0, 0, 0, 0) if a is None else tuple(a.stride())  # type: ignore


def _launch_forward(
    u: torch.Tensor,
    key: torch.Tensor | None,
    query: torch.Tensor | None,
    rate: torch.Tensor | None,
    semiring: str,
    inclusive: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, N, D = u.shape
    R = _rank(key, query)
    dtype, ACC = _acc(u, key, query)
    SR = SEMIRINGS[semiring]
    CHUNK = _chunk(T, semiring)
    C = triton.cdiv(T, CHUNK)
    BR, BD, warps = _blocks(R, D)
    NDB = triton.cdiv(D, BD)
    zero = _zero(dtype)
    y = torch.empty((B, T, N, D), device=u.device, dtype=dtype)
    start = torch.full((B * N, C, R, D), 0.0 if SR == 0 else zero,
                       device=u.device, dtype=dtype)
    bits = (torch.empty((2, B * N, C, R, D), device=u.device,
                        dtype=torch.int64)
            if SR == 2 else start)
    dummy = u
    empty, qpad = _pads(SR, zero)
    common = dict(
        CHUNK=CHUNK, BR=BR, BD=BD, SR=SR, INCLUSIVE=inclusive,
        HAS_K=key is not None, HAS_Q=query is not None,
        HAS_RATE=rate is not None, EMPTY=empty, QPAD=qpad, ACC=ACC,
        num_warps=warps,
    )
    args = (
        B, T, N, D, R, C, *_strides(u), *_strides(key), *_strides(query),
    )
    k = dummy if key is None else key
    q = dummy if query is None else query
    rt = dummy if rate is None else rate
    if C > 1:
        end = torch.empty_like(start)
        _forward_kernel[(B * N, C, NDB)](
            u, k, q, rt, y, dummy, dummy, start, end, bits, *args,
            MODE=0, **common,
        )
        _forward_prefix_kernel[(B * N, NDB)](
            end, start, rt, N, D, R, C,
            CHUNK=CHUNK, BR=BR, BD=BD, SR=SR, HAS_RATE=rate is not None,
            EMPTY=empty, ACC=ACC, num_warps=warps,
        )
    _forward_kernel[(B * N, C, NDB)](
        u, k, q, rt, y, dummy, dummy, start, dummy, bits, *args,
        MODE=1, **common,
    )
    return y, (bits if SR == 2 else start)


def _launch_backward(
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
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    B, T, N, D = u.shape
    R = _rank(key, query)
    dtype, ACC = _acc(u, key, query)
    SR = SEMIRINGS[semiring]
    CHUNK = _chunk(T, semiring)
    C = triton.cdiv(T, CHUNK)
    BR, BD, warps = _blocks(R, D)
    NDB = triton.cdiv(D, BD)
    zero = _zero(dtype)
    device = u.device
    dy = dy.to(dtype).contiguous()
    du = torch.empty((B, T, N, D), device=device, dtype=dtype)
    dk = (torch.empty((NDB, B, T, N, R), device=device, dtype=dtype)
          if key is not None else None)
    dq = (torch.empty((NDB, B, T, N, R), device=device, dtype=dtype)
          if need_dq else None)
    dummy = du
    k = dummy if key is None else key
    q = dummy if query is None else query
    rt = dummy if rate is None else rate
    grid = (B * N, C, NDB)
    flags = dict(
        CHUNK=CHUNK, BR=BR, BD=BD, INCLUSIVE=inclusive,
        HAS_K=key is not None, num_warps=warps,
    )
    if SR == 2:
        start = torch.zeros((B * N, C, R, D), device=device, dtype=dtype)
        if C > 1:
            end = torch.empty((2, B * N, C, R, D), device=device,
                              dtype=dtype)
            _arctic_reverse_kernel[grid](
                dy, saved, dummy, dummy, dummy, start, end,
                B, T, N, D, R, C, MODE=0, HAS_Q=need_dq, ACC=ACC, **flags,
            )
            _arctic_reverse_prefix_kernel[(B * N, NDB)](
                end, start, B, N, D, R, C, BR=BR, BD=BD, ACC=ACC,
                num_warps=warps,
            )
        _arctic_reverse_kernel[grid](
            dy, saved, du, dummy if dk is None else dk,
            dummy if dq is None else dq, start, dummy,
            B, T, N, D, R, C, MODE=1, HAS_Q=need_dq, ACC=ACC, **flags,
        )
    else:
        empty, qpad = _pads(SR, zero)
        flags.update(
            SR=SR, HAS_Q=query is not None, HAS_RATE=rate is not None,
            QPAD=qpad, ACC=ACC,
        )
        args = (
            B, T, N, D, R, C, *_strides(u), *_strides(key), *_strides(query),
        )
        yy = dummy if y is None else y
        if dq is not None:
            _forward_kernel[grid](
                u, k, q, rt, yy, dy, dq, saved, dummy, dummy, *args,
                MODE=2, EMPTY=empty, **flags,
            )
        planes = 1 if SR == 0 else 2
        start = torch.zeros((planes, B * N, C, R, D), device=device,
                            dtype=dtype)
        if SR == 1:
            start[1] = zero
        if C > 1:
            end = torch.empty_like(start)
            _reverse_kernel[grid](
                dy, yy, u, k, q, rt, dummy, dummy, dummy, end, *args,
                MODE=0, ZERO=zero, **flags,
            )
            _reverse_prefix_kernel[(B * N, NDB)](
                end, start, rt, B, T, N, D, R, C,
                CHUNK=CHUNK, BR=BR, BD=BD, SR=SR, HAS_RATE=rate is not None,
                ZERO=zero, ACC=ACC, num_warps=warps,
            )
        _reverse_kernel[grid](
            dy, yy, u, k, q, rt, du, dummy if dk is None else dk, start,
            dummy, *args, MODE=1, ZERO=zero, **flags,
        )
    return (
        du,
        None if dk is None else dk.sum(0),
        None if dq is None else dq.sum(0),
    )


# --------------------------------------------------------------------------
# the custom ops
# --------------------------------------------------------------------------
def _empty(like: torch.Tensor) -> torch.Tensor:
    return like.new_empty((0,))


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
        return _launch_forward(u, key, query, rate, semiring, inclusive)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(du, dk, dq)``, with an empty tensor for a gradient not asked
    for."""
    with torch.cuda.device(u.device):
        du, dk, dq = _launch_backward(
            dy, y, saved, u, key, query, rate, semiring, inclusive, need_dq,
        )
    return (
        du.to(u.dtype),
        _empty(u) if dk is None else dk.to(key.dtype),  # type: ignore
        _empty(u) if dq is None else dq.to(
            query.dtype if query is not None else u.dtype),
    )


@_level_scan_backward_op.register_fake
def _(dy, y, saved, u, key, query, rate, semiring, inclusive, need_dq):
    B, T, N, D = u.shape
    R = _rank(key, query)
    du = torch.empty_like(u, memory_format=torch.contiguous_format)
    dk = (_empty(u) if key is None
          else key.new_empty((B, T, N, R)))
    dq = (_empty(u) if not need_dq
          else u.new_empty((B, T, N, R),
                           dtype=query.dtype if query is not None
                           else u.dtype))
    return du, dk, dq


def rate_gradient(
    semiring: str,
    inclusive: bool,
    u: torch.Tensor,
    du: torch.Tensor,
    dy: torch.Tensor,
    query: torch.Tensor | None,
    dq: torch.Tensor | None,
) -> torch.Tensor:
    """``dL/d rate`` per head, ``(N,)``.

    The decay weighs a pair ``(t, s)`` by ``exp(-rate (t - delta - s))``, so
    ``dL/d rate = -sum_{t,s} (t - delta - s) c(t, s)`` over the pair
    contributions ``c``. Summed over ``s`` they are ``dy_t . y_t`` (reals)
    or ``dy_t`` (log domain), over ``t`` they are ``du_s . u_s`` or
    ``du_s``, and in the reals ``dy_t . y_t = q_t . dq_t``. The sums over
    time run in float64. Accurate to the error of the ``O(T)`` terms, the
    same as autograd through the PyTorch path's ``exp(rate * t)``.
    """
    T = u.shape[1]
    if semiring == "reals":
        assert dq is not None
        outgoing = (dq * query).sum(-1) if query is not None else dq[..., 0]
        incoming = (du * u).sum(-1)
    else:
        outgoing = dy.sum(-1)
        incoming = du.sum(-1)
    t = torch.arange(T, device=u.device, dtype=torch.float64).view(1, -1, 1)
    # t = 0 of an inner pair reads the empty state: no pair ends there.
    end = (t - (0.0 if inclusive else 1.0)).clamp_min(0.0)
    return (
        (t * incoming.double()).sum((0, 1))
        - (end * outgoing.double()).sum((0, 1))
    )


def _setup(ctx, inputs, output) -> None:
    u, key, query, rate, semiring, inclusive = inputs
    y, saved = output
    ctx.semiring = semiring
    ctx.inclusive = inclusive
    ctx.save_for_backward(
        u, key, query, rate, y if semiring == "log" else None, saved,
    )


def _backward(ctx, dy: torch.Tensor, _: torch.Tensor):
    u, key, query, rate, y, saved = ctx.saved_tensors
    need_rate = rate is not None and ctx.needs_input_grad[3]
    need_dq = query is not None or (ctx.semiring == "reals" and need_rate)
    du, dk, dq = _level_scan_backward_op(
        dy.contiguous(), y, saved, u, key, query, rate, ctx.semiring,
        ctx.inclusive, need_dq,
    )
    drate = None
    if need_rate:
        drate = rate_gradient(
            ctx.semiring, ctx.inclusive, u, du, dy, query, dq,
        ).to(rate.dtype)
    return (
        du,
        dk if key is not None else None,
        dq if query is not None else None,
        drate,
        None,
        None,
    )


_level_scan_op.register_autograd(_backward, setup_context=_setup)


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
