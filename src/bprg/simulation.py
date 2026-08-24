import jax
import jax.numpy as jnp

from bprg.type import (
    AdjacencyMatrix,
    Array,
    Kernel,
    Key,
    Simplex,
    SimulationNum,
    ThresholdMeasure,
    TypeDomain,
    TypeMeasure,
    TypeNum,
    VertexNum,
    VertexThreshold,
    VertexType,
    VertexTypeIndex,
)

KEY = jax.random.key(0)


# ----------
# Graph simulation
# ----------
def simulate_graph(
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain,
    nv: VertexNum,
    ns: TypeNum,
    key: Key = KEY,
) -> tuple[VertexTypeIndex, VertexType, VertexThreshold, AdjacencyMatrix]:
    """
    Simulate a random graph. Return the vertex type indices and vertex types sampled from the type distribution,
    vertex thresholds from the threshold distribution, and adjacency matrix sampled from the kernel.

    Args:
        κ: kernel function
        μ: type measure
        η: threshold distribution
        nv: number of vertices
        ns: number of types
        S: type domain
        key: random key

    Returns:
        vsid: vertex type indices
        vs: vertex types
        vk: vertex thresholds
        A: adjacency matrix
    """
    s = jnp.linspace(S[0], S[1], ns)
    Δμ_s = μ(s)
    vsid, vs, vk, A = _simulate_graph(κ, η, s, Δμ_s, nv, key)
    return vsid, vs, vk, A


@jax.jit(static_argnames=("κ", "η", "nv"))
def _simulate_graph(
    κ: Kernel,
    η: ThresholdMeasure,
    s: VertexType,
    Δμ_s: Simplex,
    nv: VertexNum,
    key: Key = KEY,
) -> tuple[VertexTypeIndex, VertexType, VertexThreshold, AdjacencyMatrix]:
    """
    Simulate a random graph. Return the vertex type indices and vertex types sampled from the type distribution,
    vertex thresholds from the threshold distribution, and adjacency matrix sampled from the kernel.

    Args:
        κ: kernel function
        η: threshold distribution
        nv: number of vertices
        ns: number of types
        s: type grid
        Δμ_s: type measure
        S: type domain
        key: random key

    Returns:
        vsid: vertex type indices
        vs: vertex types
        vk: vertex thresholds
        A: adjacency matrix
    """
    # type
    ns = s.shape[0]

    # random keys
    key_s, key_v, key_u = jax.random.split(key, 3)

    # generate vertex types
    vsid = _sample_typeindex(Δμ_s, ns, nv, key_s)
    vs = s[vsid]

    # generate vertex thresholds
    η_sk = η(vs)
    vk = _sample_threshold(η_sk, key_v)

    # generate adjacency matrix
    A = _sample_adjacency_matrix(vs, κ, nv, key_u)

    return vsid, vs, vk, A


def _sample_typeindex(Δμ_s: Simplex, ns: TypeNum, nv: VertexNum, key: Key) -> VertexTypeIndex:
    """
    Assign types to vertices by random sampling from the type measure Δμ_s.

    Args:
        Δμ_s: probability distribution over types
        ns: number of types
        nv: number of vertices
        key: random key

    Returns:
        vsid: array of sampled vertex type indices
    """
    sid = jnp.arange(ns)
    vsid = jax.random.choice(key, a=sid, shape=(nv,), p=Δμ_s)
    vsid = jnp.sort(vsid)
    return vsid


def _sample_threshold(η_sk: Array[Simplex], key: Key) -> VertexThreshold:
    """
    Assign thresholds to vertices by random sampling from the threshold distribution η_sk.

    Args:
        η_sk: array of threshold distributions for each vertex
        key: random key

    Returns:
        vk: array of sampled vertex thresholds
    """
    vk = jax.random.categorical(key, jnp.log(η_sk), axis=1)
    return vk


def _sample_adjacency_matrix(vs: VertexType, κ: Kernel, nv: VertexNum, key: Key) -> AdjacencyMatrix:
    κ_xy = κ(vs[:, None], vs[None, :])
    P = (κ_xy / nv).clip(min=0.0, max=1.0)
    A = jax.random.bernoulli(key, p=P).astype(jnp.int32)
    return A


def _apportion_typeindex(Δμ_s: Simplex, ns: TypeNum, nv: VertexNum) -> tuple[VertexTypeIndex, Array[int]]:
    """
    Experimental. Assign types to vertices proportional to the type measure Δμ_s. Rounding error is absorbed by the
    first and last types to ensure exact total of nv vertices.

    Args:
        Δμ_s: probability distribution over types
        ns: number of types
        nv: number of vertices

    Returns:
        vsid: array of vertex type indices
        nv_s: array of number of vertices per type
    """
    sid = jnp.arange(ns)

    # Round normally
    nv_s = jnp.round(nv * Δμ_s).astype(jnp.int32)

    # Calculate error (rounding might not sum to exactly nv)
    r = nv - nv_s.sum()
    r1 = r // 2
    r2 = r - r1

    # Split the rounding error
    nv_s = nv_s.at[0].add(r1)
    nv_s = nv_s.at[-1].add(r2)

    # Create vertex type indices
    vsid = jnp.repeat(sid, nv_s, total_repeat_length=nv)
    return vsid, nv_s


def _apportion_threshold(η_k: Simplex, nv: VertexNum, key: Key) -> VertexThreshold:
    """Allocate vertices to thresholds using Largest Remainder Method.

    Ensures exact total of nv vertices while minimizing bias to any single threshold.
    """
    nk = jnp.arange(η_k.shape[0])

    # Floor all allocations
    nv_k_floor = jnp.floor(nv * η_k).astype(jnp.int32)

    # Calculate remainder
    remainder = nv - nv_k_floor.sum()

    # Calculate fractional parts
    fractional = nv * η_k - nv_k_floor

    # Create mask: thresholds with largest fractional parts get an extra vertex
    # Compute rank: double argsort inverts the sort to get rank of each element
    rank = jnp.argsort(jnp.argsort(fractional))
    mask = (rank >= η_k.shape[0] - remainder).astype(jnp.int32)

    # Final allocation
    nv_k = nv_k_floor + mask

    # Create vertex threshold indices
    vk = jnp.repeat(nk, nv_k, total_repeat_length=nv)
    vk = jax.random.permutation(key, vk)
    return vk


# ----------
# Boostrap percolation
# ----------
def simulate_percolation_final(vk: VertexThreshold, A: AdjacencyMatrix) -> VertexThreshold:
    """Simulate bootstrap percolation and return the final threshold vector.

    Args:
        vk: vertex thresholds
        A: adjacency matrix
    Returns:
        vk_final: threshold vector after the process stabilizes
    """
    return _simulate_percolation_final(vk, A)


def simulate_percolation_history(vk: VertexThreshold, A: AdjacencyMatrix) -> tuple[Array[VertexThreshold], Array[bool]]:
    """Simulate bootstrap percolation on a graph.

    Args:
        vk: vertex thresholds
        A: adjacency matrix

    Returns:
        vk_hist: vertex thresholds at each step, shape (n_steps+1, nv)
        ea_hist: boolean adjacency masks of the edges leaving that step's
            newly-infected vertices, shape (n_steps+1, nv, nv).
            ea_hist[t, i, j] is True iff vertex i became infected at step t
            and (i, j) is an edge of A — i.e. the edges that "caused"
            new infections downstream.
    """
    vk_cur = vk
    i_all = vk == 0
    i_new = i_all

    vk_hist = [vk_cur]
    ea_hist = [_active_edges(i_new, A)]

    while jnp.any(i_new):
        vk_new, i_all, i_new = _percolate_step(vk_cur, i_new, i_all, A)
        vk_hist.append(vk_new)
        ea_hist.append(_active_edges(i_new, A))
        vk_cur = vk_new

    return jnp.stack(vk_hist, axis=0), jnp.stack(ea_hist, axis=0)


@jax.jit()
def _simulate_percolation_final(vk: VertexThreshold, A: AdjacencyMatrix) -> VertexThreshold:
    """Simulate bootstrap percolation and return the final threshold vector.

    Args:
        vk: vertex thresholds
        A: adjacency matrix

    Returns:
        vk_final: threshold vector after the process stabilizes
    """
    vk_cur = vk
    i_all = vk == 0
    i_new = i_all
    state = (vk_cur, i_all, i_new)

    def cond_fun(state):
        _, _, i_new = state
        return jnp.any(i_new)

    def body_fun(state):
        vk_cur, i_all, i_new = state
        vk_new, i_all, i_new = _percolate_step(vk_cur, i_new, i_all, A)
        return vk_new, i_all, i_new

    vk_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, state)
    return vk_final


@jax.jit()
def _percolate_step(
    vk_cur: VertexThreshold,
    i_new: Array[bool],
    i_all: Array[bool],
    A: AdjacencyMatrix,
) -> tuple[VertexThreshold, Array[bool], Array[bool]]:
    vk_new = (vk_cur - i_new.astype(jnp.int32) @ A).clip(min=0)
    i_new = (vk_new == 0) & ~i_all
    i_all = i_all | i_new
    return vk_new, i_all, i_new


@jax.jit()
def _active_edges(i_new: Array[bool], A: AdjacencyMatrix) -> Array[bool]:
    """
    Mask of edges (i, j) where i was newly infected this round.
    """
    return A.astype(bool) & i_new[:, None]


# ----------
#  Monte Carlo
# ----------
def estimate_infection(
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain,
    nv: VertexNum,
    ns: TypeNum,
    nm: SimulationNum,
    key: Key = KEY,
) -> Array[float]:
    """
    Monte Carlo estimate of final proportion of infected vertices by type after the bootstrap percolation.

    Args:
        κ: kernel function
        μ: type measure
        η: threshold distribution
        S: type domain
        nv: number of vertices
        ns: number of types
        nm: number of simulations
        key: random key

    Returns:
        pi_s: final proportion of infected vertices by type, shape (ns,)
    """
    return _estimate_infection(κ, μ, η, S, nv, ns, nm, key)


def estimate_infection_apportioned_type(
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain,
    nv: VertexNum,
    ns: TypeNum,
    nm: SimulationNum,
    key: Key = KEY,
) -> Array:
    """
    Monte Carlo estimate of final proportion of infected vertices by type after the bootstrap percolation,
    using apportioned types to reduce variance.

    Args:
        κ: kernel function
        μ: type measure
        η: threshold distribution
        S: type domain
        nv: number of vertices
        ns: number of types
        nm: number of simulations
        key: random key

    Returns:
        pi_s: final proportion of infected vertices by type, shape (ns,)
    """
    return _estimate_infection_apportioned_type(κ, μ, η, S, nv, ns, nm, key)


@jax.jit(static_argnames=("κ", "μ", "η", "S", "nv", "ns", "nm"))
def _estimate_infection_apportioned_type(
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain,
    nv: VertexNum,
    ns: TypeNum,
    nm: SimulationNum,
    key: Key = KEY,
) -> Array:
    s = jnp.linspace(S[0], S[1], ns)
    Δμ_s = μ(s)

    vsid, nv_s = _apportion_typeindex(Δμ_s, ns, nv)
    vs = s[vsid]
    η_sk = η(vs)
    keys = jax.random.split(key, nm)

    def scan_body(ni_s_cum, sub_key):
        vk = _sample_threshold(η_sk, sub_key)
        A = _sample_adjacency_matrix(vs, κ, nv, sub_key)
        vk_final = _simulate_percolation_final(vk, A)
        ni_s = count_infected_by_typeindex(vsid, vk_final, ns)
        return ni_s_cum + ni_s, None

    ni_s_total, _ = jax.lax.scan(scan_body, jnp.zeros(ns), keys)
    pi_s = ni_s_total / nm / nv_s
    return pi_s


@jax.jit(static_argnames=("κ", "μ", "η", "S", "nv", "ns", "nm"))
def _estimate_infection(
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain,
    nv: VertexNum,
    ns: TypeNum,
    nm: SimulationNum,
    key: Key = KEY,
) -> Array:
    # Precompute type grid and measure once
    s = jnp.linspace(S[0], S[1], ns)
    Δμ_s = μ(s)

    keys = jax.random.split(key, nm)

    def scan_body(ni_s_cum, sub_key):
        vsid, _, vk, A = _simulate_graph(κ, η, s, Δμ_s, nv, sub_key)
        vk_final = _simulate_percolation_final(vk, A)
        ni_s = count_infected_by_typeindex(vsid, vk_final, ns)
        return ni_s_cum + ni_s, None

    ni_s_total, _ = jax.lax.scan(scan_body, jnp.zeros(ns), keys)
    pi_s = ni_s_total / nm / nv / Δμ_s
    return pi_s


# ----------
# Counting infected vertices by type
# ----------
def count_infected_by_type(
    vs: VertexType,
    vk: VertexThreshold,
    S: TypeDomain,
    ns: TypeNum,
) -> Array:
    """
    Count infected vertices (vk == 0) by vertex type.

    Args:
        vs: vertex type vector
        vk: vertex threshold vector
        S: type domain
        ns: number of types

    Returns:
        n_si: infected count per type, shape (ns,)
    """
    s = jnp.linspace(S[0], S[1], ns)
    type_idx = jnp.searchsorted(s, vs)
    infected = (vk == 0).astype(jnp.int32)
    n_si = jnp.bincount(type_idx, weights=infected, length=ns)
    return n_si


def count_infected_by_typeindex(
    vsid: VertexTypeIndex,
    vk: VertexThreshold,
    ns: TypeNum,
) -> Array:
    """
    Count infected vertices (vk == 0) by vertex type index.

    Args:
        vsid: vertex type indices
        vk: vertex threshold vector
        ns: number of types

    Returns:
        n_si: infected count per type, shape (ns,)
    """
    infected = (vk == 0).astype(jnp.int32)
    n_si = jnp.bincount(vsid, weights=infected, length=ns)
    return n_si


# if __name__ == "__main__":
#     import plotly.graph_objects as go

#     from bprg.preset import preset_lips
#     from bprg.solver import FixpointSolver, NeuralNetwork

#     key = KEY
#     κ = preset_lips.κ
#     μ = preset_lips.μ
#     η = preset_lips.η
#     S = preset_lips.S
#     nv = 3000
#     ns = 1000
#     nx = 1000
#     ny = 1000

#     s = jnp.linspace(S[0], S[1], ns)
#     Δμ_s = μ(s)

#     fnn = NeuralNetwork()
#     solver = FixpointSolver(fnn=fnn, κ=κ, μ=μ, η=η, S=S, nx=nx, ny=ny)
#     result = solver.solve()
#     f = result["f"]
#     Ψf = result["Ψf"]

#     s = jnp.linspace(S[0], S[1], ns)
#     f_s = f(s)
#     Ψf_s = Ψf(s)

#     go.Figure(
#         data=[go.Scatter(x=s, y=f_s, mode="lines", name="f(s)"), go.Scatter(x=s, y=Ψf_s, mode="lines", name="Ψf(s)")]
#     )

#     # type_idx = jnp.searchsorted(s, vs)
#     # infected = (vk == 0).astype(jnp.int32)
#     # c_si = jnp.bincount(type_idx, weights=infected, length=ns)
#     # Δμ_s = μ(s)
#     # c_si / nv / Δμ_s

#     # idx = jnp.arange(nv)
#     # idx = jax.random.permutation(jax.random.key(0), idx)
#     # nk_list = (nv * jnp.array([0.1, 0, 0.9])).astype(int)
#     # nk_idx = nk_list.cumsum()

#     # kv = jnp.zeros(nv)
#     # for nk in nk_idx:
#     #     kv_incre = kv[idx[nk:]] + 1
#     #     kv = kv.at[idx[nk:]].set(kv_incre)

#     # jnp.unique_counts(kv)

#     fmc_sample_both = estimate_infection(κ, μ, η, S, nv, ns, nm=100)
#     fmc_gridtype = _estimate_infection_gridtype(κ, μ, η, S, nv, ns, nm=100)
#     fmc_gridboth = _estimate_infection_gridboth(κ, μ, η, S, nv, ns, nm=100)

#     go.Figure(
#         data=[
#             go.Scatter(x=s, y=fmc_sample_both, mode="markers", name="fmc(s)"),
#             go.Scatter(x=s, y=fmc_gridtype, mode="markers", name="fmc_typegrid(s)"),
#             go.Scatter(x=s, y=fmc_gridboth, mode="markers", name="fmc_bothgrid(s)"),
#             go.Scatter(x=s, y=f_s, mode="lines", name="Ψf(s)"),
#         ]
#     )
