import jax
import jax.numpy as jnp
from jax import Array

from bprg.type import (
    Function,
    IntegralNum,
    Kernel,
    ThresholdMeasure,
    TypeDomain,
    TypeMeasure,
    VertexType,
)

INTEGRAL_NUM = 1000
TYPE_DOMAIN = (0.0, 1.0)


def Λ(
    f: Function,
    x: VertexType,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    r"""
    Compute the operator
    $$
    Λ_κ\[f\](x) = ∫_{y ∈ S} κ(y, x) f(y) \mathrm{d}μ(y)
    $$
    for each $x ∈ S$ in the input array.

    Args:
        f: A function that takes an array of vertices and returns an array of values.
        x: An array of vertices at which to evaluate the operator.
        κ: A kernel function that takes two arrays of vertices and returns an array of values.
        μ: A measure function that takes an array of vertices and returns an array of measure values.
        S: A tuple defining the domain of integration (default is (0.0, 1.0)).
        ny: The number of points to use for numerical integration (default is 1000).

    Returns:
        An array of values representing $Λ_κ[f](x)$ for each $x$ in the input array.
    """
    return _Λ_jit(f, x, κ, μ, S, ny)


def Πk(
    f: Function,
    x: VertexType,
    k: int,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    r"""
    Compute the operator
    $$
    Π_κ^k\[f\](x) = \frac{(Λ_κ\[f\](x))^k}{k!} e^{-Λ_κ\[f\](x)}
    $$
    for each $x$ in the input array.

    Args:
        f: A function that takes an array of vertices and returns an array of values.
        x: An array of vertices at which to evaluate the operator.
        k: The threshold of vertices.
        κ: A kernel function that takes two arrays of vertices and returns an array of values.
        μ: A measure function that takes an array of vertices and returns an array of measure values.
        S: A tuple defining the domain of integration (default is (0.0, 1.0)).
        ny: The number of points to use for numerical integration (default is 1000).

    Returns:
        An array of values representing $Π_κ^k[f](x)$ for each $x$ in the input array.
    """
    return _Πk_jit(f, x, k, κ, μ, S, ny)


def Π(
    f: Function,
    x: VertexType,
    K: int,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    r"""
    Compute the operator $Π_κ^k[f](x)$ for $k ∈ \{0, 1, …, K\}$ for each $x$ in the input array.

    Args:
        f: A function that takes an array of vertices and returns an array of values.
        x: An array of vertices at which to evaluate the operator.
        K: The maximum threshold of vertices.
        κ: A kernel function that takes two arrays of vertices and returns an array of values.
        μ: A measure function that takes an array of vertices and returns an array of measure values.
        S: A tuple defining the domain of integration (default is (0.0, 1.0)).
        ny: The number of points to use for numerical integration (default is 1000).

    Returns:
        An array of shape (len(x), K+1) representing $Π_κ^k[f](x)$ for each $x$ in the input array and for $k = 0, 1, …, K$.
    """
    return _Π_jit(f, x, K, κ, μ, S, ny)


def Ψ(
    f: Function,
    x: VertexType,
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    r"""
    Compute the operator
    $$
    Ψ_κ\[f\](x) = ∑_{k=0}^{K} η_k(x) \left(1 - ∑\_{j=0}^{k-1} Π_κ^j\[f\](x)\right)
    $$
    for each $x$ in the input array.

    Args:
        f: A function that takes an array of vertices and returns an array of values.
        x: An array of vertices at which to evaluate the operator.
        κ: A kernel function that takes two arrays of vertices and returns an array of values.
        μ: A measure function that takes an array of vertices and returns an array of measure values.
        η: A threshold measure function that takes an array of vertices and returns an array of threshold values.
        S: A tuple defining the domain of integration (default is (0.0, 1.0)).
        ny: The number of points to use for numerical integration (default is 1000).

    Returns:
        An array of values representing $Ψ_κ[f](x)$ for each $x$ in the input array.
    """
    return _Ψ_jit(f, x, κ, μ, η, S, ny)


@jax.jit(static_argnames=["f", "κ", "μ", "S", "ny"])
def _Λ_jit(
    f: Function,
    x: VertexType,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    y = jnp.linspace(S[0], S[1], ny)
    f_y = _f(f, y)
    κy_x = _κy(κ, x, y)
    Δμ_y = _μ(μ, y)
    Λf_x = _Λf(f_y, κy_x, Δμ_y)
    return Λf_x


def _Λf(f_y: Array, κy_x: Array, Δμ_y: Array) -> Array:
    Λf_x = jnp.sum(κy_x * f_y * Δμ_y, axis=1)
    return Λf_x


@jax.jit(static_argnames=["f", "κ", "μ", "S", "ny"])
def _Πk_jit(
    f: Function,
    x: VertexType,
    k: int,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    Λf_x = _Λ_jit(f, x, κ, μ, S, ny)
    Πf_k_x = _Πkf(Λf_x, k)
    return Πf_k_x


def _factorial(n: int) -> int:
    return jnp.arange(1, n + 1).prod()


def _Πkf(Λf_x: Array, k: int) -> Array:
    Πkf_x = jnp.power(Λf_x, k) / _factorial(k) * jnp.exp(-Λf_x)
    return Πkf_x


@jax.jit(static_argnames=["f", "κ", "μ", "S", "ny"])
def _Π_jit(
    f: Function,
    x: VertexType,
    K: int,
    κ: Kernel,
    μ: TypeMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    Λf_x = _Λ_jit(f, x, κ, μ, S, ny)
    Πf_xk = _Πf(Λf_x, K)
    return Πf_xk


def _Πf(Λf_x: Array, K: int) -> Array:
    Πf_xk = jnp.stack([_Πkf(Λf_x, k) for k in range(K)], axis=1)
    Πf_xk = jnp.concatenate([jnp.zeros((Λf_x.shape[0], 1)), Πf_xk], axis=1)
    return Πf_xk


@jax.jit(static_argnames=["f", "κ", "μ", "η", "S", "ny"])
def _Ψ_jit(
    f: Function,
    x: VertexType,
    κ: Kernel,
    μ: TypeMeasure,
    η: ThresholdMeasure,
    S: TypeDomain = TYPE_DOMAIN,
    ny: IntegralNum = INTEGRAL_NUM,
) -> Array:
    y = jnp.linspace(S[0], S[1], ny)
    η_x = η(x)
    K = η_x.shape[1] - 1
    f_y = _f(f, y)
    κy_x = _κy(κ, x, y)
    Δμ_y = _μ(μ, y)
    Λf_x = _Λf(f_y, κy_x, Δμ_y)
    Πf_xk = _Πf(Λf_x, K)
    Ψf_x = _Ψf(η_x, Πf_xk)
    return Ψf_x


def _Ψf(η_xk: Array, Πf_xk: Array) -> Array:
    Ψf_x = ((1 - Πf_xk.cumsum(axis=1)) * η_xk).sum(axis=1)
    return Ψf_x


def _f(f: Function, x: VertexType) -> Array:
    return f(x)


def _κy(κ: Kernel, x: VertexType, y: VertexType) -> Array:
    return κ(y[None, :], x[:, None])


def _μ(μ: TypeMeasure, x: VertexType) -> Array:
    return μ(x)
