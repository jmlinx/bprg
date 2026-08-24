import jax
import jax.numpy as jnp
from jax import Array

from bprg.type import BootstrapPercolationRandomGraph, Simplex, ThresholdMeasure, VertexType


# Kernels
def κ_lips(x: VertexType, y: VertexType) -> Array:
    r"""Lipschitz kernel:
    $κ(x, y) = \frac{10 \sqrt{x^2 + y^2}}{1 + \sqrt{|x - y|}}$

    Args:
        x: An array of vertex types.
        y: An array of vertex types.

    Returns:
        An array of kernel values.
    """
    return 10 * jnp.sqrt(x**2 + y**2) / (1 + jnp.sqrt(jnp.abs(x - y)))


def κ_expo(x: VertexType, y: VertexType) -> Array:
    r"""Exponential kernel:
    $κ(x, y) = \frac{5 (e^{x + 0.5 y} - 1)}{1 + \sqrt{|x - y|}}$

    Args:
        x: An array of vertex types.
        y: An array of vertex types.

    Returns:
        An array of kernel values.
    """
    return 5 * (jnp.exp(x + 0.5 * y) - 1) / (1 + jnp.sqrt(jnp.abs(x - y)))


def κ_sing(x: VertexType, y: VertexType) -> Array:
    r"""Kernel with singularity:
    $κ(x, y) = \frac{x + y}{|x - y| + |x - 1| + |y - 1|}$

    Args:
        x: An array of vertex types.
        y: An array of vertex types.

    Returns:
        An array of kernel values.
    """
    return (x + y) / (jnp.abs(x - y) + jnp.abs(x - 1) + jnp.abs(y - 1))


# Vertex type measure
def μ_lebe(x: VertexType) -> Array:
    r"""Vertex type Lebesgue measure:
    $μ(I) = |I|$

    Args:
        x: An array of vertex types.

    Returns:
        An array of measure values.
    """
    Δμ = jnp.empty_like(x)
    Δμ = Δμ.at[0].set(x[1] - x[0])
    Δμ = Δμ.at[-1].set(x[-1] - x[-2])
    Δμ = Δμ.at[1:-1].set((x[2:] - x[:-2]) / 2)
    Δμ = Δμ / Δμ.sum()  # Normalize
    return Δμ


# Threshold measure
def make_η_constant(margin: Simplex) -> ThresholdMeasure:
    r"""Create a constant vertex threshold measure function that is independent of the vertex type.

    Args:
        margin: Probability distribution of vertex thresholds (sums to 1).

    Returns:
        ThresholdMeasure: A function that takes a vertex type and returns the threshold distribution.
    """
    assert jnp.allclose(margin.sum(), 1.0), "margin must be a probability distribution (sum to 1)"

    def _η_constant(x: VertexType) -> Simplex:
        return jnp.array(margin)

    η_const_vmap = jax.vmap(_η_constant, in_axes=0, out_axes=0)

    # Generate LaTeX cases from margin
    cases = " \\\\\n            ".join([f"{float(m):.1f}, & k = {i}" for i, m in enumerate(margin)])
    η_const_vmap.__doc__ = f"""Vertex threshold measure:
            $$η_k(s) ≡ \\begin{{cases}}
            {cases}
            \\end{{cases}}$$
            """
    return η_const_vmap


# Vertex type space
S = (0.0, 1.0)

# Truncated vertex type space to avoid singularity at (1, 1)
S_ε = (0.0, 1.0 - 1e-2)

# Vertex threshold measure: η_0 = 0.1, η_1 = 0.0, η_2 = 0.9
MARGIN = jnp.array([0.1, 0.0, 0.9])
η_const = make_η_constant(MARGIN)

# Vertex number for simulation
NV = 1000

# Number of discretized vertex types over type space S
NS = 100

# Bootstrap Percolation Random Graph presets
preset_lips = BootstrapPercolationRandomGraph(κ=κ_lips, μ=μ_lebe, η=η_const, S=S, nv=NV, ns=NS)
preset_expo = BootstrapPercolationRandomGraph(κ=κ_expo, μ=μ_lebe, η=η_const, S=S, nv=NV, ns=NS)
preset_sing = BootstrapPercolationRandomGraph(κ=κ_sing, μ=μ_lebe, η=η_const, S=S_ε, nv=NV, ns=NS)

preset_dict = {"Lipschitz": preset_lips, "Exponential": preset_expo, "Singular": preset_sing}

nn_config_dict = {
    "Lipschitz": {"layers": 2, "nodes": 20, "epochs": 500},
    "Exponential": {"layers": 2, "nodes": 20, "epochs": 500},
    "Singular": {"layers": 2, "nodes": 32, "epochs": 1000},
}

if __name__ == "__main__":
    import plotly.graph_objects as go

    from bprg.solver import FixpointSolver, NeuralNetwork

    κ_lips = preset_lips.κ
    κ_expo = preset_expo.κ
    κ_sing = preset_sing.κ
    ns = preset_sing.ns

    fnn = NeuralNetwork()
    solver = FixpointSolver(fnn=fnn, κ=κ_sing, μ=μ_lebe, η=η_const, S=S_ε, nx=ns, ny=ns)
    result = solver.solve()
    f = result["f"]
    ψf = result["Ψf"]
    x = jnp.linspace(S_ε[0], S_ε[1], ns)

    go.Figure(
        data=[
            go.Scatter(x=x, y=f(x), mode="lines", name="f(x)"),
            go.Scatter(x=x, y=ψf(x), mode="lines", name="Ψ(f)(x)"),
        ],
        layout=go.Layout(
            title="Neural Network Approximation of f and Ψ(f)",
            xaxis_title="x",
            yaxis_title="Function values",
        ),
    ).show()
