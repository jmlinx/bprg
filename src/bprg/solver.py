"""
Neural network approximation for function using Flax
"""

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from bprg import operator as op
from bprg.simulation import estimate_infection
from bprg.type import (
    Array,
    Function,
    IntegralNum,
    Kernel,
    Parameters,
    SimulationNum,
    ThresholdMeasure,
    TypeDomain,
    TypeMeasure,
    TypeNum,
    VertexNum,
)

EPOCHS = 1000
SEED = 0
LR_SCHEDULE = optax.cosine_decay_schedule(init_value=0.05, decay_steps=EPOCHS)
OPTIMIZER = optax.adam(learning_rate=LR_SCHEDULE, b1=0.9, b2=0.999, eps=1e-10)


# Default neural network
class NeuralNetwork(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = x.reshape(-1, 1)  # Ensure input is 2D
        x = nn.Dense(20)(x)
        x = nn.tanh(x)
        x = nn.Dense(20)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x.squeeze(-1)  # Return a 1D array


FNN = NeuralNetwork()


class FixpointSolver(nn.Module):
    r"""
    Solving the fixed point problem
    $$
    Ψ_κ\[f\](·) = f(·)
    $$
    with neural network $f_{nn}$ by
    $$
    \min_θ ∫\_{s ∈ \mathcal{S}} \|Ψ_κ\[f_{nn}\](s; θ) - f_{nn}(s; θ)\|^2 \mathrm{d}μ(s)
    $$

    """

    fnn: NeuralNetwork  # The neural network model
    κ: Kernel  # Kernel: K(x, y)
    μ: TypeMeasure  # Measure: μ(x)
    η: ThresholdMeasure  # Threshold measure: η(x)
    S: TypeDomain  # Domain: (S_min, S_max)
    nx: IntegralNum = 1000  # Number of nodes
    ny: IntegralNum = 1000  # Number of nodes

    def __post_init__(self):
        self.x = jnp.linspace(self.S[0], self.S[1], self.nx)
        self.y = jnp.linspace(self.S[0], self.S[1], self.ny)
        self.Δμ_y = op._μ(self.μ, self.y)
        self.η_xk = self.η(self.x)
        self.K = self.η_xk.shape[1] - 1

    def __call__(self, x, training: bool = True):
        return self.fnn(x, training=training)

    def f(self, params: Parameters, x: Array) -> Array:
        r"""
        Evaluate the neural function $f_{nn}(x; θ)$.

        Args:
            params: model parameters
            x: array of vertex types

        Returns:
            array of function values $f_{nn}(x; θ)$
        """
        return self.fnn.apply(params, x, training=False)

    def Ψf(self, params: Parameters, x: Array) -> Array:
        r"""
        Evaluate the operator image $Ψ[f](x; θ)$.

        Args:
            params: model parameters
            x: array of vertex types

        Returns:
            array of function values $Ψ[f](x; θ)$
        """
        η_xk = self.η(x)
        K = η_xk.shape[1] - 1
        κy_x = op._κy(self.κ, x, self.y)

        f_y = self.f(params, self.y)
        Λf_x = op._Λf(f_y, κy_x, self.Δμ_y)
        Πf_xk = op._Πf(Λf_x, K)
        Ψf_x = op._Ψf(η_xk, Πf_xk)
        return Ψf_x

    def get_f(self, params: Parameters) -> Function:
        r"""
        Return the neural function $f_{nn}(·; θ)$ with $θ$ fixed.

        Args:
            params: model parameters

        Returns:
            function $f_{nn}(·; θ)$
        """
        return partial(self.f, params)

    def get_Ψf(self, params: Parameters) -> Function:
        r"""
        Return the operator image $Ψ[f](·; θ)$ with $θ$ fixed.

        Args:
            params: model parameters

        Returns:
            function $Ψ[f](·; θ)$
        """
        return partial(self.Ψf, params)

    @partial(jax.jit, static_argnames=("self", "optimizer"))
    def train_step(
        self,
        params: Parameters,
        opt_state: optax.OptState,
        optimizer: optax.GradientTransformation,
    ) -> tuple[Parameters, optax.OptState, float]:
        """Perform one training step.

        Args:
            params: model parameters
            opt_state: optimizer state
            optimizer: optax optimizer (static)

        Returns:
            Updated params, opt_state, and loss
        """
        loss = self.loss_fn(params)
        grads = jax.grad(self.loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
        return params, opt_state, loss

    def _f(self, params, training: bool = True) -> Array:
        """Evaluate the neural function."""
        return self.fnn.apply(params, self.x, training=training)

    def _Ψf(self, params, training: bool = False) -> Array:
        """Evaluate the Ψ operator on the neural function."""
        f_y = self.fnn.apply(params, self.y, training=training)
        κy_x = op._κy(self.κ, self.x, self.y)
        Λf_x = op._Λf(f_y, κy_x, self.Δμ_y)

        Πf_xk = op._Πf(Λf_x, self.K)
        Ψf_x = op._Ψf(self.η_xk, Πf_xk)
        return Ψf_x

    def loss_fn(self, params: Parameters) -> float:
        """Compute the total loss.

        Args:
            params: model parameters

        Returns:
            scalar loss value
        """
        f_x = self._f(params, training=True)
        ψf_x = self._Ψf(params, training=False)
        diff = ψf_x - f_x
        loss = jnp.mean(jnp.abs(diff))
        return loss

    def solve(
        self,
        epochs: int = EPOCHS,
        optimizer: optax.GradientTransformation = OPTIMIZER,
        seed: int = SEED,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Train the network to solve the fixed point problem.

        Args:
            epochs: number of training epochs
            optimizer: optax optimizer
            seed: integer seed for JAX random key for initialization
            verbose: whether to print loss at regular intervals

        Returns:
            Dictionary containing trained function, Ψf operator, parameters, and losses
        """
        # Initialize parameters
        key = jax.random.PRNGKey(seed)
        params = self.fnn.init(key, self.x)
        opt_state = optimizer.init(params)

        # Training loop
        losses = jnp.zeros(epochs)
        for epoch in range(epochs):
            params, opt_state, loss = self.train_step(params, opt_state, optimizer)
            losses = losses.at[epoch].set(loss)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        # Get the final function
        f = self.get_f(params)
        Ψf = self.get_Ψf(params)

        result = {}
        result["f"] = f
        result["Ψf"] = Ψf
        result["params"] = params
        result["losses"] = losses

        return result


class MonteCarloSolver:
    """
    Initialize the Monte Carlo solver.

    Args:
        κ: kernel function
        μ: type measure function
        η: threshold measure function
        S: type domain
        nv: number of vertices
        ns: number of types
    """

    def __init__(
        self,
        κ: Kernel,
        μ: TypeMeasure,
        η: ThresholdMeasure,
        S: TypeDomain,
        nv: VertexNum = 1000,
        ns: TypeNum = 100,
    ):

        self.κ = κ
        self.μ = μ
        self.η = η
        self.S = S
        self.nv = nv
        self.ns = ns

    def solve(self, nm: SimulationNum = 1000, seed: int = SEED) -> Array[float]:
        """
        Run Monte Carlo simulations to estimate the final fraction of infected vertices after the bootstrap percolation.

        Args:
            nm: number of Monte Carlo simulations
            seed: random seed for reproducibility

        Returns:
            Array of estimated final fractions of infected vertices by type
        """
        return estimate_infection(self.κ, self.μ, self.η, self.S, self.nv, self.ns, nm=nm, seed=jax.random.key(seed))
