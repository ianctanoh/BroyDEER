"""
low_rank_rnn.py
Low-rank RNN dynamical system.

Discretized version of the continuous-time low-rank RNN from:
  Pals et al., "Inferring stochastic low-rank recurrent neural networks
  from neural data", NeurIPS 2024.

Full N-dimensional dynamics (no projection to latent space):

  x_t = a * x_{t-1} + (dt/tau) * M @ N^T @ phi(x_{t-1}) + epsilon_t

where:
  - x_t in R^N       (neuron activities)
  - J = M @ N^T       rank-R weight matrix, M, N in R^{N x R}
  - a = 1 - dt/tau    decay factor
  - phi               element-wise nonlinearity (tanh or piecewise-linear)
  - epsilon_t          noise (driver)

Jacobian structure:  a * I_N + (dt/tau) * M @ N^T @ diag(phi'(x))
  = diagonal + rank R  -->  ideal for Broyden-DEER
"""

import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class LowRankRNN(eqx.Module):
    N: int
    R: int
    M: jnp.ndarray      # (N, R)
    Nt: jnp.ndarray     # (N, R) -- named Nt to avoid confusion with N (int)
    a: float             # decay = 1 - dt/tau
    dt_over_tau: float
    phi: str             # "tanh" or "relu"

    def __init__(self, N: int, R: int, key: jr.PRNGKey,
                 tau: float = 10.0, dt: float = 1.0,
                 g: float = 1.0, phi: str = "tanh"):
        """
        Args:
            N    : number of neurons
            R    : rank of weight matrix J = M @ N^T
            key  : PRNG key
            tau  : time constant
            dt   : discretization timestep
            g    : gain (controls spectral radius of J)
            phi  : nonlinearity, "tanh" or "relu"
        """
        self.N = N
        self.R = R
        self.a = 1.0 - dt / tau
        self.dt_over_tau = dt / tau
        self.phi = phi

        k1, k2 = jr.split(key)
        # Scale so J = M @ Nt^T has spectral norm ~ g
        self.M  = jr.normal(k1, (N, R)) * (g / jnp.sqrt(N))
        self.Nt = jr.normal(k2, (N, R)) * (1.0 / jnp.sqrt(R))

    def _phi(self, x):
        if self.phi == "relu":
            return jnp.maximum(x, 0.0)
        else:  # tanh
            return jnp.tanh(x)

    def deer_fxn(self, state, input):
        """
        One step of the low-rank RNN.

        Args:
            state : x_{t-1}, shape (N,)
            input : epsilon_t (noise), shape (N,)

        Returns:
            x_t : shape (N,)
        """
        recurrent = self.M @ (self.Nt.T @ self._phi(state))
        return self.a * state + self.dt_over_tau * recurrent + input

    def scan_fxn(self, state, input):
        out = self.deer_fxn(state, input)
        return out, out
