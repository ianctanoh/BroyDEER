"""
diffusion.py
DDIM sampling as a discrete-time dynamical system for DEER.

Standard DDIM sampling iterates from a noisy latent x_T toward a clean x_0
via:
    x_{t-1} = coef_x[t] * x_t + coef_eps[t] * eps_theta(x_t, t)
              + coef_noise[t] * z_t.

We treat the entire trajectory as the fixed point of a forward-time dynamical
system suitable for DEER / Broyden-DEER / accelerated-Broyden / full-Broyden:

    state[s+1] = f(state[s], driver[s])

where state[s] = x_{T-s}, so state[0] is pure noise and state[T] is the clean
image. The driver at BroyDEER step s carries the per-step injected noise z_t
and the original DDIM timestep index t = T - s.

The dynamics function evaluates a noise predictor eps_theta -- which can be a
real neural network (SD UNet, DiT) or a closed-form toy model -- and applies
one DDIM update.

The toy factory `make_toy_diffusion` plugs in `ToyEps`, the Bayes-optimal
noise predictor for data ~ N(mu, I). This gives a fully analytic test problem
where the BroyDEER fixed point is known to match sequential DDIM exactly.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


# --- noise schedules ---

def linear_beta_schedule(T, beta_min=1e-4, beta_max=1e-1):
    """Linear beta schedule, length T+1 (one beta per latent index)."""
    return jnp.linspace(beta_min, beta_max, T + 1)


def cumulative_alpha(betas):
    """alpha_bar_t = prod_{s <= t} (1 - beta_s)."""
    alphas = 1.0 - betas
    return jnp.cumprod(alphas)


# --- DDIM coefficients ---

def build_ddim_coeffs(alpha_bar, eta=0.0):
    """
    Build per-timestep coefficients for one DDIM step
        x_{t-1} = coef_x[t] * x_t + coef_eps[t] * eps_theta(x_t, t)
                  + coef_noise[t] * z_t.

    Args:
        alpha_bar : (T+1,) cumulative noise schedule, alpha_bar_t. Convention:
                    alpha_bar[0] ~ 1 (clean), alpha_bar[T] small (noisy).
        eta       : DDIM stochasticity. eta = 0 -> deterministic ODE sampler;
                    eta > 0 -> SDE sampler with per-step injected Gaussian
                    noise.

    Returns:
        coef_x, coef_eps, coef_noise, noise_thr : each (T+1,). Index 0 is a
        no-op placeholder; meaningful entries are at indices t = 1, ..., T.
        `noise_thr[t]` is the eta-independent natural DDIM noise scale used
        by ParaTAA's noise-relative stopping criterion (it equals
        `coef_noise[t] / eta` when eta != 0 and is well-defined at eta = 0).
    """
    ab_t = alpha_bar
    ab_tm1 = jnp.concatenate([jnp.array([1.0]), alpha_bar[:-1]])

    one_minus_ab_t = jnp.clip(1.0 - ab_t, 1e-20)
    one_minus_ab_tm1 = jnp.clip(1.0 - ab_tm1, 0.0)

    # Eta-free natural DDIM noise scale (matches ParaTAA's `noise_thr`):
    #   noise_thr[t] = sqrt((1-ab_{t-1})/(1-ab_t)) * sqrt(1 - ab_t/ab_{t-1})
    noise_thr = jnp.sqrt(one_minus_ab_tm1 / one_minus_ab_t) \
                * jnp.sqrt(jnp.clip(1.0 - ab_t / ab_tm1, 0.0))
    noise_thr = noise_thr.at[0].set(0.0)

    sig = eta * noise_thr

    coef_x = jnp.sqrt(ab_tm1) / jnp.sqrt(ab_t)
    coef_eps = jnp.sqrt(jnp.clip(one_minus_ab_tm1 - sig**2, 0.0)) \
              - jnp.sqrt(ab_tm1) * jnp.sqrt(one_minus_ab_t) / jnp.sqrt(ab_t)
    coef_noise = sig

    # placeholder identity transition at t = 0 (never used by the dynamics
    # but keeps the arrays uniformly indexable)
    coef_x = coef_x.at[0].set(1.0)
    coef_eps = coef_eps.at[0].set(0.0)

    return coef_x, coef_eps, coef_noise, noise_thr


# --- toy noise predictor (Bayes-optimal eps_theta for N(mu, I) data) ---

class ToyEps(eqx.Module):
    """
    Closed-form eps predictor for data ~ N(mu, I). Stand-in for a trained
    neural noise predictor; useful for validating BroyDEER integration on a
    DDIM problem with a known answer.

    For x_0 ~ N(mu, I), the marginal at time t is
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,
    so x_t ~ N(sqrt(alpha_bar_t) * mu, I) and the Bayes-optimal noise
    predictor (derived from Tweedie's formula) is

        eps*(x_t, t) = sqrt(1 - alpha_bar_t) * (x_t - sqrt(alpha_bar_t) * mu).

    No training, no parameters, no autograd -- just a closed-form function
    that mimics the (x, t) -> eps signature of a real noise predictor.
    """
    mu: jnp.ndarray         # (D,) target Gaussian mean
    alpha_bar: jnp.ndarray  # (T+1,) cumulative noise schedule

    def __call__(self, x, t):
        ab = self.alpha_bar[t]
        return jnp.sqrt(jnp.clip(1.0 - ab, 0.0)) * (x - jnp.sqrt(ab) * self.mu)


# --- mixture-of-Gaussians noise predictor (Bayes-optimal for sum_k pi_k N(mu_k, I)) ---

class MixtureToyEps(eqx.Module):
    """
    Closed-form Bayes-optimal eps predictor for data ~ sum_k pi_k N(mu_k, I).

    The marginal at time t is sum_k pi_k N(sqrt(alpha_bar_t) mu_k, I), so the
    soft-assignment posterior over cluster identity is
        w_k(x_t) = softmax_k( log pi_k - 0.5 ||x_t - sqrt(alpha_bar_t) mu_k||^2 ).

    The Bayes-optimal noise predictor has the same shape as ToyEps but with the
    cluster mean replaced by the soft-weighted average mu_bar(x_t):
        eps*(x_t, t) = sqrt(1 - alpha_bar_t) * (x_t - sqrt(alpha_bar_t) mu_bar(x_t))
        mu_bar(x_t)  = sum_k w_k(x_t) mu_k.

    Unlike ToyEps, eps* is now NONLINEAR in x_t. The Jacobian
        d eps* / d x = sqrt(1 - alpha_bar_t) * (I - alpha_bar_t * Sigma_w(x_t))
    where
        Sigma_w(x_t) = sum_k w_k(x_t) mu_k mu_k^T - mu_bar(x_t) mu_bar(x_t)^T
    is a soft-assignment covariance of the cluster means with rank at most K-1.
    The DDIM-step Jacobian is therefore diagonal-plus-low-rank:
        J_s = c_s I_D - beta_s Sigma_w(x_t)
    which is exactly the structure Broyden / accelerated-Broyden / full-Broyden
    are designed to exploit, and breaks the DEER == quasi-DEER degeneracy of
    the linear-Gaussian ToyEps case.
    """
    pis: jnp.ndarray         # (K,) mixture weights, sum to 1
    mus: jnp.ndarray         # (K, D) cluster means
    alpha_bar: jnp.ndarray   # (T+1,) cumulative noise schedule

    def __call__(self, x, t):
        ab = self.alpha_bar[t]
        s = jnp.sqrt(ab)
        diffs = x[None, :] - s * self.mus                                # (K, D)
        logits = jnp.log(self.pis) - 0.5 * jnp.sum(diffs * diffs, axis=-1)  # (K,)
        w = jax.nn.softmax(logits)                                       # (K,)
        mu_bar = jnp.einsum("k,kd->d", w, self.mus)                      # (D,)
        return jnp.sqrt(jnp.clip(1.0 - ab, 0.0)) * (x - s * mu_bar)


# --- main DDIM dynamical system wrapper ---

class DDIMDynamics(eqx.Module):
    """
    DDIM sampling viewed as a forward-time discrete dynamical system:

        state[s+1] = coef_x[t] * state[s]
                     + coef_eps[t] * eps_model(state[s], t)
                     + coef_noise[t] * z[s]

    where t = T - s is the original DDIM timestep. With BroyDEER's convention
    (initial_state plus a length-T trajectory of states), state[0] = x_T is
    pure noise and the final state of the trajectory is the clean image x_0.

    Fields:
        D          : latent dimension
        T          : number of DDIM transitions (so T+1 latents in total)
        eps_model  : callable (x, t) -> eps prediction. Can be ToyEps or any
                     real neural noise predictor wrapped to match this
                     signature.
        coef_x, coef_eps, coef_noise : (T+1,) precomputed DDIM coefficients
        noise_thr  : (T+1,) eta-free natural DDIM noise scale, used by
                     ParaTAA's noise-relative stopping criterion. Defined
                     even at eta = 0 (where coef_noise is identically zero).
    """
    D: int = eqx.field(static=True)
    T: int = eqx.field(static=True)
    eps_model: eqx.Module
    coef_x: jnp.ndarray
    coef_eps: jnp.ndarray
    coef_noise: jnp.ndarray
    noise_thr: jnp.ndarray

    def deer_fxn(self, state, driver):
        """
        Args:
            state  : (D,) current noisy latent x_t
            driver : (D + 1,) concatenation of [z_t (D), t (1)] where z_t is
                     the per-step injected noise (used only when eta > 0) and
                     t is the original DDIM timestep cast to float.

        Returns:
            (D,) next latent x_{t-1}.
        """
        z = driver[: self.D]
        t = driver[self.D].astype(jnp.int32)
        eps = self.eps_model(state, t)
        return (self.coef_x[t] * state
                + self.coef_eps[t] * eps
                + self.coef_noise[t] * z)

    def scan_fxn(self, state, driver):
        out = self.deer_fxn(state, driver)
        return out, out


# --- factories ---

def make_toy_diffusion(D=16, T=50, mu=None, eta=0.0,
                       beta_min=1e-4, beta_max=1e-1, seed=0):
    """
    Build a toy DDIM dynamical system whose noise predictor is the closed-form
    Bayes-optimal score for data ~ N(mu, I). Useful for validating BroyDEER
    integration on the DDIM problem structure without any neural network.

    Args:
        D        : latent dimension
        T        : number of DDIM transitions (T + 1 latents in trajectory)
        mu       : (D,) target Gaussian mean. If None, sampled from N(0, 9 I).
        eta      : DDIM stochasticity. eta = 0 -> ODE sampler.
        beta_min : start of linear beta schedule
        beta_max : end of linear beta schedule
        seed     : PRNG seed for mu (if not given), x_T, and per-step noise

    Returns:
        deer_fxn : bound method (state, driver) -> next state
        dynamics : the DDIMDynamics object (use dynamics.scan_fxn with
                   jax.lax.scan to get the sequential reference trajectory)
        x0       : (D,) initial state = pure noise sample (= x_T)
        drivers  : (T, D + 1) per-step driver array
        params   : dict of hyperparameters and ground-truth quantities
    """
    key = jr.PRNGKey(seed)
    k_mu, k_init, k_noise = jr.split(key, 3)

    if mu is None:
        mu = jr.normal(k_mu, (D,)) * 3.0

    betas = linear_beta_schedule(T, beta_min=beta_min, beta_max=beta_max)
    alpha_bar = cumulative_alpha(betas)
    coef_x, coef_eps, coef_noise, noise_thr = build_ddim_coeffs(
        alpha_bar, eta=eta
    )

    eps_model = ToyEps(mu=mu, alpha_bar=alpha_bar)
    dynamics = DDIMDynamics(
        D=D, T=T, eps_model=eps_model,
        coef_x=coef_x, coef_eps=coef_eps, coef_noise=coef_noise,
        noise_thr=noise_thr,
    )

    # closure-based deer_fxn / scan_fxn (not bound methods of an eqx.Module).
    # Bound methods of equinox modules trip jax.lax.scan's caching when the
    # module has jnp.ndarray fields, because lax.scan tries to hash the
    # function and equinox's __hash__ recurses into the array fields.
    # Closures are hashed by Python identity, so this is safe.
    D_local = D
    coef_x_local = coef_x
    coef_eps_local = coef_eps
    coef_noise_local = coef_noise

    def deer_fxn(state, driver):
        z = driver[:D_local]
        t = driver[D_local].astype(jnp.int32)
        eps = eps_model(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    def scan_fxn(state, driver):
        out = deer_fxn(state, driver)
        return out, out

    # initial state = pure noise (BroyDEER initial_state = x_T)
    x0 = jr.normal(k_init, (D,))

    # per-step injected noise vectors (only used when eta > 0)
    z_per_step = jr.normal(k_noise, (T, D))

    # at BroyDEER step s, the original DDIM timestep is t = T - s
    # so drivers[0] uses t = T (first DDIM transition x_T -> x_{T-1})
    # and drivers[T - 1] uses t = 1 (last DDIM transition x_1 -> x_0)
    t_indices = (T - jnp.arange(T)).astype(jnp.float32)

    drivers = jnp.concatenate([z_per_step, t_indices[:, None]], axis=1)

    params = {
        "D": D, "T": T, "eta": eta,
        "beta_min": beta_min, "beta_max": beta_max,
        "mu": mu, "alpha_bar": alpha_bar,
        "coef_x": coef_x, "coef_eps": coef_eps, "coef_noise": coef_noise,
        "noise_thr": noise_thr,
        "scan_fxn": scan_fxn,  # for jax.lax.scan reference runs
    }

    return deer_fxn, dynamics, x0, drivers, params


def make_toy_diffusion_mog(D=16, T=50, K=4, mus=None, pis=None, eta=0.0,
                           beta_min=1e-4, beta_max=1e-1,
                           cluster_scale=2.0, seed=0):
    """
    Build a toy DDIM dynamical system whose noise predictor is the closed-form
    Bayes-optimal score for data ~ sum_k pi_k N(mu_k, I). Drop-in replacement
    for `make_toy_diffusion` that exposes a NONLINEAR eps with diagonal+low-rank
    Jacobian, breaking the DEER == quasi-DEER degeneracy of the single-Gaussian
    case.

    Jacobian structure (see MixtureToyEps docstring):
        J_s = c_s I_D - beta_s Sigma_w(x_t),  rank(Sigma_w) <= K - 1.

    Args:
        D             : latent dimension
        T             : number of DDIM transitions (T + 1 latents in trajectory)
        K             : number of mixture components
        mus           : (K, D) cluster means. If None, sampled from
                        N(0, cluster_scale^2 I) per coordinate.
        pis           : (K,) mixture weights. If None, uniform 1/K.
        eta           : DDIM stochasticity. eta = 0 -> ODE sampler.
        beta_min      : start of linear beta schedule
        beta_max      : end of linear beta schedule
        cluster_scale : per-coordinate stdev of mu_k draws when mus is None.
                        Tune to control how separated the clusters are relative
                        to noise: larger -> sharper soft assignments (smaller
                        DEER vs quasi gap), smaller -> more uniform assignments.
        seed          : PRNG seed

    Returns:
        deer_fxn : closure (state, driver) -> next state
        dynamics : DDIMDynamics object wrapping a MixtureToyEps eps_model
        x0       : (D,) initial state = pure noise sample (= x_T)
        drivers  : (T, D + 1) per-step driver array
        params   : dict of hyperparameters and ground-truth quantities
    """
    key = jr.PRNGKey(seed)
    k_mu, k_init, k_noise = jr.split(key, 3)

    if mus is None:
        mus = jr.normal(k_mu, (K, D)) * cluster_scale
    if pis is None:
        pis = jnp.ones(K) / K
    pis = pis / pis.sum()

    betas = linear_beta_schedule(T, beta_min=beta_min, beta_max=beta_max)
    alpha_bar = cumulative_alpha(betas)
    coef_x, coef_eps, coef_noise, noise_thr = build_ddim_coeffs(
        alpha_bar, eta=eta
    )

    eps_model = MixtureToyEps(pis=pis, mus=mus, alpha_bar=alpha_bar)
    dynamics = DDIMDynamics(
        D=D, T=T, eps_model=eps_model,
        coef_x=coef_x, coef_eps=coef_eps, coef_noise=coef_noise,
        noise_thr=noise_thr,
    )

    # closure-based deer_fxn / scan_fxn (see make_toy_diffusion for rationale)
    D_local = D
    coef_x_local = coef_x
    coef_eps_local = coef_eps
    coef_noise_local = coef_noise

    def deer_fxn(state, driver):
        z = driver[:D_local]
        t = driver[D_local].astype(jnp.int32)
        eps = eps_model(state, t)
        return (coef_x_local[t] * state
                + coef_eps_local[t] * eps
                + coef_noise_local[t] * z)

    def scan_fxn(state, driver):
        out = deer_fxn(state, driver)
        return out, out

    x0 = jr.normal(k_init, (D,))
    z_per_step = jr.normal(k_noise, (T, D))
    t_indices = (T - jnp.arange(T)).astype(jnp.float32)
    drivers = jnp.concatenate([z_per_step, t_indices[:, None]], axis=1)

    params = {
        "D": D, "T": T, "K": K, "eta": eta,
        "beta_min": beta_min, "beta_max": beta_max,
        "mus": mus, "pis": pis, "alpha_bar": alpha_bar,
        "coef_x": coef_x, "coef_eps": coef_eps, "coef_noise": coef_noise,
        "noise_thr": noise_thr,
        "scan_fxn": scan_fxn,
    }

    return deer_fxn, dynamics, x0, drivers, params
