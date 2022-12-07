import jax.numpy as jnp
from jax import jit, lax
from functools import partial


def kronvec_sync_jit(theta_i: jnp.array,
                     p: jnp.array,
                     start: int,
                     stop: int,
                     n: int) -> jnp.array:
    def loop_body(j, p):
        p = p.reshape((2 ** (2 * n - 1), 4), order="C")
        p = p.at[:, (1, 2)].set(0.)
        p = p.at[:, 3].multiply(theta_i[j])
        p = p.flatten(order="F")
        return p
    return lax.fori_loop(start, stop, loop_body, p)


@partial(jit, static_argnums=3)
def kronvec_sync(log_theta: jnp.array, p: jnp.array, i: int, n: int, diag=True) -> jnp.array:
    theta_i = jnp.exp(log_theta.at[i, :].get())
    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, 0, i, n)

    # Non Diagonal Kronecker factors
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = p.at[:, (1, 2)].set(0.)
    p = p.at[:, 3].set(theta_i[i] * p.at[:, 0].get())
    p = lax.cond(diag,
                 lambda p: p.at[:, 0].set(-p.at[:, 3].get()),
                 lambda p: p.at[:, 0].set(0.), p)
    p = p.flatten(order="F")

    # Non Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, i+1, n, n)

    # Diagonal Kronecker factor for seeding
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(0.)
    p = p.flatten(order="F")
    return p

def sync_transp_q(p, tij):
    p = p.at[:, 0].add(-p.at[:, 3].get())
    p = p.at[:, 0].multiply(-tij)
    return p


@partial(jit, static_argnums=3)
def kronvec_sync_transp(log_theta: jnp.array, p: jnp.array, i: int, n: int, diag=True) -> jnp.array:
    theta_i = jnp.exp(log_theta.at[i, :].get())
    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, 0, i, n)

    # Non diagonal Kronecker factor
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = p.at[:, (1, 2)].set(0.)
    p = lax.cond(diag,
                 sync_transp_q,
                 lambda p, tij: p.at[:, 0].set(p.at[:, 3].get()*theta_i[i]), p, theta_i[i])
    p = p.at[:, 3].set(0.)
    p = p.flatten(order="F")

    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, i+1, n, n)

    # Diagonal Kronecker factor for seeding
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(0.)
    p = p.flatten(order="F")
    return p