import numpy as np
import pandas as pdt
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
from ssr_kronvec_jax import k2d1t, k2ntt, k2dt0
import Utilityfunctions as utils
import ssr_kronvec_jax as  ssr_kv_jx

@partial(jit, static_argnames=["diag", "n", "transpose"])
def kronvec(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    n: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:

    @partial(jit, static_argnames=["diag", "n", "transpose"])
    def _kronvec(
        log_theta: jnp.array,
        p: jnp.array,
        i: int,
        n: int,
        state: jnp.array,
        diag: bool = True,
        transpose: bool = False
    ) -> jnp.array:

        y = p.copy()

        def loop_body_diag(j, val):

            val = lax.switch(
                index=state.at[j].get(),
                branches=[
                    lambda x: x,
                    lambda x: k2d1t(p=x, theta=jnp.exp(
                        log_theta.at[i, j].get()))
                ],
                operand=val
            )

            return val

        # Diagonal Kronecker factors
        y = lax.fori_loop(lower=0, upper=i,
                          body_fun=loop_body_diag, init_val=y)

        # Non-diagonal Kronecker factor
        y = lax.switch(
            index=state.at[i].get(),
            branches=[
                lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
                lambda x: k2ntt(p=x, theta=jnp.exp(
                    log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            ],
            operand=y
        )

        # Diagonal Kronecker factors
        y = lax.fori_loop(lower=i+1, upper=n,
                          body_fun=loop_body_diag, init_val=y)

        return y

    return lax.cond(
        not diag and state[i] != 1,
        lambda: jnp.zeros_like(p),
        lambda: _kronvec(
            log_theta=log_theta,
            p=p,
            i=i,
            n=n,
            state=state,
            diag=diag,
            transpose=transpose
        ),
    )

@ partial(jit, static_argnames=["n", "state_size"])
def kron_diag_i(
        log_theta: jnp.array,
        i: int,
        n: int,
        state: jnp.array,
        state_size: int) -> jnp.array:
    
    diag = jnp.ones(2 ** state_size)

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[j].get(),
            branches=[
                lambda val: val,
                lambda val: k2d1t(val, jnp.exp(log_theta[i, j])),
            ],
            operand=val
        )
        return val
    
    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag
    )

    diag = lax.switch(
        index=state.at[i].get(),
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
        ],
        operand=diag
    )

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    return diag


@partial(jit, static_argnames=["n", "state_size"])
def kron_diag(
        log_theta: jnp.array,
        n: int,
        state: jnp.array,
        state_size: int) -> jnp.array:

    def body_fun(i, val):

        val += kron_diag_i(log_theta=log_theta, i=i, n=n, state=state, state_size=state_size)
        return val
    
    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros(shape=2**state_size)
    )