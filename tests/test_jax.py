import fss_jax as fj
import kronecker_vector as kv
import Utilityfunctions as ut
import jax.numpy as jnp
import numpy as np
from timeit import default_timer as timer

# Build theta
n = 12
theta = ut.random_theta(n, 0.4)
i = 1
p0 = np.zeros(2**(2*n+1))
p0[0] = 1

# test the numpy implementation
start = timer()
res_np = kv.kronvec_sync(theta, p0, i, n, transpose=False)
res_met_np = kv.kronvec_met(theta, p0, i, n, transpose=False)
res_prim_np = kv.kronvec_prim(theta, p0, i, n, transpose=False)
res_seed_np = kv.kronvec_seed(theta, p0, n, transpose=False)
end = timer()
print(end - start)

# test the jax implementation
p0_jax = jnp.zeros(2**(2*n+1))
p0_jax = p0_jax.at[0].set(1)
theta_jax = jnp.array(theta)

# Compile the function for a fair comparison
fj.kronvec_sync(theta_jax, p0_jax, i, n, True)
fj.kronvec_met(theta_jax, p0_jax, i, n, True)
fj.kronvec_prim(theta_jax, p0_jax, i, n, True)
fj.kronvec_seed(theta_jax, p0_jax, n, True)

start = timer()
res_jax = fj.kronvec_sync(theta_jax, p0_jax, i, n, True)
res_met_jax = fj.kronvec_met(theta_jax, p0_jax, i, n, True)
res_prim_jax = fj.kronvec_prim(theta_jax, p0_jax, i, n, True)
res_seed_jax = fj.kronvec_seed(theta_jax, p0_jax, n, True)
end = timer()
print(end - start)
fj.qvec(theta_jax, p0_jax, True, n)

start = timer()
res_q = kv.qvec(theta, p0, True)
end = timer()
print(end - start)
start = timer()
res_q_jax = fj.qvec(theta_jax, p0_jax, True, n)
end = timer()
print(end - start)

# Check if both versions produce the same output
res_np = jnp.array(res_np)
res_met_np = jnp.array(res_met_np)
res_prim_np = jnp.array(res_prim_np)
res_seed_np = jnp.array(res_seed_np)
res_q_np = jnp.array(res_q)
assert(jnp.allclose(res_np, res_jax))
assert(jnp.allclose(res_met_np, res_met_jax))
assert(jnp.allclose(res_prim_np, res_prim_jax))
assert(jnp.allclose(res_seed_np, res_seed_jax))
assert(jnp.allclose(res_q_np, res_q_jax))

