import jax
import jax.numpy as jnp
from jax import custom_jvp, custom_vjp
import numpy as np
from jax.config import config

config.update("jax_enable_x64", True)
import sys


@custom_vjp
def svd(A):
    return jnp.linalg.svd(A, full_matrices=False)


def _safe_reciprocal(x, epsilon=1e-20):
    return x / (x * x + epsilon)


def h(x):
    return jnp.conj(jnp.transpose(x))


def jaxsvd_fwd(A):
    u, s, v = svd(A)
    return (u, s, v), (u, s, v)


def jaxsvd_bwd(r, tangents):
    U, S, V = r
    du, ds, dv = tangents

    dU = jnp.conj(du)
    dS = jnp.conj(ds)
    dV = jnp.transpose(dv)

    ms = jnp.diag(S)
    ms1 = jnp.diag(_safe_reciprocal(S))
    dAs = U @ jnp.diag(dS) @ V

    F = S * S - (S * S)[:, None]
    F = _safe_reciprocal(F) - jnp.diag(jnp.diag(_safe_reciprocal(F)))

    J = F * (h(U) @ dU)
    dAu = U @ (J + h(J)) @ ms @ V

    K = F * (V @ dV)
    dAv = U @ ms @ (K + h(K)) @ V

    O = h(dU) @ U @ ms1
    dAc = -1 / 2.0 * U @ (jnp.diag(jnp.diag(O - jnp.conj(O)))) @ V

    dAv = dAv + U @ ms1 @ h(dV) @ (jnp.eye(jnp.size(V[1, :])) - h(V) @ V)
    dAu = dAu + (jnp.eye(jnp.size(U[:, 1])) - U @ h(U)) @ dU @ ms1 @ V
    grad_a = jnp.conj(dAv + dAu + dAs + dAc)
    return (grad_a,)


svd.defvjp(jaxsvd_fwd, jaxsvd_bwd)


def l(A):
    u, s, v = svd(A)
    return jnp.real(u[0, -1] * v[-1, 0])


def test(m, n):
    Ax = np.random.randn(m, n)
    Ay = np.random.randn(m, n)
    A = jnp.array(Ax + 1.0j * Ay).astype(jnp.complex128)
    # auto diff
    DA_ad = jax.grad(l)(A)

    print("auto diff:\n", DA_ad)
    # numerical
    d = 1e-6
    DA = np.zeros(shape=(m, n), dtype=np.complex128)
    for i in range(0, m):
        for j in range(0, n):
            dA = np.zeros(shape=(m, n))
            dA[i, j] = 1
            DA[i, j] = (l(A + d * dA) - l(A)) / d - 1.0j * (
                l(A + d * 1.0j * dA) - l(A)
            ) / d
    print("numerical:\n", DA)
    # difference
    # print("difference:\n",DA-DA_ad)
    print("close?:\n", np.allclose(DA, DA_ad))


if __name__ == "__main__":
    test(int(sys.argv[1]), int(sys.argv[2]))
