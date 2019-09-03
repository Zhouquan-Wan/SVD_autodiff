from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.linalg import adjoint as h
import sys

def safe_inverse(x, epsilon=1E-12):
    return x / (x ** 2 + epsilon)


tf.enable_eager_execution()


@tf.custom_gradient
def SVD_AD(A):
    S1, U, V = tf.svd(A)

    def grad(*dy):
        d = 1e-10
        dS, dU, dV = dy
        dtype = U.dtype
        S = tf.cast(S1, dtype=dtype)
        dS = tf.cast(dS, dtype=dtype)
        ms = tf.diag(S)
        dAs = U @ tf.diag(dS) @ h(V)

        F = S * S - (S * S)[:, None]
        F = safe_inverse(F) - tf.diag(tf.diag_part(safe_inverse(F)))

        J = F * (h(U) @ dU)
        dAu = U @ (J + h(J)) @ ms @ h(V)

        K = F * (h(V) @ dV)
        dAv = U @ ms @ (K + h(K)) @ h(V)

        O = h(dU) @ U @ tf.diag(safe_inverse(S))
        dAc = 1 / 2.0 * h(V @ (tf.matrix_diag(tf.diag_part(O - tf.conj(O)))) @ h(U))

        dAv = dAv + U @ tf.diag(safe_inverse(S)) @ h(dV) @ (
            tf.eye(tf.size(V[:, 1]), dtype=dtype) - V @ h(V)
        )
        dAu = dAu + h(
            V
            @ tf.diag(safe_inverse(S))
            @ h(dU)
            @ (tf.eye(tf.size(U[:, 1]), dtype=dtype) - U @ h(U))
        )
        return dAv + dAu + dAs + dAc

    return [S1, U, V], grad


def l(A):
    S, U, V = SVD_AD(A)
    return tf.real(U[0, 0] * tf.conj(V[0, 0]))


def test(m, n):
    Ax = np.random.randn(m, n)
    Ay = np.random.randn(m, n)
    Ax = Ax.astype(np.complex128)
    Ay = Ay.astype(np.complex128)
    A = tf.constant(Ax + 1.0j * Ay)
    # auto diff
    with tf.GradientTape() as t:
        t.watch(A)
        y = l(A)
        DA_ad = t.gradient(y, A)
    DA_ad = DA_ad.numpy()
    print("auto diff:\n", DA_ad)
    # numerical
    m = tf.size(A[:, 1])
    n = tf.size(A[1, :])
    d = 1e-6
    DA = np.zeros(shape=(m, n))
    DA = DA.astype(np.complex64)
    for i in range(0, m):
        for j in range(0, n):
            dA = np.zeros(shape=(m, n))
            dA[i, j] = 1
            dA = tf.constant(dA, dtype=A.dtype)
            DA[i, j] = (
                tf.cast((l(A + d * dA) - l(A)) / d, dtype=A.dtype)
                + 1.0j * tf.cast((l(A + d * 1.0j * dA) - l(A)) / d, dtype=A.dtype)
            ).numpy()
    print("numerical:\n", DA)
    # difference
    # print("difference:\n",DA-DA_ad)
    print("close?:\n", np.allclose(DA, DA_ad))


if __name__ == "__main__":
    test(int(sys.argv[1]), int(sys.argv[2]))
