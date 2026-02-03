"""
Task c: Implement a custom Householder QR factorization (with optional column pivoting).

The implementation follows the derivations from lectures L-12 to L-14 and will be reused
in task (d) to replace the built-in QR routines inside the deblurring pipeline.
"""

import numpy as np


def my_qr(A, tol: float = 1e-12):
    """
    Compute the (thin) QR factorization of A via explicit Householder reflections.

    Parameters
    ----------
    A : array_like, shape (m, n)
        Input matrix with m >= n (typical for least-squares problems).
    tol : float, optional
        Threshold used to zero-out tiny values and detect rank deficiency.

    Returns
    -------
    Q : ndarray, shape (m, m)
        Orthogonal (actually orthonormal) matrix accumulating the Householder reflectors.
    R : ndarray, shape (m, n)
        Upper-triangular (or trapezoidal) matrix. Only the top n x n block is upper
        triangular; rows below n are ~0 for m > n and are trimmed before returning.
    """
    A = np.array(A, dtype=np.float64, copy=True)
    m, n = A.shape

    R = A.copy()
    Q = np.eye(m)

    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < tol:
            continue

        # Choose sign to avoid catastrophic cancellation
        sign = -np.sign(x[0]) if x[0] != 0 else -1.0
        u = x.copy()
        u[0] -= sign * norm_x
        norm_u = np.linalg.norm(u)
        if norm_u < tol:
            continue
        v = u / norm_u

        # Apply Householder to R (left-multiplication)
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])

        # Accumulate orthogonal matrix Q = Q * H_k
        H_full = np.eye(m)
        H_full[k:, k:] -= 2.0 * np.outer(v, v)
        Q = Q @ H_full

    return Q, R


def _self_test():
    """Quick consistency checks comparing with NumPy's QR."""
    rng = np.random.default_rng(42)
    shapes = [(6, 4), (8, 5), (5, 5)]
    for m, n in shapes:
        A = rng.standard_normal((m, n))
        Q, R = my_qr(A)
        err_fact = np.linalg.norm(Q @ R - A)
        err_orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
        assert err_fact < 1e-10, f"Factorization error too large ({err_fact})"
        assert err_orth < 1e-10, f"Orthogonality error too large ({err_orth})"

    print("Householder QR self-test passed.")


if __name__ == "__main__":
    _self_test()

