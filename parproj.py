import numpy as np
import heapq

def project_weighted_column(x, w, ub):
    x = np.asarray(x).copy()
    w = np.asarray(w).copy()
    n = len(x)

    assert np.all(w > 0), "Weights must be strictly positive."

    x0 = max(0, x[0])
    z = np.maximum(x, 0)
    values = w[0] * z[1:] / w[1:]
    B = []
    B_star = []

    for j, val in enumerate(values, start=1):
        if val <= x0:
            B.append(j)
        elif val > 1:
            B_star.append(j)

    p = w[0] * x[0] + sum(w[j] * x[j] for j in B_star)
    q = w[0]**2 + sum(w[j]**2 for j in B_star)
    t = w[0] * p / q if q > 0 else 0
    t = min(max(0, t), ub)

    heap = [(w[0] * x[j] / w[j], j) for j in B]
    heapq.heapify(heap)

    while heap and t < heap[0][0]:
        _, j = heapq.heappop(heap)
        B_star.append(j)
        p += w[j] * x[j]
        q += w[j]**2
        t = w[0] * p / q if q > 0 else 0
        t = min(max(0, t), ub)

    z = np.zeros_like(x)
    z[0] = t
    for j in range(1, n):
        if j in B_star:
            z[j] = t * w[j] / w[0]
        else:
            z[j] = min(max(0, x[j]), t * w[j] / w[0])

    return z

def proj_omega_weighted(C, w=None, ub=1.0, diag_idx=None, verbose=False):
    C = np.asarray(C)
    n, m = C.shape
    assert n == m, "Matrix C must be square."

    if w is None:
        w = np.ones(n)
    if diag_idx is None:
        diag_idx = np.arange(n)

    X = np.zeros_like(C)

    for j in range(n):
        z = C[:, j].copy()
        d_idx = diag_idx[j]

        z[[0, d_idx]] = z[[d_idx, 0]]
        w[[0, d_idx]] = w[[d_idx, 0]]

        z_proj = project_weighted_column(z, w, ub)

        z_proj[[0, d_idx]] = z_proj[[d_idx, 0]]
        w[[0, d_idx]] = w[[d_idx, 0]]

        X[:, j] = z_proj

        if verbose:
            err = np.linalg.norm(C[:, j] - z_proj)
            print(f"Col {j}, diag at {d_idx}, projection error: {err:.4f}")

    return X
