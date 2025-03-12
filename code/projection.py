import numpy as np


##############
##  Dykstra ##
##############
def project_onto_Sigma(X0, max_iter=100, tol=1e-8):
    """
    Attempt to project a nonnegative matrix X0 onto the set:
        Sigma = { X >= 0 | X_{ii} <= 1, and w_i X_{ij} <= w_j X_{ii} }
    via an iterative 'repair' approach.

    Parameters
    ----------
    X0       : np.array, shape (n, n). Initial matrix to be projected.
    max_iter : int, maximum number of repair passes.
    tol      : float, stopping threshold on Frobenius change.

    Returns
    -------
    X : np.array, shape (n, n), the repaired matrix (approx. projection).
    """

    X = np.maximum(X0, 0.0)  # Ensure nonnegative to start
    n = X.shape[0]
    assert X.shape[1] == n, "Must be an n×n matrix"

    for it in range(max_iter):
        X_old = X.copy()

        # 1) Clamp diagonal to be at most 1
        for i in range(n):
            if X[i, i] > 1:
                X[i, i] = 1.0

        # 2) Recompute column sums w_j = sum_k X[k, j]
        w = X.sum(axis=0)  # shape (n,)

        # 3) Enforce ratio constraints: for each (i, j),
        #    w_i * X[i, j] <= w_j * X[i, i].
        #    => X[i, j] <= (w_j / w_i) * X[i, i], if w_i > 0
        for j in range(n):
            colsum_j = w[j]
            for i in range(n):
                if w[i] > 0.0 and X[i, i] > 0.0:
                    ratio_max = (colsum_j / w[i]) * X[i, i]

                    if X[i, j] > ratio_max:
                        diff = X[i, j] - ratio_max
                        X[i, j] = ratio_max
                        colsum_j -= diff
                else:
                    if X[i, j] > 0:
                        colsum_j -= X[i, j]
                        X[i, j] = 0.0

            w[j] = colsum_j

        # 4) Enforce nonnegativity once more (just in case)
        X = np.maximum(X, 0.0)

        # 5) Check for convergence in Frobenius norm
        diff = np.linalg.norm(X - X_old, 'fro')
        if diff < tol:
            break

    return X

def dykstra_projection(X0, project_funcs, max_iter=1000, tol=1e-6):
    """
    X0           : initial m×n matrix (np.array).
    project_funcs: list of functions [P1, P2, ..., PK]
                   Each Pi(X) projects onto set C_i.
    max_iter     : maximum number of full cycles.
    tol          : stopping threshold for changes in X.
    Returns X, the projection of X0 onto intersection of all sets C1,...,CK
    """
    K = len(project_funcs)
    R = [np.zeros_like(X0) for _ in range(K)]
    
    X = X0.copy()
    for iter in range(max_iter):
        X_old = X.copy()
        
        # Cycle over sets
        for k in range(K):
            Y = project_funcs[k](X + R[k])
            R[k] = X + R[k] - Y
            X = Y
        
        diff = np.linalg.norm(X - X_old, 'fro')
        if diff < tol:
            break
    
    return X


#####################
## Bold projection ##
#####################
# (Note: this is a more efficient but likely not working projection strategy)
def project_X(X):
    """
    Project X onto the set Omega = {X >= 0, X(i,j) <= X(i,i) <= 1}.
    """
    X = np.maximum(X, 0)  # enforce nonnegativity
    diag = np.diag(X)
    diag_clipped = np.clip(diag, None, 1)
    diag_matrix = np.tile(diag_clipped, (X.shape[1], 1)).T
    X = np.minimum(X, diag_matrix)
    for i in range(X.shape[0]):
        X[i, i] = diag_clipped[i]
    return X