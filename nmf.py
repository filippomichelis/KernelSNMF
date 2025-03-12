import numpy as np 

#ToDo: integrate the main algorithm in this class creating a method
class NMF:
    def __init__(self, n_components, init='random', random_state=None):
        """
        Nonnegative Matrix Factorization (NMF) class.
        
        Parameters:
          n_components : Target rank (number of components).
          init         : Initialization method ('random' or 'nndsvd').
          random_state : Seed for random number generator.
        """
        self.n_components = n_components
        self.init = init
        self.random_state = np.random.RandomState(random_state)
        self.W = None
        self.H = None

    def _initialize_matrices(self, X):
        m, n = X.shape
        r = self.n_components
        if self.init == 'random':
            # Random initialization (values in [0,1))
            self.W = self.random_state.rand(m, r)
            self.H = self.random_state.rand(r, n)
        elif self.init == 'nndsvd':
            # NNDSVD initialization (Nonnegative Double SVD)
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            W = np.zeros((m, r))
            H = np.zeros((r, n))
            # First component
            W[:, 0] = np.sqrt(S[0]) * np.maximum(U[:, 0], 0)
            H[0, :] = np.sqrt(S[0]) * np.maximum(Vt[0, :], 0)
            for j in range(1, r):
                x = U[:, j]
                y = Vt[j, :]
                xp = np.maximum(x, 0)
                xn = np.maximum(-x, 0)
                yp = np.maximum(y, 0)
                yn = np.maximum(-y, 0)
                m1 = np.linalg.norm(xp) * np.linalg.norm(yp)
                m2 = np.linalg.norm(xn) * np.linalg.norm(yn)
                if m1 >= m2:
                    u = xp / (np.linalg.norm(xp) + 1e-10)
                    v = yp / (np.linalg.norm(yp) + 1e-10)
                    sigma = m1
                else:
                    u = xn / (np.linalg.norm(xn) + 1e-10)
                    v = yn / (np.linalg.norm(yn) + 1e-10)
                    sigma = m2
                W[:, j] = np.sqrt(S[j] * sigma) * u
                H[j, :] = np.sqrt(S[j] * sigma) * v
            self.W = W
            self.H = H
        else:
            raise ValueError("Unknown initialization method: " + self.init)

    def fit(self, X, method='mu_fro', max_iter=200, tol=1e-4, verbose=False):
        """
        Factorize the nonnegative matrix X into W and H.
        
        Parameters:
          X        : Input nonnegative data matrix.
          method   : Fitting method. Options:
                     - 'mu_fro' : Multiplicative updates (Frobenius norm).
                     - 'mu_kl'  : Multiplicative updates (KL divergence).
                     - 'als'    : Alternating Least Squares.
          max_iter : Maximum number of iterations.
          tol      : Tolerance for convergence.
          verbose  : If True, print progress information.
          
        Returns:
          Tuple (W, H) with the factorized matrices.
        """
        self._initialize_matrices(X)
        if method == 'mu_fro':
            self._fit_mu_fro(X, max_iter, tol, verbose)
        elif method == 'mu_kl':
            self._fit_mu_kl(X, max_iter, tol, verbose)
        elif method == 'als':
            self._fit_als(X, max_iter, tol, verbose)
        else:
            raise ValueError("Unknown fitting method: " + method)
        return self.W, self.H

    def _fit_mu_fro(self, X, max_iter, tol, verbose):
        """Multiplicative updates using the Frobenius norm."""
        epsilon = 1e-10
        for i in range(max_iter):
            # Update H
            numerator = np.dot(self.W.T, X)
            denominator = np.dot(np.dot(self.W.T, self.W), self.H) + epsilon
            self.H *= numerator / denominator
            
            # Update W
            numerator = np.dot(X, self.H.T)
            denominator = np.dot(self.W, np.dot(self.H, self.H.T)) + epsilon
            self.W *= numerator / denominator
            
            error = np.linalg.norm(X - np.dot(self.W, self.H), ord='fro')
            if verbose and (i % 10 == 0 or i == max_iter - 1):
                print("MU-Fro Iteration {}: error = {:.4f}".format(i, error))
            if error < tol:
                break

    def _fit_mu_kl(self, X, max_iter, tol, verbose):
        """Multiplicative updates using the Kullback–Leibler divergence."""
        epsilon = 1e-10
        one = np.ones(X.shape)
        for i in range(max_iter):
            WH = np.dot(self.W, self.H) + epsilon
            # Update H
            numerator = np.dot(self.W.T, X / WH)
            denominator = np.dot(self.W.T, one) + epsilon
            self.H *= numerator / denominator
            
            WH = np.dot(self.W, self.H) + epsilon
            # Update W
            numerator = np.dot((X / WH), self.H.T)
            denominator = np.dot(one, self.H.T) + epsilon
            self.W *= numerator / denominator
            
            divergence = np.sum(X * np.log((X + epsilon) / (WH + epsilon)) - X + WH)
            if verbose and (i % 10 == 0 or i == max_iter - 1):
                print("MU-KL Iteration {}: divergence = {:.4f}".format(i, divergence))
            if divergence < tol:
                break

    def _fit_als(self, X, max_iter, tol, verbose):
        """
        Alternating Least Squares (ALS) update.
        This version computes closed-form solutions for W and H 
        and then projects negative values to zero.
        """
        for i in range(max_iter):
            # Update W with fixed H
            HHT = np.dot(self.H, self.H.T)
            try:
                invHHT = np.linalg.inv(HHT)
            except np.linalg.LinAlgError:
                invHHT = np.linalg.pinv(HHT)
            self.W = np.dot(np.dot(X, self.H.T), invHHT)
            self.W = np.maximum(self.W, 0)
            
            # Update H with fixed W
            WTW = np.dot(self.W.T, self.W)
            try:
                invWTW = np.linalg.inv(WTW)
            except np.linalg.LinAlgError:
                invWTW = np.linalg.pinv(WTW)
            self.H = np.dot(invWTW, np.dot(self.W.T, X))
            self.H = np.maximum(self.H, 0)
            
            error = np.linalg.norm(X - np.dot(self.W, self.H), ord='fro')
            if verbose and (i % 10 == 0 or i == max_iter - 1):
                print("ALS Iteration {}: error = {:.4f}".format(i, error))
            if error < tol:
                break

    @staticmethod
    def _project_X(X):
        """
        Project X onto the set Omega = {X >= 0, X(i,j) <= X(i,i) <= 1}.
        """
        # Enforce non-negativity
        X = np.maximum(X, 0)
        
        # Extract and clip diagonal entries to be at most 1
        diag = np.diag(X)
        diag_clipped = np.clip(diag, None, 1)
        
        # Create a matrix where each row i has the diagonal value X(i,i)
        diag_matrix = np.tile(diag_clipped, (X.shape[1], 1)).T
        
        # Enforce X(i,j) <= X(i,i)
        X = np.minimum(X, diag_matrix)
        
        # Ensure the diagonal is exactly the clipped values
        for i in range(X.shape[0]):
            X[i, i] = diag_clipped[i]
        
        return X

    def fit_kernel_separable(self, M, K, lambda_param=1.0, max_iter=100, r=5, verbose=False):
        """
        Kernel Separable NMF algorithm.
        
        Parameters:
            M           : Input matrix.
            K           : Kernel matrix (typically computed as M^T M or via a kernel function).
            lambda_param: Regularization parameter.
            max_iter    : Number of iterations.
            r           : Number of columns (indices) to select based on the largest diagonal entries.
            verbose     : If True, print progress information.
            
        Returns:
            X           : The final matrix X after updates.
            indices     : The indices corresponding to the r largest entries in diag(X).
        """
        n = K.shape[0]  # Assume K is an n x n matrix.
        # Initialize X as the identity matrix.
        X = np.eye(n)
        I = np.eye(n)
        epsilon = 1e-10
        
        for it in range(max_iter):
            # Compute KX and add a small epsilon to avoid division by zero.
            KX = np.dot(K, X)
            denom = 2 * lambda_param * KX + epsilon
            
            # Compute the update factor elementwise.
            numerator = 2 * lambda_param * K - I
            update_factor = numerator / denom
            
            # Update X elementwise.
            X = X * update_factor
            
            # Project X onto Omega = {X >= 0, X(i,j) <= X(i,i) <= 1}
            X = self._project_X(X)
            
            if verbose and (it % 10 == 0 or it == max_iter - 1):
                diag_norm = np.linalg.norm(np.diag(X))
                print("Kernel Separable Iteration {}: ||diag(X)|| = {:.4f}".format(it, diag_norm))
        
        # Select the indices corresponding to the r largest diagonal entries.
        diag_X = np.diag(X)
        indices = np.argsort(diag_X)[-r:][::-1]  # sorted in descending order
        
        return X, indices