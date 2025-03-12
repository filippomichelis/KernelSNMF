import numpy as np

def compute_kernel_matrix(X, kernel='linear', **kwargs):
    """
    Compute a kernel matrix for the input data X.
    
    Parameters:
      X       : Input data matrix (samples x features).
      kernel  : Type of kernel. Options:
                - 'linear'
                - 'polynomial' (parameters: degree, coef0)
                - 'rbf' (parameter: gamma; default: 1/n_features)
                - 'sigmoid' (parameters: alpha, coef0)
      **kwargs: Additional keyword arguments for kernel parameters.
      
    Returns:
      Kernel matrix (samples x samples).
    """
    if kernel == 'linear':
        return np.dot(X, X.T)
    elif kernel == 'polynomial':
        degree = kwargs.get('degree', 3)
        coef0 = kwargs.get('coef0', 1)
        return (np.dot(X, X.T) + coef0) ** degree
    elif kernel == 'rbf':
        gamma = kwargs.get('gamma', None)
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # Compute squared Euclidean distances
        sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        return np.exp(-gamma * sq_dists)
    elif kernel == 'sigmoid':
        alpha = kwargs.get('alpha', 0.01)
        coef0 = kwargs.get('coef0', 0)
        return np.tanh(alpha * np.dot(X, X.T) + coef0)
    else:
        raise ValueError("Unknown kernel type: " + kernel)


def poly_features_degree2(x):
    """
    Compute explicit polynomial features of degree 2 for a 3D vector x.
    For x = [x1, x2, x3] (assumed nonnegative), we return:
      [ x1^2, x2^2, x3^2, sqrt(2)*x1*x2, sqrt(2)*x1*x3, sqrt(2)*x2*x3 ]
    """
    x = np.array(x)
    return np.array([
        x[0]**2,
        x[1]**2,
        x[2]**2,
        np.sqrt(2)*x[0]*x[1],
        np.sqrt(2)*x[0]*x[2],
        np.sqrt(2)*x[1]*x[2]
    ])

