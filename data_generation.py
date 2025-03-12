import numpy as np 

#ToDo: Ensure that there are no dusplicates of the anchors due to the sparsity induced on H
#ToDo: Figure out data generaiton process for separable matrices in RKHS
def generate_separable_data(n_anchors, n_samples, dimension, seed=1, kernel=None, sparsity_on_H=0.75):
    """
    Generate separable data in the RKHS using anchor vectors.
    
    Parameters:
        n_anchors     : Number of anchor vectors.
        n_observations: Number of data points (columns).
        dimension     : Dimension of the original space.
        seed          : Random seed for reproducibility.
        kernel        : Kernel function to use (default: None).
    Returns:
        anchors_orig : Anchor vectors in the original space.
        M_simulated  : Simulated data matrix in the RKHS.
    """
    np.random.seed(seed)
    
    # Generate nonnegative anchor vectors in the original space
    anchors_orig = np.random.rand(dimension, n_anchors)
    
    if kernel is None:
        A = anchors_orig
    else:
        # Compute the explicit mapping for each anchor to get the anchor matrix in the feature space.
        # ToDO: ensure this is the correct approach
        A = np.zeros((kernel.intrinsic_dimension, n_anchors))  # ToDo: code intrinsic dimension from kernel
        for i in range(n_anchors):
            A[:, i] = kernel.project(anchors_orig[:, i])

    # Generate H with desired sparsity
    H = (np.random.rand(n_anchors, n_samples) < (1 - sparsity_on_H)) * np.random.rand(n_anchors, n_samples)
    H[:, :n_anchors] = np.eye(n_anchors)  # force first r columns to be pure anchor contributions

    # Normalize columns of H (e.g. to sum to one)
    H = H / (np.sum(H, axis=0, keepdims=True) + 1e-10)

    # Form the simulated data matrix
    M_simulated = anchors_orig @ H
    #M_simulated = np.maximum(M_simulated, 0)  # ensure nonnegativity # ToDo: RaiseError
    
    return M_simulated, anchors_orig, H
