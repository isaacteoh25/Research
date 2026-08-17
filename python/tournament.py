import numpy as np

def tournament(best_mse, children, n_pop, n_change):
    """
    Tournament selection for genetic algorithm
    
    Parameters:
    -----------
    best_mse : numpy.ndarray
        Array of best MSE values for each individual
    children : numpy.ndarray
        Population array
    n_pop : int
        Population size
    n_change : int
        Number of individuals to select
        
    Returns:
    --------
    parent1, parent2 : numpy.ndarray
        Two selected parents
    """
    # Random permutation
    s = np.random.permutation(n_pop)[:n_change]
    parents = children[s, :, :, :]
    
    # Reshape and find minimum indices
    mse_reshaped = best_mse[s].reshape(2, 2)
    ind = np.argmin(mse_reshaped, axis=0)
    
    parent1 = parents[ind[0], :, :, :]
    parent2 = parents[ind[1] + 2, :, :, :]
    
    return parent1, parent2
