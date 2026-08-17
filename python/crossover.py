import numpy as np

def crossover(parent1, parent2, n_class):
    """
    Perform crossover operation between two parents
    
    Parameters:
    -----------
    parent1 : numpy.ndarray
        First parent with shape (3, n_class, features)
    parent2 : numpy.ndarray
        Second parent with shape (3, n_class, features)
    n_class : int
        Number of classes
        
    Returns:
    --------
    child1, child2 : numpy.ndarray
        Two offspring from crossover
    """
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    
    for i in range(n_class):
        for j in range(parent1.shape[2]):
            alpha = np.random.rand()
            child1[:, i, j] = alpha * parent1[:, i, j] + (1 - alpha) * parent2[:, i, j]
            child2[:, i, j] = alpha * parent2[:, i, j] + (1 - alpha) * parent1[:, i, j]
    
    return child1, child2
