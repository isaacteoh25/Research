import numpy as np
import skfuzzy as fuzz

def get_population(n_class, input_data, n_pop):
    """
    Generate initial population for optimization
    
    Parameters:
    -----------
    n_class : int
        Number of classes
    input_data : numpy.ndarray
        Input data, samples x features
    n_pop : int
        Population size
        
    Returns:
    --------
    population : numpy.ndarray
        Initial population with shape (n_pop, 3, n_class, features)
    """
    center, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        input_data.T, 3, 2, error=1e-5, maxiter=100
    )  # center = center cluster, u = membership level
    
    hi = np.argmax(u, axis=0)  # the class corresponding to the max value
    samples, features = input_data.shape
    population = np.zeros((n_pop, 3, n_class, features))
    
    for loop in range(n_pop):
        a = np.zeros((n_class, features))
        b = np.full((n_class, features), 2)
        c = np.zeros((n_class, features))
        
        for k in range(n_class):
            for i in range(features):
                # premise parameter: a
                Ri = np.max(input_data[:, i]) - np.min(input_data[:, i])
                r = np.sum(hi == k)
                a_temp = Ri / (2 * r - 2) if r > 1 else Ri / 2
                a_lower = a_temp * 0.5
                a_upper = a_temp * 1.5
                a[k, i] = np.random.uniform(a_lower, a_upper)
                
                # premise parameter: c
                dcc = np.random.uniform(1.9, 2.1)
                c_lower = center[k, features-1] - dcc / 2
                c_upper = center[k, features-1] + dcc / 2
                c[k, i] = np.random.uniform(c_lower, c_upper)
        
        population[loop, 0, :, :] = a
        population[loop, 1, :, :] = b
        population[loop, 2, :, :] = c
    
    return population
