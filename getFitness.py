import numpy as np

def get_fitness(premise, n_class, input_data, target):
    """
    Calculate fitness value for each premise parameter (a, b, c) of one particle
    
    Parameters:
    -----------
    premise : numpy.ndarray
        Premise parameter, size: 3 (a, b, and c parameter) x n_class x features
    n_class : int
        Total classes
    input_data : numpy.ndarray
        Input data, samples x features
    target : numpy.ndarray
        Target values
        
    Returns:
    --------
    mse : float
        Mean squared error (fitness value)
    output : numpy.ndarray
        Predicted output
    """
    samples, features = input_data.shape
    H = []
    Mu = np.zeros((samples, n_class, features))
    
    # Calculate membership grades
    # forward pass
    for i in range(samples):
        # Calculate firing strength
        w1 = np.ones(n_class)
        mu = np.zeros((n_class, features))
        
        for k in range(n_class):
            for j in range(features):
                # mu: miu of one sample
                mu[k, j] = 1 / (1 + ((input_data[i, j] - premise[2, k, j]) / premise[0, k, j]) ** (2 * premise[1, k, j]))
                w1[k] *= mu[k, j]  # fill w of k-th class
                Mu[i, k, j] = mu[k, j]
        
        # Calculate Normalised Firing
        w = w1 / np.sum(w1)  # w = w bar of one row / one sample data
        XZ = []
        
        # Generate X of f=XZ
        for k in range(n_class):
            XZ.extend(w[k] * input_data[i, :])
            XZ.append(w[k])
        
        H.append(XZ)
    
    H = np.array(H)
    
    # find consequent parameter (p, q, r)
    beta = np.linalg.pinv(H) @ target  # moore pseudo inverse
    output = H @ beta  # calculate weight to output
    error = target - output
    mse = np.mean(error ** 2)  # calculate MSE
    
    return mse, output
