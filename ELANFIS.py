import time
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from PlotResult import plot_result

# Start timer
start_time = time.time()

# Load dataset
data = pd.read_csv('routput.csv').values
input_data = data[:, :-1]
target = data[:, -1]

# Initialize parameter
center, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    input_data.T, 3, 2, error=1e-3, maxiter=100
)  # center = center cluster, u = membership level

samples, features = input_data.shape
n_class = 3  # total classes
ep = 0
ep_max = 1
yy, hi = np.argmax(u, axis=0), np.argmax(u, axis=0)  # hi = the class corresponding to the max value

best_mse = 1000
best_a = np.zeros((n_class, features))
best_b = np.full((n_class, features), 2)
best_c = np.zeros((n_class, features))
best_output = None

while ep < ep_max:
    ep += 1
    a = np.zeros((n_class, features))
    b = np.zeros((n_class, features))
    c = np.zeros((n_class, features))
    
    # Estimating random mf parameters
    for k in range(n_class):
        for i in range(features):
            # premise parameter: a
            Rj = np.max(input_data[:, i]) - np.min(input_data[:, i])
            m = np.sum(hi == k)
            a_temp = Rj / (2 * m - 2) if m > 1 else Rj / 2
            a_lower = a_temp * 0.5
            a_upper = a_temp * 1.5
            a[k, i] = np.random.uniform(a_lower, a_upper)
            
            # premise parameter: c
            dcc = np.random.uniform(1.9, 2.1)
            c_lower = center[k, features-1] - dcc / 2
            c_upper = center[k, features-1] + dcc / 2
            c[k, i] = np.random.uniform(c_lower, c_upper)
    
    H = []
    Mu = np.zeros((samples, n_class, features))
    
    # Calculate membership grades
    for i in range(samples):
        # Calculate firing strength
        w1 = np.ones(n_class)
        mu = np.zeros((n_class, features))
        
        for k in range(n_class):
            for j in range(features):
                mu[k, j] = 1 / (1 + ((input_data[i, j] - c[k, j]) / a[k, j]) ** (2 * b[k, j]))
                w1[k] *= mu[k, j]
                Mu[i, k, j] = mu[k, j]
        
        # Calculate Normalised Firing
        w = w1 / np.sum(w1)
        ZX = []
        
        # Generate X of f=XZ
        for k in range(n_class):
            ZX.extend(w[k] * input_data[i, :])
            ZX.append(w[k])
        
        H.append(ZX)
    
    H = np.array(H)
    
    # consequent parameter (p, q, r)
    beta = np.linalg.pinv(H) @ target  # moore pseudo inverse
    
    output = H @ beta  # calculate output from weight
    
    error = target - output
    mse = np.mean(error ** 2)  # calculate MSE
    
    if mse < best_mse:  # update min error
        best_mse = mse
        best_output = output
        best_a = a
        best_b = b
        best_c = c
    
    print(f'Iteration {ep}: Best Cost = {best_mse}')

# Create FIS output structure
fis = {'output': best_output}

elapsed_time = time.time() - start_time
plot_result(target, fis['output'], 'ELANFIS', elapsed_time)
