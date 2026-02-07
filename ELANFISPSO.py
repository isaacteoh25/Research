import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from getFitness import get_fitness
from getPopulation import get_population
from PlotResult import plot_result

# Start timer
start_time = time.time()

# Load dataset
data = pd.read_csv('routput.csv').values
input_data = data[:, :-1]
target = data[:, -1]

# Parameter initialization
center, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    input_data.T, 3, 2, error=1e-5, maxiter=100
)  # center = center cluster, u = membership level

samples, features = input_data.shape
n_class = 3  # [can be change]
ep = 0
ep_max = 400  # [can be change]

# Population initialization
n_pop = 5
vel = np.zeros((n_pop, 3, n_class, features))  # velocity matrix of an iteration

c1 = 1.2
c2 = 1.2

pop = get_population(n_class, input_data, n_pop)
output = 0

# PSO
# initialize pBest
best_mse = np.full(n_pop, 100.0)
p_best_pos = np.zeros((n_pop, 3, n_class, features))

# calculate fitness function
p_best_output = np.zeros((len(target), n_pop))
for i in range(n_pop):
    pop_position = pop[i, :, :, :]
    mse, output = get_fitness(pop_position, n_class, input_data, target)
    if mse < best_mse[i]:
        best_mse[i] = mse
        p_best_pos[i, :, :, :] = pop_position
        p_best_output[:, i] = output

# PSO
# find gBest
idx = np.argmin(best_mse)
best_sol = best_mse[idx]
g_best_pos = p_best_pos[idx, :, :, :]

Et = []

# ITERATION
while ep < ep_max:
    ep += 1
    
    # calculate velocity and update particle
    # vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
    # pi(t + 1) = pi(t) + vi(t + 1)
    r1 = np.random.rand()
    r2 = np.random.rand()
    
    for i in range(n_pop):
        vel[i, :, :, :] = vel[i, :, :, :] + (c1 * r1) * (p_best_pos[i, :, :, :] - pop[i, :, :, :]) + (c2 * r2) * (g_best_pos - pop[i, :, :, :])
        pop[i, :, :, :] = pop[i, :, :, :] + vel[i, :, :, :]
    
    # calculate fitness value and update pBest
    for i in range(n_pop):
        pop_position = pop[i, :, :, :]
        mse, output = get_fitness(pop_position, n_class, input_data, target)
        if mse < best_mse[i]:
            best_mse[i] = mse
            p_best_pos[i, :, :, :] = pop_position
            p_best_output[:, i] = output
    
    # find gBest
    idx = np.argmin(best_mse)
    best_sol = best_mse[idx]
    g_best_pos = p_best_pos[idx, :, :, :]
    best_output = p_best_output[:, idx]
    
    Et.append(best_sol)
    
    # Draw the SSE plot
    plt.plot(range(1, ep+1), Et)
    plt.title(f'Epoch {ep} -> MSE = {Et[ep-1]}')
    plt.grid(True)
    plt.pause(0.001)
    print(f'Iteration {ep}: Best Cost = {Et[ep-1]}')

plt.figure()

elapsed_time = time.time() - start_time

plot_result(target, best_output, 'ELANFIS-PSO', elapsed_time)
