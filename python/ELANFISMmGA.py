import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from getFitness import get_fitness
from getPopulation import get_population
from tournament import tournament
from crossover import crossover
from PlotResult import plot_result

# Start timer
start_time = time.time()

# Load dataset
data = pd.read_csv('routput.csv').values
input_data = data[:, :-1]
target = data[:, -1]

# Initialize parameter
samples, features = input_data.shape
n_class = 3  # [can be change]
ep = 1
ep_max = 400  # [can be change]

n_pop = 5

parents = np.zeros((n_pop, 3, n_class, features))

pop = get_population(n_class, input_data, n_pop)

# mGA
count = 0  # generations count with same fitness
output = 0
m0 = 5  # generations num in which mean remains constant
best_mse = np.full(n_pop, 100.0)
mean_log = np.zeros(ep_max + 1)
mean_log[0] = 0
n_change = 4  # changing population num

vel = np.zeros((n_pop, 3, n_class, features))
c1 = 1.2
c2 = 1.2

Et = np.zeros(ep_max + 1)  # Initialize Et array

# Initialize PBestpos outside the loop
p_best_pos = np.zeros((n_pop, 3, n_class, features))
p_best_output = np.zeros((len(target), n_pop))

while ep < ep_max and count < ep_max // 2:
    
    # calculate fitness function
    for i in range(n_pop):
        pop_position = pop[i, :, :, :]
        mse, output = get_fitness(pop_position, n_class, input_data, target)
        if mse < best_mse[i]:
            best_mse[i] = mse
            p_best_pos[i, :, :, :] = pop_position
            p_best_output[:, i] = output
    
    # mGA
    # find gBest
    idx = np.argmin(best_mse)
    best_sol = best_mse[idx]
    g_best_pos = p_best_pos[idx, :, :, :]
    best_parent = pop[idx, :, :, :]
    best_output = p_best_output[:, idx]
    
    mean_v = np.sum(best_mse) / n_pop  # Mean of the cost function
    
    # CONVERGENCE CHECK
    if ep > m0:
        cur_mean = np.sum(mean_log[ep-m0:ep]) / m0
    else:
        cur_mean = np.sum(mean_log[:ep]) / ep
    
    if abs(abs(cur_mean) - abs(mean_v)) <= 0.05 * abs(mean_v) or ep % 10 == 0:
        # Estimate random mf parameters
        pop[:n_change, :, :, :] = get_population(n_class, input_data, n_change)
    
    r1 = np.random.rand()
    r2 = np.random.rand()
    
    for k in range(0, n_change, 2):
        parents[k, :, :, :], parents[k+1, :, :, :] = tournament(best_mse, pop, n_pop, n_change)
        
        pop[k, :, :, :], pop[k+1, :, :, :] = crossover(parents[k, :, :, :], parents[k+1, :, :, :], n_class)
        
        # vi(t + 1) = wvi(t) + c1r1(pbi(t) - pi(t)) + c2r2(pg(t) - pi(t))
        # pi(t + 1) = pi(t) + vi(t + 1)
        vel[k, :, :, :] = vel[k, :, :, :] + (c1 * r1) * (p_best_pos[k, :, :, :] - pop[k, :, :, :]) + (c2 * r2) * (g_best_pos - pop[k, :, :, :])
        vel[k+1, :, :, :] = vel[k+1, :, :, :] + (c1 * r1) * (p_best_pos[k+1, :, :, :] - pop[k+1, :, :, :]) + (c2 * r2) * (g_best_pos - pop[k+1, :, :, :])
        pop[k, :, :, :] = pop[k, :, :, :] + vel[k, :, :, :]
        pop[k+1, :, :, :] = pop[k+1, :, :, :] + vel[k+1, :, :, :]
    
    pop[n_pop-1, :, :, :] = best_parent
    
    ep += 1
    Et[ep] = best_sol
    mean_log[ep] = mean_v
    
    if abs(Et[ep] - Et[ep-1]) <= 0.001 * abs(Et[ep]):
        count += 1
    else:
        count = 0
    
    plt.plot(range(1, ep+1), Et[1:ep+1])
    plt.title(f'Epoch {ep} -> MSE = {Et[ep]:.6f}')
    plt.grid(True)
    plt.pause(0.001)
    print(f'Iteration {ep}: Best Cost = {Et[ep]:.6f}')

plt.figure()

elapsed_time = time.time() - start_time

plot_result(target, best_output, 'ELANFIS-MmGA', elapsed_time)
