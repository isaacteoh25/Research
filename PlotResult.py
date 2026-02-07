import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import csv

def plot_result(target, output, name, elapsed_time):
    """
    Plot the results comparing target and output
    
    Parameters:
    -----------
    target : numpy.ndarray
        Target values
    output : numpy.ndarray
        Predicted output values
    name : str
        Name for the plot title and output file
    elapsed_time : float
        Execution time in seconds
    """
    # Calculate error metrics
    error = target - output
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    error_mean = np.mean(error)
    error_std = np.std(error)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 8))
    
    # Subplot 1: Target vs Output
    ax1 = plt.subplot(2, 2, (1, 2))
    plt.plot(target, 'b', label='Target')
    plt.plot(output, 'r', label='Output')
    plt.legend()
    times = round(elapsed_time, 2)
    plt.title(f'{name}, time={times}s')
    plt.xlabel('Sample Index')
    plt.grid(True)
    
    # Subplot 2: Error plot
    ax2 = plt.subplot(2, 2, 3)
    plt.plot(error)
    plt.legend(['Error'])
    plt.title(f'RMSE = {rmse:.4f}, MSE = {mse:.4f}')
    plt.grid(True)
    
    # Subplot 3: Error histogram with fit
    ax3 = plt.subplot(2, 2, 4)
    n, bins, patches = plt.hist(error, bins=50, density=True, alpha=0.7, color='blue')
    
    # Fit a normal distribution
    mu, std = norm.fit(error)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r-', linewidth=2)
    
    plt.title(f'Error StD. = {error_std:.4f}, Error Mean = {error_mean:.4f}')
    
    plt.tight_layout()
    
    # Regression plot (if available)
    try:
        fig2 = plt.figure(figsize=(8, 8))
        plt.scatter(target, output, alpha=0.5)
        plt.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', lw=2)
        plt.xlabel('Target')
        plt.ylabel('Output')
        plt.title('Regression Plot')
        plt.grid(True)
    except Exception as e:
        print(f"Could not create regression plot: {e}")
    
    # Save to CSV
    filename = f'{name}.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Target', 'Output'])
        for t, o in zip(target, output):
            writer.writerow([f'{t:.5f}', f'{o:.5f}'])
    
    plt.show()
