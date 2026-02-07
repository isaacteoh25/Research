# MATLAB to Python Conversion

This repository contains Python implementations of MATLAB code for ANFIS (Adaptive Neuro-Fuzzy Inference System) with various optimization algorithms.

## Files Converted

### Main Scripts
- **ANFIS.py** - Basic ANFIS implementation
- **ELANFIS.py** - Extended Learning ANFIS
- **ELANFISMmGA.py** - ELANFIS with Modified Micro Genetic Algorithm
- **ELANFISPSO.py** - ELANFIS with Particle Swarm Optimization

### Supporting Modules
- **CreateInitialFIS.py** - Initialize Fuzzy Inference System
- **crossover.py** - Genetic algorithm crossover operation
- **getFitness.py** - Fitness function calculation
- **getPopulation.py** - Population initialization
- **PlotResult.py** - Results visualization and export
- **tournament.py** - Tournament selection for GA

## Installation

1. Install Python 3.8 or higher
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run any of the main scripts:

```bash
# Basic ANFIS
python ANFIS.py

# ELANFIS
python ELANFIS.py

# ELANFIS with Modified Micro GA
python ELANFISMmGA.py

# ELANFIS with PSO
python ELANFISPSO.py
```

## Dependencies

- **numpy** - Numerical computations
- **pandas** - Data handling
- **matplotlib** - Plotting and visualization
- **scipy** - Scientific computing
- **scikit-fuzzy** - Fuzzy logic operations (replaces MATLAB's Fuzzy Logic Toolbox)
- **scikit-learn** - Machine learning utilities

## Data

The scripts expect a CSV file named `routput.csv` in the same directory. The CSV should have:
- Input features in all columns except the last one
- Target values in the last column
- Header row (will be skipped)

## Key Differences from MATLAB

1. **Array Indexing**: Python uses 0-based indexing (MATLAB uses 1-based)
2. **Matrix Operations**: Uses `@` operator or `np.dot()` instead of `*`
3. **Fuzzy Logic**: Uses `scikit-fuzzy` library instead of MATLAB's Fuzzy Logic Toolbox
4. **Plotting**: Uses `matplotlib` instead of MATLAB's plotting functions
5. **File I/O**: Uses `pandas` for CSV operations instead of `csvread`

## Output

Each script generates:
- Visualization plots showing target vs output, errors, and error distribution
- CSV file with target and output values
- Console output with iteration progress and MSE values

## Notes

- The FCM (Fuzzy C-Means) clustering is implemented using `scikit-fuzzy.cluster.cmeans`
- Some MATLAB-specific functions (like `evalfis`, `genfis3`) have been replaced with custom implementations or equivalent operations
- Timing is handled using Python's `time` module instead of MATLAB's `tic/toc`
