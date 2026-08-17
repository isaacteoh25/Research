import time
import numpy as np
import pandas as pd
from CreateInitialFIS import create_initial_fis
from PlotResult import plot_result
import skfuzzy as fuzz

# Start timer
start_time = time.time()

# Load Data
# load dataset
loaddata = pd.read_csv('routput.csv').values

Inputs = loaddata[:, :-1]
Targets = loaddata[:, -1]

data = {
    'Inputs': Inputs,
    'Targets': Targets
}

# Generate Basic FIS
fis = create_initial_fis(data, 3)

# Evaluate FIS
# Note: evalfis equivalent would need full FIS implementation
# For now, using a placeholder
output = fis.predict(data['Inputs'])

elapsed_time = time.time() - start_time
plot_result(data['Targets'], output, 'ANFIS', elapsed_time)
