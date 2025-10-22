import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

#============================Configurations============================
# Specify parameters as: {param_name: [start, end, num_intervals]}
param_specs = {
    "gait_stride": [0.75, 1.25, 15], # b20
    "gait_cadence": [0.75, 1.25, 15], # b20
}

PARAM_NAME = 'gait'
SAVE_PARQUET = False
#============================Sampling============================
param_names = list(param_specs.keys())
param_ranges = []
for spec in param_specs.values():
    if len(spec) == 1:
        # Single value case
        param_ranges.append([spec[0]])
    else:
        # Range case: [start, end, num]
        start, end, num = spec
        param_ranges.append(np.linspace(start, end, num))

# Create grid of all combinations
all_combinations = list(product(*param_ranges))

# Build DataFrame
param_idx = np.arange(len(all_combinations), dtype=np.int32)
data = np.column_stack([param_idx, np.array(all_combinations)])
columns = ['param_idx'] + param_names
df = pd.DataFrame(data, columns=columns)

# Convert float64 to float32 for all except param_idx
for col in param_names:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')

num_sample = len(df)

PARAM_PATH = f'data/rolloutParam/{PARAM_NAME}_U{num_sample}'

#============================Save============================
if SAVE_PARQUET:
    parquet_path = Path(PARAM_PATH).with_suffix('.parquet')
    df.to_parquet(parquet_path, index=False)
    print(f"Uniform sampled parameters saved to {parquet_path}")
else:
    csv_path = Path(PARAM_PATH).with_suffix('.csv')
    df.to_csv(csv_path, index=False, sep=',', float_format='%.4f')
    print(f"Uniform sampled parameters saved to {csv_path}") 