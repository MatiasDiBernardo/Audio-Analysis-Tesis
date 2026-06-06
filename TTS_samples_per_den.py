import numpy as np
from data_mos import config_TTS_MOS_es
import pandas as pd

# 1. Convert dictionary to a DataFrame
# We split the key to extract the algorithm name
data = []
for key, (mean_mos, std_mos) in config_TTS_MOS_es.items():
    algo = key.split('_')[0]
    data.append({'Algorithm': algo, 'Mean MOS': mean_mos, 'Std MOS': std_mos})

df = pd.DataFrame(data)

# 2. Group by Algorithm and calculate the mean
# We use .agg to handle both columns at once
avg_results = df.groupby('Algorithm').agg({
    'Mean MOS': 'mean',
    'Std MOS': 'mean'
}).round(3)

# 3. Print the results
print("--- Average MOS per Denoising Algorithm ---")
print(avg_results)