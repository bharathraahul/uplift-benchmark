import pandas as pd
import numpy as np
import sys
#print(sys.executable)
df = pd.read_csv("data/raw/criteo-uplift-v2.1.csv")

df = df.dropna()

df.to_parquet("data/processed/criteo-uplift-v2.1.parquet", index=False)
df1 = pd.read_parquet("data/processed/criteo-uplift-v2.1.parquet")

for i in range(12):
    df1[f'feature_{i}'] = df1[f'feature_{i}'].astype('float32')

df1['treatment'] = df1['treatment'].astype('int16')
df1['outcome'] = df1['outcome'].astype('int16')
df1['day'] = df1['day'].astype('int16')
df1['hour'] = df1['hour'].astype('int16')

df2 = df1.sample(frac=1, random_state=42).reset_index(drop=True)

