import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
#print(sys.executable)
df = pd.read_csv("data/raw/criteo-uplift-v2.1.csv")

df = df.dropna()

df.to_parquet("data/processed/criteo-uplift-v2.1.parquet", index=False)
df1 = pd.read_parquet("data/processed/criteo-uplift-v2.1.parquet")

for i in range(12):
    df1[f'f{i}'] = df1[f'f{i}'].astype('float32')

df1['treatment'] = df1['treatment'].astype('int16')
#df1['outcome'] = df1['outcome'].astype('int16')
#df1['day'] = df1['day'].astype('int16')
#df1['hour'] = df1['hour'].astype('int16')

df2 = df1.sample(frac=1, random_state=42).reset_index(drop=True)


plt.plot(df2['f0'])
plt.title('Feature 0 Distribution')
plt.xlabel('Index')
plt.ylabel('Feature 0 Value')
plt.show()

q = np.linspace(0, 1, 1001)
f0_quantiles = df2['f0'].quantile(q)

plt.figure(figsize=(10, 5))
for i in range(12):
    feature_quantiles =df2[f'f{i}'].quantile(q)
    plt.subplot(3,4,i+1)
    plt.plot(q * 100, feature_quantiles.values, linewidth=1, alpha=0.7, label=f'f{i}')
    plt.title(f'f{i}')
#plt.plot(q * 100, f0_quantiles.values, linewidth=2, color='black', label='f0', alpha=0.9)
plt.title('Feature Quantiles (sorted values)')
plt.xlabel('Percentile of rows (%)')
plt.ylabel('Feature Value')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()  


