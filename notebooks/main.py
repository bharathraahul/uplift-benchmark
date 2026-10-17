import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
#print(sys.executable)
parquet_path = "data/processed/criteo-uplift-v2.1.parquet"

# Build the downcast Parquet once from the CSV; every later run loads only the Parquet.
if not os.path.exists(parquet_path):
    df = pd.read_csv("data/raw/criteo-uplift-v2.1.csv")

    print(f'Rows with missing values (dropped): {df.isna().any(axis=1).sum()} of {len(df)}')
    df = df.dropna()

    for i in range(12):
        df[f'f{i}'] = df[f'f{i}'].astype('float32')

    df['treatment'] = df['treatment'].astype('int16')
    df['visit'] = df['visit'].astype('int8')
    df['conversion'] = df['conversion'].astype('int8')
    df['exposure'] = df['exposure'].astype('int16')

    df.to_parquet(parquet_path, index=False)

df1 = pd.read_parquet(parquet_path)

# Full shuffle, not a subsample: frac=1 keeps every row. Lower frac if things get slow.
df_shuffled = df1.sample(frac=1, random_state=42).reset_index(drop=True)


plt.plot(df_shuffled['f0'])
plt.title('Feature 0 Distribution')
plt.xlabel('Index')
plt.ylabel('Feature 0 Value')
plt.show()

q = np.linspace(0, 1, 1001)
f0_quantiles = df_shuffled['f0'].quantile(q)

plt.figure(figsize=(10, 5))
for i in range(12):
    feature_quantiles =df_shuffled[f'f{i}'].quantile(q)
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


plt.figure(figsize=(10, 5))
treated_positive = df_shuffled[df_shuffled['treatment']==1]['f0'].quantile(q)
untreated_positive = df_shuffled[df_shuffled['treatment']==0]['f0'].quantile(q)
plt.plot(q * 100, treated_positive.values, linewidth=2, color='blue', label='Ad group', alpha=0.9)
plt.plot(q * 100, untreated_positive.values, linewidth=2, color='orange', label='No-ad group', alpha=0.9)
plt.title('Feature 0 Quantiles by Treatment Group')
plt.xlabel('Percentile of rows (%)')
plt.ylabel('Feature 0 Value')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 5))
for i in range(12):
    SMD = (df_shuffled[df_shuffled['treatment']==1][f'f{i}'].mean() - df_shuffled[df_shuffled['treatment']==0][f'f{i}'].mean()) / np.sqrt((df_shuffled[df_shuffled['treatment']==1][f'f{i}'].var() + df_shuffled[df_shuffled['treatment']==0][f'f{i}'].var()) / 2)
    plt.bar(f'f{i}', SMD, color='blue', alpha=0.9)
    print(f'SMD for f{i}: {SMD:.4f}')
plt.axhline(0.1, color='red', linestyle='--', linewidth=1)
plt.axhline(-0.1, color='red', linestyle='--', linewidth=1)
plt.title('SMD by Feature: Ad group vs No-ad group')
plt.xlabel('Feature')
plt.ylabel('SMD Value')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/smd_balance.png', dpi=150)
plt.show()


visit_mean_treated = df_shuffled[df_shuffled['treatment']==1]['visit'].mean()
visit_mean_untreated = df_shuffled[df_shuffled['treatment']==0]['visit'].mean()
print(f'Mean visit for Ad group: {visit_mean_treated:.6f}')
print(f'Mean visit for No-ad group: {visit_mean_untreated:.6f}')

ate_visit = visit_mean_treated - visit_mean_untreated
print(f'ATE for visit: {ate_visit:.6f}')

conversion_mean_treated = df_shuffled[df_shuffled['treatment']==1]['conversion'].mean()
conversion_mean_untreated = df_shuffled[df_shuffled['treatment']==0]['conversion'].mean()
print(f'Mean conversion for Ad group: {conversion_mean_treated:.6f}')
print(f'Mean conversion for No-ad group: {conversion_mean_untreated:.6f}')

ate_conversion = conversion_mean_treated - conversion_mean_untreated
print(f'ATE for conversion: {ate_conversion:.6f}')  


exposure_mean_treated = df_shuffled[df_shuffled['treatment']==1]['exposure'].mean()
exposure_mean_untreated = df_shuffled[df_shuffled['treatment']==0]['exposure'].mean()
print(f'Mean exposure for Ad group: {exposure_mean_treated:.6f}')
print(f'Mean exposure for No-ad group: {exposure_mean_untreated:.6f}')

exposure_rate_diff = exposure_mean_treated - exposure_mean_untreated
print(f'Exposure rate difference (share of Ad group that saw an ad; not a treatment effect): {exposure_rate_diff:.6f}')

treatment_share = (df_shuffled['treatment'] == 1).mean()
print(f'Share of Ad group: {treatment_share:.4f}')

exposure_zero_no_ad = df_shuffled[df_shuffled['treatment']==0]['exposure'].value_counts()
print(f'Exposure distribution for No-ad group:\n{exposure_zero_no_ad}')


SE = np.sqrt((df_shuffled[df_shuffled['treatment']==1]['visit'].var()/df_shuffled[df_shuffled['treatment']==1].shape[0] + (df_shuffled[df_shuffled['treatment']==0]['visit'].var()/df_shuffled[df_shuffled['treatment']==0].shape[0])))
