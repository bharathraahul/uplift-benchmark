import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
#print(sys.executable)
parquet_path = "data/processed/criteo-uplift-v2.1.parquet"

# Make the Parquet file from the CSV once. Later runs just load the Parquet.
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

# Shuffle all rows. Use a smaller frac to work with fewer rows.
df_shuffled = df1.sample(frac=1, random_state=42).reset_index(drop=True)


q = np.linspace(0, 1, 1001)

plt.figure(figsize=(10, 5))
for i in range(12):
    feature_quantiles =df_shuffled[f'f{i}'].quantile(q)
    plt.subplot(3,4,i+1)
    plt.plot(q * 100, feature_quantiles.values, linewidth=1, alpha=0.7, label=f'f{i}')
    plt.title(f'f{i}')
plt.suptitle('Feature Quantiles (sorted values)')
plt.xlabel('Percentile of rows (%)')
plt.ylabel('Feature Value')
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


relative_lift_visit = ate_visit / visit_mean_untreated
relative_lift_conversion = ate_conversion / conversion_mean_untreated


# Bootstrap 95% CI for the conversion ATE, using 1M random users to keep it fast.
# Resample users within each group, so each group keeps its size.
boot_sample = df_shuffled.sample(n=1_000_000, random_state=42)
conversion_ad = boot_sample[boot_sample['treatment']==1]['conversion'].to_numpy()
conversion_no_ad = boot_sample[boot_sample['treatment']==0]['conversion'].to_numpy()

rng = np.random.default_rng(42)
n_boot = 1000
boot_ates = np.empty(n_boot)
for b in range(n_boot):
    resampled_ad = rng.choice(conversion_ad, size=len(conversion_ad), replace=True)
    resampled_no_ad = rng.choice(conversion_no_ad, size=len(conversion_no_ad), replace=True)
    boot_ates[b] = resampled_ad.mean() - resampled_no_ad.mean()

boot_low, boot_high = np.percentile(boot_ates, [2.5, 97.5])

# Normal 95% CI on the same 1M users, to compare.
ate_conversion_sub = conversion_ad.mean() - conversion_no_ad.mean()
SE_conversion_sub = np.sqrt(conversion_ad.var(ddof=1)/len(conversion_ad) + conversion_no_ad.var(ddof=1)/len(conversion_no_ad))

print(f'ATE for conversion (1M subsample): {ate_conversion_sub:.6f}')
print(f'Bootstrap 95% CI ({n_boot} resamples): [{boot_low:.6f}, {boot_high:.6f}]')
print(f'Normal 95% CI (same subsample):    [{ate_conversion_sub - 1.96*SE_conversion_sub:.6f}, {ate_conversion_sub + 1.96*SE_conversion_sub:.6f}]')


# Check: ATE after adjusting for features f0..f11 (linear regression).
# If it is close to the simple ATE, the small feature differences don't matter.
X = sm.add_constant(df_shuffled[['treatment'] + [f'f{i}' for i in range(12)]].astype('float64'))
adjusted_model = sm.OLS(df_shuffled['conversion'].astype('float64'), X).fit(cov_type='HC1')
ate_conversion_adjusted = adjusted_model.params['treatment']
adjusted_low, adjusted_high = adjusted_model.conf_int().loc['treatment']

print(f'Simple ATE for conversion:   {ate_conversion:.6f}')
print(f'Adjusted ATE for conversion: {ate_conversion_adjusted:.6f}, 95% CI [{adjusted_low:.6f}, {adjusted_high:.6f}]')
print(f'Adjusted vs simple: {(ate_conversion_adjusted - ate_conversion) / ate_conversion * 100:+.1f}%')


# Results table: one row per outcome.
# Reuse the conversion model from above; only visit needs a new fit.
results = []
for outcome in ['visit', 'conversion']:
    ad = df_shuffled[df_shuffled['treatment']==1][outcome]
    no_ad = df_shuffled[df_shuffled['treatment']==0][outcome]
    ate = ad.mean() - no_ad.mean()
    se = np.sqrt(ad.var()/len(ad) + no_ad.var()/len(no_ad))
    if outcome == 'conversion':
        adjusted_ate = ate_conversion_adjusted
    else:
        adjusted_ate = sm.OLS(df_shuffled[outcome].astype('float64'), X).fit(cov_type='HC1').params['treatment']
    results.append({
        'outcome': outcome,
        'no-ad rate': no_ad.mean(),
        'ad rate': ad.mean(),
        'ATE': ate,
        '95% CI': f'[{ate - 1.96*se:.6f}, {ate + 1.96*se:.6f}]',
        'relative lift': f'{ate / no_ad.mean():.1%}',
        'adjusted ATE': adjusted_ate,
    })
results_table = pd.DataFrame(results).set_index('outcome')
print(results_table.to_string(float_format='{:.6f}'.format))
results_table.to_csv('figures/phase2_results.csv')


# Conversion rate in each feature's deciles, for the ad and no-ad groups.
# Deciles use rank, so every decile has the same number of rows.
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for i, ax in enumerate(axes.flat):
    bins = pd.qcut(df_shuffled[f'f{i}'].rank(method='first'), 10, labels=False)
    rates = df_shuffled.groupby([bins, 'treatment'])['conversion'].mean().unstack()
    ax.plot(range(len(rates)), rates[1] * 100, marker='o', label='Ad group')
    ax.plot(range(len(rates)), rates[0] * 100, marker='o', label='No-ad group')
    ax.set_title(f'f{i}')
    ax.set_xlabel('Feature decile (low to high)')
    ax.set_ylabel('Conversion rate (%)')
    ax.grid(alpha=0.3)
axes[0, 0].legend()
fig.suptitle('Conversion rate by feature decile: Ad group vs No-ad group')
plt.tight_layout()
plt.savefig('figures/feature_conversion_by_treatment.png', dpi=150)
plt.show()


