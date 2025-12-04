import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("COMPREHENSIVE ASTHMA CORRELATION ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load data and define neighborhoods by borough
# ============================================================================
print("\n[1/5] Loading data...")
df = pd.read_csv('DATA/CLEANED/FINAL_MERGED_DATASET.csv')
print(f"  ✓ Loaded: {df.shape}")

# Borough definitions
boroughs = {
    'Staten Island': [
        "Northern SI", "Southern SI", "Port Richmond",
        "Stapleton - St. George", "Willowbrook", "South Beach - Tottenville"
    ],
    'Bronx': [
        "Fordham - Bronx Pk", "Kingsbridge - Riverdale", "Northeast Bronx",
        "Pelham - Throgs Neck", "South Bronx", "Crotona - Tremont",
        "High Bridge - Morrisania", "Hunts Point - Mott Haven"
    ],
    'Manhattan': [
        "Central Harlem - Morningside Heights", "Chelsea-Village",
        "Downtown - Heights - Slope", "East Harlem",
        "Union Square-Lower Manhattan", "Upper East Side-Gramercy",
        "Upper West Side", "Washington Heights", "Chelsea - Clinton",
        "Gramercy Park - Murray Hill", "Greenwich Village - SoHo",
        "Lower Manhattan"
    ],
    'Queens': [
        "Bayside Little Neck-Fresh Meadows", "Flushing - Clearview",
        "Jamaica", "Long Island City - Astoria", "Ridgewood - Forest Hills",
        "Rockaways", "Southeast Queens", "Southwest Queens", "West Queens",
        "Fresh Meadows", "Bayside - Little Neck"
    ],
    'Brooklyn': [
        "Bedford Stuyvesant - Crown Heights", "Bensonhurst - Bay Ridge",
        "Borough Park", "Canarsie - Flatlands", "Coney Island - Sheepshead Bay",
        "East Flatbush - Flatbush", "Sunset Park", "Williamsburg - Bushwick",
        "Greenpoint", "East New York"
    ]
}

# Create borough column
def assign_borough(neighborhood):
    for borough, neighborhoods in boroughs.items():
        if neighborhood in neighborhoods:
            return borough
    return 'Unknown'

df['borough'] = df['neighborhood'].apply(assign_borough)

print(f"\n  Borough distribution:")
for borough in boroughs.keys():
    count = (df['borough'] == borough).sum()
    print(f"    {borough}: {count} records")

# ============================================================================
# STEP 2: Prepare asthma outcome variables
# ============================================================================
print("\n[2/5] Preparing asthma outcome variables...")

# We'll use multiple asthma outcomes
asthma_outcomes = {
    'Adult Asthma Prevalence': 'age_adjusted_asthma_percent',
    'Adult ED Visits': 'age_adjusted_ed_rate_per_10k',
    'Child (0-4) ED Visits': 'ed_rate_per_10k_age_0_4',
    'Child (5-17) ED Visits': 'ed_rate_per_10k_age_5_17'
}

print(f"  ✓ Asthma outcomes: {list(asthma_outcomes.keys())}")

# ============================================================================
# STEP 3: Calculate correlations for continuous variables
# ============================================================================
print("\n[3/5] Calculating Pearson correlations (continuous variables)...")

# Continuous variables
continuous_vars = {
    'NO2 (Air Quality)': 'NO2_Avg',
    'PM2.5 (Particulate Matter)': 'PM_Avg',
    'Mold Complaints': 'mold_complaints',
    'Poverty Rate': 'poverty_rate'
}

# Create results dataframe
results = []

for outcome_name, outcome_col in asthma_outcomes.items():
    print(f"\n  {outcome_name}:")

    for var_name, var_col in continuous_vars.items():
        # Filter valid data
        valid_data = df[[outcome_col, var_col]].dropna()

        if len(valid_data) > 2:
            # Pearson correlation
            r, p = pearsonr(valid_data[outcome_col], valid_data[var_col])

            results.append({
                'Asthma Outcome': outcome_name,
                'Variable': var_name,
                'Correlation (r)': r,
                'P-value': p,
                'Significance': 'Yes' if p < 0.05 else 'No',
                'N': len(valid_data),
                'Type': 'Continuous'
            })

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var_name}: r = {r:.3f}, p = {p:.4f} {sig}")

# ============================================================================
# STEP 4: Calculate correlations for categorical variables (tertiles)
# ============================================================================
print("\n[4/5] Calculating correlations (categorical tertiles)...")

# For categorical variables, we'll encode and use Spearman
categorical_vars = {
    'NO2 Tertiles': 'NO2_tertiles',
    'PM2.5 Tertiles': 'PM_tertiles',
    'Traffic Emissions': 'Traffic_tertiles',
    'Industrial Emissions': 'Industrial_tertiles',
    'Building Emissions': 'Building_emissions',
    'Cooking Emissions': 'cook_tertiles'
}

# Encode tertiles: Low=1, Medium=2, High=3
def encode_tertile(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str == 'Low':
        return 1
    elif val_str == 'Medium':
        return 2
    elif val_str == 'High':
        return 3
    else:
        return np.nan

for outcome_name, outcome_col in asthma_outcomes.items():
    print(f"\n  {outcome_name}:")

    for var_name, var_col in categorical_vars.items():
        # Encode categorical variable
        df_temp = df.copy()
        df_temp[f'{var_col}_encoded'] = df_temp[var_col].apply(encode_tertile)

        # Filter valid data
        valid_data = df_temp[[outcome_col, f'{var_col}_encoded']].dropna()

        if len(valid_data) > 2:
            # Spearman correlation (for ordinal data)
            rho, p = spearmanr(valid_data[outcome_col], valid_data[f'{var_col}_encoded'])

            results.append({
                'Asthma Outcome': outcome_name,
                'Variable': var_name,
                'Correlation (r)': rho,
                'P-value': p,
                'Significance': 'Yes' if p < 0.05 else 'No',
                'N': len(valid_data),
                'Type': 'Categorical'
            })

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {var_name}: ρ = {rho:.3f}, p = {p:.4f} {sig}")

# ============================================================================
# STEP 5: Borough-specific mold correlations
# ============================================================================
print("\n[5/5] Calculating borough-specific mold correlations...")

print("\n  Mold Complaints vs Adult Asthma ED Visits by Borough:")
for borough in boroughs.keys():
    borough_data = df[df['borough'] == borough].copy()
    valid_data = borough_data[['age_adjusted_ed_rate_per_10k', 'mold_complaints']].dropna()

    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['age_adjusted_ed_rate_per_10k'],
                       valid_data['mold_complaints'])

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {borough}: r = {r:.3f}, p = {p:.4f}, n = {len(valid_data)} {sig}")

        results.append({
            'Asthma Outcome': f'Adult ED Visits ({borough})',
            'Variable': 'Mold Complaints',
            'Correlation (r)': r,
            'P-value': p,
            'Significance': 'Yes' if p < 0.05 else 'No',
            'N': len(valid_data),
            'Type': 'Borough-specific'
        })

# ============================================================================
# STEP 6: Create summary tables
# ============================================================================
print("\n" + "="*80)
print("CORRELATION SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)

# Sort by absolute correlation
results_df['abs_corr'] = results_df['Correlation (r)'].abs()
results_df_sorted = results_df.sort_values('abs_corr', ascending=False)

print("\n📊 TOP 10 STRONGEST CORRELATIONS (All Outcomes):")
print("-" * 80)
top10 = results_df_sorted.head(10)
for idx, row in top10.iterrows():
    sig = "***" if row['P-value'] < 0.001 else "**" if row['P-value'] < 0.01 else "*" if row['P-value'] < 0.05 else ""
    print(f"{row['Asthma Outcome']:35} ← {row['Variable']:25} r={row['Correlation (r)']:6.3f} {sig}")

# Save full results
results_df_sorted.to_csv('DATA/CLEANED/correlation_results.csv', index=False)
print(f"\n✓ Saved full results to: DATA/CLEANED/correlation_results.csv")

# ============================================================================
# STEP 7: Create visualization
# ============================================================================
print("\n[7/7] Creating correlation heatmap...")

# Filter for Adult ED Visits and continuous + main categorical variables
outcome_col = 'age_adjusted_ed_rate_per_10k'
viz_data = df[df['year'] == 2020].copy()  # Use 2020 data

# Encode categorical variables
for var_col in categorical_vars.values():
    viz_data[f'{var_col}_encoded'] = viz_data[var_col].apply(encode_tertile)

# Select variables for heatmap
heatmap_vars = {
    'Adult ED Visits': outcome_col,
    'NO2': 'NO2_Avg',
    'PM2.5': 'PM_Avg',
    'Mold': 'mold_complaints',
    'Poverty': 'poverty_rate',
    'NO2 Level': 'NO2_tertiles_encoded',
    'PM2.5 Level': 'PM_tertiles_encoded',
    'Traffic': 'Traffic_tertiles_encoded',
    'Industrial': 'Industrial_tertiles_encoded',
    'Building': 'Building_emissions_encoded',
    'Cooking': 'cook_tertiles_encoded'
}

# Create correlation matrix
heatmap_df = viz_data[[col for col in heatmap_vars.values() if col in viz_data.columns]].dropna()
heatmap_df.columns = [k for k, v in heatmap_vars.items() if v in heatmap_df.columns]

corr_matrix = heatmap_df.corr()

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Asthma ED Visits & Environmental Factors (2020)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('DATA/CLEANED/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved heatmap to: DATA/CLEANED/correlation_heatmap.png")

# ============================================================================
# STEP 8: Interpretation guide
# ============================================================================
print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)

print("\n📖 Correlation Strength:")
print("  |r| < 0.3  : Weak correlation")
print("  0.3 ≤ |r| < 0.5 : Moderate correlation")
print("  |r| ≥ 0.5  : Strong correlation")

print("\n📖 Significance Levels:")
print("  ***  p < 0.001 (highly significant)")
print("  **   p < 0.01  (very significant)")
print("  *    p < 0.05  (significant)")
print("  (no star) = not significant")

print("\n📖 Correlation Direction:")
print("  Positive (+): As X increases, asthma increases")
print("  Negative (-): As X increases, asthma decreases")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Identify significant correlations
sig_results = results_df[results_df['Significance'] == 'Yes'].copy()
sig_results_sorted = sig_results.sort_values('abs_corr', ascending=False)

if len(sig_results_sorted) > 0:
    print(f"\n✅ Found {len(sig_results_sorted)} significant correlations")
    print("\nStrongest significant relationships:")
    for idx, row in sig_results_sorted.head(5).iterrows():
        direction = "↑" if row['Correlation (r)'] > 0 else "↓"
        print(f"  {direction} {row['Asthma Outcome']} ← {row['Variable']}: r={row['Correlation (r)']:.3f}")
else:
    print("\n⚠️  No significant correlations found")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)