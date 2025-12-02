import pandas as pd
import numpy as np

print("="*80)
print("MERGING ALL DATASETS AT UHF42 NEIGHBORHOOD LEVEL")
print("="*80)

# ============================================================================
# STEP 1: Load all cleaned asthma datasets
# ============================================================================
print("\n[1/6] Loading asthma datasets...")

# Load asthma prevalence (adults)
asthma_adults = pd.read_csv('DATA/CLEANED/adults_with_asthma_cleaned.csv')
print(f"  ✓ Adults with asthma: {asthma_adults.shape}")

# Load asthma ED visits (all age groups)
asthma_ed_adults = pd.read_csv('DATA/CLEANED/asthma_ed_visits_adults_cleaned.csv')
asthma_ed_0_4 = pd.read_csv('DATA/CLEANED/asthma_ed_visits_age_0_4_cleaned.csv')
asthma_ed_5_17 = pd.read_csv('DATA/CLEANED/asthma_ed_visits_age_5_17_cleaned.csv')
print(f"  ✓ Asthma ED visits (adults): {asthma_ed_adults.shape}")
print(f"  ✓ Asthma ED visits (0-4): {asthma_ed_0_4.shape}")
print(f"  ✓ Asthma ED visits (5-17): {asthma_ed_5_17.shape}")

# ============================================================================
# STEP 2: Load poverty data (already at UHF level)
# ============================================================================
print("\n[2/6] Loading poverty data...")
poverty_data = pd.read_csv('DATA/CLEANED/pov_data[cleaned].csv')

# Filter to UHF level only (3-digit codes like 101, 201, etc.)
uhf_poverty = poverty_data[poverty_data['NTA_CODE'].astype(str).str.len() == 3].copy()
uhf_poverty = uhf_poverty.rename(columns={
    'NTA_CODE': 'uhf_code',
    'NTA_NAME': 'neighborhood',
    'Households_Below_Poverty': 'households_below_poverty',
    'Poverty_percent': 'poverty_rate'
})
uhf_poverty['uhf_code'] = uhf_poverty['uhf_code'].astype(int)
print(f"  ✓ Poverty data (UHF level): {uhf_poverty.shape}")

# ============================================================================
# STEP 3: Create master neighborhood reference
# ============================================================================
print("\n[3/6] Creating master neighborhood reference...")

# Use the most recent year (2023) from asthma data as reference
master_ref = asthma_adults[asthma_adults['year'] == 2020][['uhf_code', 'neighborhood']].copy()
master_ref = master_ref.sort_values('uhf_code').reset_index(drop=True)
print(f"  ✓ Master reference: {len(master_ref)} UHF neighborhoods")
print(f"  ✓ UHF codes: {master_ref['uhf_code'].min()} to {master_ref['uhf_code'].max()}")

# ============================================================================
# STEP 4: Merge asthma datasets by year and UHF code
# ============================================================================
print("\n[4/6] Merging asthma datasets...")

# Start with asthma prevalence
merged = asthma_adults.copy()

# Merge ED visits (adults)
merged = merged.merge(
    asthma_ed_adults[['year', 'uhf_code', 'age_adjusted_ed_rate_per_10k', 'estimated_annual_ed_visits']],
    on=['year', 'uhf_code'],
    how='left',
    suffixes=('', '_ed_adults')
)

# Merge ED visits (age 0-4)
merged = merged.merge(
    asthma_ed_0_4[['year', 'uhf_code', 'ed_rate_per_10k_age_0_4', 'estimated_annual_ed_visits_age_0_4']],
    on=['year', 'uhf_code'],
    how='left'
)

# Merge ED visits (age 5-17)
merged = merged.merge(
    asthma_ed_5_17[['year', 'uhf_code', 'ed_rate_per_10k_age_5_17', 'estimated_annual_ed_visits_age_5_17']],
    on=['year', 'uhf_code'],
    how='left'
)

print(f"  ✓ Merged asthma data: {merged.shape}")
print(f"  ✓ Years: {merged['year'].min()} to {merged['year'].max()}")

# ============================================================================
# STEP 5: Add poverty data (static - no year dimension)
# ============================================================================
print("\n[5/6] Adding poverty data...")

# Poverty data is static (no year), so merge once
merged = merged.merge(
    uhf_poverty[['uhf_code', 'households_below_poverty', 'poverty_rate']],
    on='uhf_code',
    how='left'
)

print(f"  ✓ Merged with poverty: {merged.shape}")

# ============================================================================
# STEP 6: Clean and organize final dataset
# ============================================================================
print("\n[6/6] Organizing final dataset...")

# Reorder columns for clarity
column_order = [
    'year',
    'uhf_code',
    'neighborhood',
    # Asthma prevalence
    'age_adjusted_asthma_percent',
    'estimated_adults_with_asthma',
    # Asthma ED visits (adults)
    'age_adjusted_ed_rate_per_10k',
    'estimated_annual_ed_visits',
    # Asthma ED visits (children 0-4)
    'ed_rate_per_10k_age_0_4',
    'estimated_annual_ed_visits_age_0_4',
    # Asthma ED visits (children 5-17)
    'ed_rate_per_10k_age_5_17',
    'estimated_annual_ed_visits_age_5_17',
    # Socioeconomic
    'poverty_rate',
    'households_below_poverty',
    # Flags
    'statistically_significant'
]

merged = merged[column_order]

# Sort by year and neighborhood
merged = merged.sort_values(['year', 'neighborhood']).reset_index(drop=True)

print(f"\n✓ Final merged dataset: {merged.shape}")
print(f"  - {merged['year'].nunique()} years ({merged['year'].min()}-{merged['year'].max()})")
print(f"  - {merged['uhf_code'].nunique()} neighborhoods")
print(f"  - {len(column_order)} variables")

# ============================================================================
# STEP 7: Save merged dataset
# ============================================================================
output_file = 'DATA/CLEANED/merged_asthma_poverty_data.csv'
merged.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

# ============================================================================
# STEP 8: Generate summary statistics
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS (2023)")
print("="*80)

df_2023 = merged[merged['year'] == 2023].copy()

print("\nAsthma Prevalence (Adults):")
print(f"  Mean: {df_2023['age_adjusted_asthma_percent'].mean():.1f}%")
print(f"  Range: {df_2023['age_adjusted_asthma_percent'].min():.1f}% - {df_2023['age_adjusted_asthma_percent'].max():.1f}%")

print("\nAsthma ED Visits (Adults) per 10k:")
print(f"  Mean: {df_2023['age_adjusted_ed_rate_per_10k'].mean():.1f}")
print(f"  Range: {df_2023['age_adjusted_ed_rate_per_10k'].min():.1f} - {df_2023['age_adjusted_ed_rate_per_10k'].max():.1f}")

print("\nAsthma ED Visits (Children 0-4) per 10k:")
print(f"  Mean: {df_2023['ed_rate_per_10k_age_0_4'].mean():.1f}")
print(f"  Range: {df_2023['ed_rate_per_10k_age_0_4'].min():.1f} - {df_2023['ed_rate_per_10k_age_0_4'].max():.1f}")

print("\nAsthma ED Visits (Children 5-17) per 10k:")
print(f"  Mean: {df_2023['ed_rate_per_10k_age_5_17'].mean():.1f}")
print(f"  Range: {df_2023['ed_rate_per_10k_age_5_17'].min():.1f} - {df_2023['ed_rate_per_10k_age_5_17'].max():.1f}")

print("\nPoverty Rate:")
print(f"  Mean: {df_2023['poverty_rate'].mean():.1f}%")
print(f"  Range: {df_2023['poverty_rate'].min():.1f}% - {df_2023['poverty_rate'].max():.1f}%")

# ============================================================================
# STEP 9: Check for missing data
# ============================================================================
print("\n" + "="*80)
print("DATA QUALITY CHECK")
print("="*80)

print("\nMissing values by column (2023):")
missing = df_2023.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, count in missing.items():
        pct = (count / len(df_2023)) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")
else:
    print("  ✓ No missing values!")

# ============================================================================
# STEP 10: Top/Bottom neighborhoods
# ============================================================================
print("\n" + "="*80)
print("HIGHEST & LOWEST ASTHMA RATES (2023)")
print("="*80)

print("\nTop 5 neighborhoods by adult asthma prevalence:")
top_asthma = df_2023.nlargest(5, 'age_adjusted_asthma_percent')[['neighborhood', 'age_adjusted_asthma_percent', 'poverty_rate']]
for idx, row in top_asthma.iterrows():
    print(f"  {row['neighborhood']}: {row['age_adjusted_asthma_percent']:.1f}% asthma, {row['poverty_rate']:.1f}% poverty")

print("\nTop 5 neighborhoods by adult ED visits per 10k:")
top_ed = df_2023.nlargest(5, 'age_adjusted_ed_rate_per_10k')[['neighborhood', 'age_adjusted_ed_rate_per_10k', 'poverty_rate']]
for idx, row in top_ed.iterrows():
    print(f"  {row['neighborhood']}: {row['age_adjusted_ed_rate_per_10k']:.1f} ED visits/10k, {row['poverty_rate']:.1f}% poverty")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("1. Add 311 mold complaint data (needs geocoding to UHF)")
print("2. Add air quality data (needs aggregation from NTA to UHF)")
print("3. Add spatial data (UHF boundaries for mapping)")
print("="*80)