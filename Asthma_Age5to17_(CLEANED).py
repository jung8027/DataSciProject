import pandas as pd
import re

# Load the data
df = pd.read_csv('DATA/NYC EH Data Portal - Asthma emergency department visits (age 5 to 17) (full table).csv')

print("Original shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())
print("\nYears in data:", sorted(df['TimePeriod'].unique()))
print("\nGeo types:", df['GeoType'].unique())

# Clean the data
def extract_numeric(value):
    """
    Extract numeric value from strings like '11.6' or '8,000'
    Handles asterisks (statistical significance markers) and special characters
    Returns None if value cannot be parsed
    """
    if pd.isna(value):
        return None

    value_str = str(value)

    # Handle special characters
    if value_str.strip() == '†':
        return None

    # Remove asterisks (they indicate unstable estimates/small numbers)
    value_str = value_str.replace('*', '').strip()

    # Remove commas and convert to float
    value_str = value_str.replace(',', '').strip()

    try:
        return float(value_str)
    except:
        return None

# Create cleaned dataframe
df_clean = df.copy()

# Filter to only UHF42 level (neighborhood-level data)
df_clean = df_clean[df_clean['GeoType'] == 'UHF42'].copy()

print(f"\n\nFiltered to UHF42 neighborhoods: {len(df_clean)} rows")

# Extract clean numeric values
df_clean['estimated_annual_rate_clean'] = df_clean['Estimated annual rate per 10,000'].apply(extract_numeric)
df_clean['number_clean'] = df_clean['Number'].apply(extract_numeric)

# Create a flag for statistically unstable values (marked with *)
df_clean['unstable_estimate'] = df_clean['Estimated annual rate per 10,000'].astype(str).str.contains('\*', na=False)

# Select and rename columns
df_clean = df_clean[[
    'TimePeriod',
    'GeoID',
    'Geography',
    'estimated_annual_rate_clean',
    'number_clean',
    'unstable_estimate'
]].copy()

# Rename columns for clarity
df_clean.columns = [
    'year',
    'uhf_code',
    'neighborhood',
    'ed_rate_per_10k_age_5_17',
    'estimated_annual_ed_visits_age_5_17',
    'unstable_estimate'
]

# Sort by year and neighborhood
df_clean = df_clean.sort_values(['year', 'neighborhood']).reset_index(drop=True)

print("\n\nCleaned data shape:", df_clean.shape)
print("\nCleaned columns:", df_clean.columns.tolist())
print("\nSample of cleaned data (most recent year):")
print(df_clean[df_clean['year'] == 2023].head(10))

print("\n\nData Summary for 2023:")
print(df_clean[df_clean['year'] == 2023].describe())

print("\n\nMissing values:")
print(df_clean.isnull().sum())

# Count unstable estimates
print(f"\n\nUnstable estimates (marked with *): {df_clean['unstable_estimate'].sum()} out of {len(df_clean)} records")

# Save cleaned data
output_file = 'asthma_ed_visits_age_5_17_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n\nCleaned data saved to: {output_file}")

# Display key statistics for 2023
df_2023 = df_clean[df_clean['year'] == 2023]
print("\n\n2023 Key Statistics (Age 5-17):")
print(f"Neighborhoods analyzed: {df_2023['neighborhood'].nunique()}")
print(f"Average ED visit rate: {df_2023['ed_rate_per_10k_age_5_17'].mean():.1f} per 10,000 children")
print(f"Highest ED rate: {df_2023['ed_rate_per_10k_age_5_17'].max():.1f} per 10,000 in {df_2023.loc[df_2023['ed_rate_per_10k_age_5_17'].idxmax(), 'neighborhood']}")
print(f"Lowest ED rate: {df_2023['ed_rate_per_10k_age_5_17'].min():.1f} per 10,000 in {df_2023.loc[df_2023['ed_rate_per_10k_age_5_17'].idxmin(), 'neighborhood']}")

# Show trends over time for a few neighborhoods
print("\n\nTrends for selected neighborhoods (2019-2023):")
selected_neighborhoods = ['High Bridge - Morrisania', 'Upper East Side', 'East Harlem']
for neigh in selected_neighborhoods:
    neigh_data = df_clean[(df_clean['neighborhood'] == neigh) & (df_clean['year'] >= 2019)]
    if not neigh_data.empty:
        print(f"\n{neigh}:")
        print(neigh_data[['year', 'ed_rate_per_10k_age_5_17', 'unstable_estimate']].to_string(index=False))

# Compare 2019 (pre-pandemic) vs 2023 rates
df_2019 = df_clean[df_clean['year'] == 2019]
if not df_2019.empty and not df_2023.empty:
    print(f"\n\nPandemic Impact Comparison (Age 5-17):")
    print(f"2019 Average ED rate: {df_2019['ed_rate_per_10k_age_5_17'].mean():.1f} per 10,000")
    print(f"2023 Average ED rate: {df_2023['ed_rate_per_10k_age_5_17'].mean():.1f} per 10,000")
    pct_change = ((df_2023['ed_rate_per_10k_age_5_17'].mean() - df_2019['ed_rate_per_10k_age_5_17'].mean()) / df_2019['ed_rate_per_10k_age_5_17'].mean()) * 100
    print(f"Change: {pct_change:.1f}%")

# Compare age groups (if other cleaned files exist)
print("\n\n" + "="*60)
print("AGE GROUP COMPARISON NOTES:")
print("="*60)
print("This is school-age children (5-17) asthma ED visits.")
print("Compare with:")
print("  - Age 0-4: Young children, often higher rates")
print("  - Adults: Generally lower rates than children")
print("\nSchool-age children typically show:")
print("  - Moderate rates (between young children and adults)")
print("  - Seasonal patterns (worse during school year)")
print("  - Exercise-induced asthma effects")