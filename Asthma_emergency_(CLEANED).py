import pandas as pd
import re

# Load the data
df = pd.read_csv('DATA/NYC EH Data Portal - Asthma emergency department visits (adults) (full table).csv')

print("Original shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())
print("\nYears in data:", sorted(df['TimePeriod'].unique()))
print("\nGeo types:", df['GeoType'].unique())

# Clean the data
def extract_numeric(value):
    """
    Extract numeric value from strings like '11.6' or '8,000'
    Returns None if value cannot be parsed
    """
    if pd.isna(value):
        return None

    value_str = str(value)

    # Handle special characters
    if value_str.strip() == '†':
        return None

    # Remove commas and convert to float
    value_str = value_str.replace(',', '').strip()

    try:
        return float(value_str)
    except:
        return None

# Create cleaned dataframe
df_clean = df.copy()

# Filter to only UHF42 level (neighborhood-level data) - this is the most granular useful level
df_clean = df_clean[df_clean['GeoType'] == 'UHF42'].copy()

print(f"\n\nFiltered to UHF42 neighborhoods: {len(df_clean)} rows")

# Extract clean numeric values
df_clean['age_adjusted_rate_clean'] = df_clean['Age-adjusted rate per 10,000'].apply(extract_numeric)
df_clean['estimated_annual_rate_clean'] = df_clean['Estimated annual rate per 10,000'].apply(extract_numeric)
df_clean['number_clean'] = df_clean['Number'].apply(extract_numeric)

# Select and rename columns
df_clean = df_clean[[
    'TimePeriod',
    'GeoID',
    'Geography',
    'age_adjusted_rate_clean',
    'estimated_annual_rate_clean',
    'number_clean'
]].copy()

# Rename columns for clarity
df_clean.columns = [
    'year',
    'uhf_code',
    'neighborhood',
    'age_adjusted_ed_rate_per_10k',
    'estimated_annual_ed_rate_per_10k',
    'estimated_annual_ed_visits'
]

# Sort by year and neighborhood
df_clean = df_clean.sort_values(['year', 'neighborhood']).reset_index(drop=True)

print("\n\nCleaned data shape:", df_clean.shape)
print("\nCleaned columns:", df_clean.columns.tolist())
print("\nSample of cleaned data (most recent year):")
print(df_clean[df_clean['year'] == 2023].head(10))

print("\n\nData Summary:")
print(df_clean[df_clean['year'] == 2023].describe())

print("\n\nMissing values:")
print(df_clean.isnull().sum())

# Save cleaned data
output_file = 'asthma_ed_visits_adults_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n\nCleaned data saved to: {output_file}")

# Display key statistics for 2023
df_2023 = df_clean[df_clean['year'] == 2023]
print("\n\n2023 Key Statistics:")
print(f"Neighborhoods analyzed: {df_2023['neighborhood'].nunique()}")
print(f"Average ED visit rate: {df_2023['age_adjusted_ed_rate_per_10k'].mean():.1f} per 10,000")
print(f"Highest ED rate: {df_2023['age_adjusted_ed_rate_per_10k'].max():.1f} per 10,000 in {df_2023.loc[df_2023['age_adjusted_ed_rate_per_10k'].idxmax(), 'neighborhood']}")
print(f"Lowest ED rate: {df_2023['age_adjusted_ed_rate_per_10k'].min():.1f} per 10,000 in {df_2023.loc[df_2023['age_adjusted_ed_rate_per_10k'].idxmin(), 'neighborhood']}")

# Show trends over time for a few neighborhoods
print("\n\nTrends for selected neighborhoods (2019-2023):")
selected_neighborhoods = ['Hunts Point - Mott Haven', 'Upper East Side', 'Rockaways']
for neigh in selected_neighborhoods:
    neigh_data = df_clean[(df_clean['neighborhood'] == neigh) & (df_clean['year'] >= 2019)]
    if not neigh_data.empty:
        print(f"\n{neigh}:")
        print(neigh_data[['year', 'age_adjusted_ed_rate_per_10k']].to_string(index=False))