import pandas as pd
import re

# Load the data
df = pd.read_csv('DATA/NYC EH Data Portal - Adults with asthma (full table).csv')

print("Original shape:", df.shape)
print("\nOriginal columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Clean the data
def extract_numeric_from_confidence_interval(value):
    """
    Extract the main numeric value from strings like '11.6 (6.5, 19.8)' or '8,000'
    Returns None if value cannot be parsed
    """
    if pd.isna(value):
        return None

    # Convert to string
    value_str = str(value)

    # Remove asterisks (they indicate statistical significance)
    value_str = value_str.replace('*', '').strip()

    # Pattern to extract first number before parentheses
    match = re.match(r'([\d,\.]+)', value_str)
    if match:
        # Remove commas and convert to float
        return float(match.group(1).replace(',', ''))

    return None

# Create cleaned dataframe
df_clean = df.copy()

# Filter to only UHF34 level (neighborhood-level data)
df_clean = df_clean[df_clean['GeoType'] == 'UHF34'].copy()

print(f"\n\nFiltered to UHF34 neighborhoods: {len(df_clean)} rows")

# Extract clean numeric values
df_clean['age_adjusted_percent_clean'] = df_clean['Age-adjusted percent'].apply(extract_numeric_from_confidence_interval)
df_clean['number_clean'] = df_clean['Number'].apply(extract_numeric_from_confidence_interval)
df_clean['percent_clean'] = df_clean['Percent'].apply(extract_numeric_from_confidence_interval)

# Create a flag for statistically significant values (marked with *)
df_clean['is_significant'] = df_clean['Age-adjusted percent'].astype(str).str.contains('\*', na=False)

# Select and rename columns for clarity
df_clean = df_clean[[
    'TimePeriod',
    'GeoID',
    'Geography',
    'age_adjusted_percent_clean',
    'number_clean',
    'percent_clean',
    'is_significant'
]].copy()

# Rename columns to be more descriptive
df_clean.columns = [
    'year',
    'uhf_code',
    'neighborhood',
    'age_adjusted_asthma_percent',
    'estimated_adults_with_asthma',
    'asthma_percent',
    'statistically_significant'
]

# Sort by neighborhood
df_clean = df_clean.sort_values('neighborhood').reset_index(drop=True)

print("\n\nCleaned data shape:", df_clean.shape)
print("\nCleaned columns:", df_clean.columns.tolist())
print("\nSample of cleaned data:")
print(df_clean.head(10))

print("\n\nData Summary:")
print(df_clean.describe())

print("\n\nMissing values:")
print(df_clean.isnull().sum())

# Save cleaned data
output_file = 'adults_with_asthma_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n\nCleaned data saved to: {output_file}")

# Display key statistics
print("\n\nKey Statistics:")
print(f"Neighborhoods analyzed: {df_clean['neighborhood'].nunique()}")
print(f"Average asthma prevalence: {df_clean['age_adjusted_asthma_percent'].mean():.1f}%")
print(f"Highest asthma rate: {df_clean['age_adjusted_asthma_percent'].max():.1f}% in {df_clean.loc[df_clean['age_adjusted_asthma_percent'].idxmax(), 'neighborhood']}")
print(f"Lowest asthma rate: {df_clean['age_adjusted_asthma_percent'].min():.1f}% in {df_clean.loc[df_clean['age_adjusted_asthma_percent'].idxmin(), 'neighborhood']}")