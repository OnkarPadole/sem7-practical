from pathlib import Path

import pandas as pd

FOLDER = Path(__file__).resolve().parent
INPUT_FILE = FOLDER / "RealEstate_Prices.csv"
CLEANED_FILE = FOLDER / "cleaned_real_estate_prices.csv"
ENCODED_FILE = FOLDER / "real_estate_encoded.csv"
SUMMARY_FILE = FOLDER / "real_estate_summary.csv"

# Create a small sample dataset so the program runs when no CSV is present.
# Replace this sample with your real dataset for meaningful market results.
if not INPUT_FILE.exists():
    sample = pd.DataFrame(
        {
            "Sale Price": [250000, 315000, 190000, 420000, 275000, 360000],
            "Bedrooms": [3, 4, 2, 5, 3, 4],
            "Bathrooms": [2, 2, 1, 3, 2, 2],
            "Area Sq Ft": [1500, 2100, 1100, 2800, 1600, 2300],
            "Property Type": [
                "House", "House", "Apartment",
                "House", "Apartment", "Condo",
            ],
            "Neighborhood": [
                "North", "West", "South", "West", "South", "North",
            ],
            "Sale Date": [
                "2024-01-15", "2024-03-10", "2024-05-21",
                "2024-07-02", "2024-09-18", "2024-11-05",
            ],
        }
    )
    sample.to_csv(INPUT_FILE, index=False)
    print(f"No input CSV found. Created sample dataset: {INPUT_FILE}")

# 1. Import data and clean column names.
df = pd.read_csv(INPUT_FILE)

def clean_column_name(name):
    """Convert column names to lowercase snake_case."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("$", "price_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )

df.columns = [clean_column_name(column) for column in df.columns]
print("\nColumns after cleaning:")
print(df.columns.tolist())

# Trim extra spaces from text values.
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Convert likely numeric fields, including prices and property measurements.
for column in df.columns:
    if any(word in column for word in ["price", "bed", "bath", "area", "sqft", "year"]):
        df[column] = pd.to_numeric(
            df[column].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce",
        )

# Parse date columns if present.
date_columns = [column for column in df.columns if "date" in column]
for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")

# Remove duplicate rows.
before = len(df)
df = df.drop_duplicates().copy()
print(f"\nDuplicate rows removed: {before - len(df)}")

# 2. Handle missing values: numeric columns use median; text uses mode.
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        median = df[column].median()
        if pd.notna(median):
            df[column] = df[column].fillna(median)
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        # Preserve missing dates; they cannot be safely inferred.
        pass
    else:
        mode = df[column].mode(dropna=True)
        df[column] = df[column].fillna(
            mode.iloc[0] if not mode.empty else "Unknown"
        )

# 3. Optional merge point:
# If you have a separate neighborhood file with a matching "neighborhood"
# column, uncomment and edit these lines:
#
# neighborhood_info = pd.read_csv(FOLDER / "Neighborhood_Info.csv")
# neighborhood_info.columns = [clean_column_name(c) for c in neighborhood_info.columns]
# df = df.merge(neighborhood_info, on="neighborhood", how="left")

# 4. Optional filtering: set values to filter the dataset, or leave as None.
# Example: LOCATION_FILTER = "North"
LOCATION_FILTER = None
PROPERTY_TYPE_FILTER = None
START_DATE = None  # Example: "2024-01-01"
END_DATE = None    # Example: "2024-12-31"

filtered = df.copy()

if LOCATION_FILTER and "neighborhood" in filtered.columns:
    filtered = filtered[
        filtered["neighborhood"].str.casefold() == LOCATION_FILTER.casefold()
    ]

if PROPERTY_TYPE_FILTER and "property_type" in filtered.columns:
    filtered = filtered[
        filtered["property_type"].str.casefold() == PROPERTY_TYPE_FILTER.casefold()
    ]

if date_columns and (START_DATE or END_DATE):
    date_column = date_columns[0]
    if START_DATE:
        filtered = filtered[filtered[date_column] >= pd.Timestamp(START_DATE)]
    if END_DATE:
        filtered = filtered[filtered[date_column] <= pd.Timestamp(END_DATE)]

print("\nRows after optional filtering:", len(filtered))

# 6. Create summary statistics grouped by neighborhood and/or property type.
group_columns = [
    column for column in ["neighborhood", "property_type"]
    if column in filtered.columns
]

price_column = next(
    (
        column for column in filtered.columns
        if "price" in column and pd.api.types.is_numeric_dtype(filtered[column])
    ),
    None,
)

if price_column and group_columns:
    summary = (
        filtered.groupby(group_columns, dropna=False)[price_column]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    summary.to_csv(SUMMARY_FILE, index=False)
    print("\nPrice summary:")
    print(summary.to_string(index=False))
    print(f"Summary saved to: {SUMMARY_FILE}")
else:
    print(
        "\nCould not create grouped price summary. "
        "Check that the CSV has a numeric price column and neighborhood/property_type."
    )

# Identify possible price outliers by IQR, report them, and keep them in the
# cleaned data because unusually expensive homes may be valid.
if price_column:
    q1 = filtered[price_column].quantile(0.25)
    q3 = filtered[price_column].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = filtered[
            (filtered[price_column] < lower) | (filtered[price_column] > upper)
        ]
        print(f"\nPotential price outliers detected: {len(outliers)}")
        print(f"Outlier bounds: {lower:.2f} to {upper:.2f}")
    else:
        print("\nPrice outlier check skipped because the price IQR is zero.")

# 5. One-hot encode categorical columns for analysis/modeling.
categorical_columns = filtered.select_dtypes(include=["object", "category"]).columns
encoded = pd.get_dummies(filtered, columns=categorical_columns, dtype=int)

# Save outputs.
filtered.to_csv(CLEANED_FILE, index=False)
encoded.to_csv(ENCODED_FILE, index=False)

print(f"\nCleaned data saved to: {CLEANED_FILE}")
print(f"Encoded data saved to: {ENCODED_FILE}")