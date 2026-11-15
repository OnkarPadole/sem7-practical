#python3 -m pip install pandas scikit-learn

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Store input and output files beside this script.
FOLDER = Path(__file__).resolve().parent

INPUT_FILE = FOLDER / "Telecom_Customer_Churn.csv"
CLEANED_FILE = FOLDER / "cleaned_telecom_customer_churn.csv"
TRAIN_FILE = FOLDER / "churn_train.csv"
TEST_FILE = FOLDER / "churn_test.csv"
SCALER_FILE = FOLDER / "churn_scaler.csv"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

# Download the dataset if it is not already in this folder.
if not INPUT_FILE.exists():
    print("Downloading the IBM Telco Customer Churn dataset...")
    try:
        urlretrieve(DATASET_URL, INPUT_FILE)
    except Exception as error:
        raise SystemExit(f"Could not download the dataset: {error}")

# Load dataset and normalize column names/spacing.
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

required_columns = {
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"
}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(
        "The CSV is not the expected Telco churn dataset. "
        f"Missing columns: {sorted(missing_columns)}"
    )

# Explore the original data.
print("Original dataset shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nFirst five rows:\n", df.head())
print("\nMissing values before cleaning:\n", df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Convert numeric columns. Blank TotalCharges values become NaN.
for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Drop exact duplicates.
df = df.drop_duplicates().copy()

# Standardize churn labels.
df["Churn"] = df["Churn"].str.lower().map(
    {"yes": "Yes", "no": "No"}
).fillna(df["Churn"])

# Fill missing numeric values with medians and categorical values with modes.
for column in df.columns:
    if pd.api.types.is_numeric_dtype(df[column]):
        median = df[column].median()
        df[column] = df[column].fillna(median)
    else:
        mode = df[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)

# Find potential outliers with the IQR rule and report counts.
# Keep these rows because high charges/tenure can be valid customers.
outlier_counts = {}
for column in df.select_dtypes(include="number").columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[column] = int(
            ((df[column] < lower) | (df[column] > upper)).sum()
        )

print("\nPotential outlier counts (IQR rule):")
print(outlier_counts)

# Feature engineering.
df["estimated_customer_value"] = df["tenure"] * df["MonthlyCharges"]
df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, float("inf")],
    labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
).astype(str)

# Export the cleaned dataset.
df.to_csv(CLEANED_FILE, index=False)
print(f"\nCleaned dataset saved to: {CLEANED_FILE}")

# Separate target and predictors. Customer ID is an identifier, not a predictor.
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
y = df["Churn"]

# Split first so preprocessing learns only from training data.
stratify_target = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_target,
)

# One-hot encode categories; match test columns to training columns.
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Scale numeric features. Convert to float before assigning decimal results.
numeric_columns = X_train.select_dtypes(include="number").columns.tolist()

if numeric_columns:
    scaler = StandardScaler()
    X_train.loc[:, numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns].astype(float)
    )
    X_test.loc[:, numeric_columns] = scaler.transform(
        X_test[numeric_columns].astype(float)
    )

    pd.DataFrame({
        "feature": numeric_columns,
        "mean": scaler.mean_,
        "scale": scaler.scale_,
    }).to_csv(SCALER_FILE, index=False)

# Save train/test datasets with the target column included.
train_output = X_train.copy()
train_output["Churn"] = y_train

test_output = X_test.copy()
test_output["Churn"] = y_test

train_output.to_csv(TRAIN_FILE, index=False)
test_output.to_csv(TEST_FILE, index=False)

print(f"Training data saved: {TRAIN_FILE} ({len(train_output)} rows)")
print(f"Testing data saved: {TEST_FILE} ({len(test_output)} rows)")
if numeric_columns:
    print(f"Scaler parameters saved: {SCALER_FILE}")