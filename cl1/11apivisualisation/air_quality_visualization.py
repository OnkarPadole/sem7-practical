from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "City_Air_Quality.csv"

# Create a small demonstration dataset if the CSV is missing.
# Replace this with actual city measurements for meaningful analysis.
if not DATA_FILE.exists():
    sample = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=14, freq="D"),
        "PM2.5": [42, 55, 38, 61, 74, 68, 50, 46, 59, 81, 70, 63, 48, 44],
        "PM10":  [75, 88, 67, 94, 110, 105, 82, 79, 91, 125, 112, 98, 80, 72],
        "CO":    [0.7, 0.8, 0.6, 0.9, 1.1, 1.0, 0.8, 0.7, 0.9, 1.2, 1.1, 0.9, 0.8, 0.7],
        "AQI":   [82, 95, 76, 108, 128, 120, 93, 87, 103, 142, 130, 115, 91, 84],
    })
    sample.to_csv(DATA_FILE, index=False)
    print(f"Created sample dataset: {DATA_FILE}")

# 1. Load and inspect the dataset
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())

# Find expected columns without depending on capitalization or spaces.
column_lookup = {column.lower().replace(" ", ""): column for column in df.columns}

def find_column(*names):
    for name in names:
        key = name.lower().replace(" ", "")
        if key in column_lookup:
            return column_lookup[key]
    return None

date_col = find_column("Date", "Datetime", "Timestamp")
aqi_col = find_column("AQI", "Air Quality Index")
pm25_col = find_column("PM2.5", "PM25")
pm10_col = find_column("PM10")
co_col = find_column("CO")

if date_col is None or aqi_col is None:
    raise ValueError("CSV must contain a date/time column and an AQI column.")

# Convert date and measurement columns to usable types.
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
measurement_cols = [c for c in [aqi_col, pm25_col, pm10_col, co_col] if c]
for column in measurement_cols:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna(subset=[date_col, aqi_col]).sort_values(date_col).copy()

if df.empty:
    raise ValueError("No valid rows remain after parsing the date and AQI columns.")

# Derive AQI categories for the distribution plot.
def aqi_category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Moderate"
    if value <= 150:
        return "Unhealthy for sensitive groups"
    if value <= 200:
        return "Unhealthy"
    if value <= 300:
        return "Very unhealthy"
    return "Hazardous"

df["AQI_Category"] = df[aqi_col].apply(aqi_category)

# 4. AQI trend over time
plt.figure(figsize=(10, 5))
plt.plot(df[date_col], df[aqi_col], marker="o", color="black", label="AQI")
plt.title("Air Quality Index Trend")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FOLDER / "aqi_trend.png", dpi=150)

# 5. Pollutant trends, each on its own subplot
pollutants = [c for c in [pm25_col, pm10_col, co_col] if c]
if pollutants:
    fig, axes = plt.subplots(
        len(pollutants), 1, figsize=(10, 4 * len(pollutants)), sharex=True
    )
    if len(pollutants) == 1:
        axes = [axes]

    for ax, pollutant in zip(axes, pollutants):
        ax.plot(df[date_col], df[pollutant], marker="o", label=pollutant)
        ax.set_title(f"{pollutant} Levels Over Time")
        ax.set_ylabel(pollutant)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(FOLDER / "pollutant_trends.png", dpi=150)

# 6. Compare AQI by date (daily mean if there are multiple readings per day)
df["Day"] = df[date_col].dt.date
daily_aqi = df.groupby("Day")[aqi_col].mean()

plt.figure(figsize=(11, 5))
daily_aqi.plot(kind="bar", color="steelblue")
plt.title("Average AQI by Date")
plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FOLDER / "aqi_by_date.png", dpi=150)

# 7. AQI distribution by derived AQI category
category_order = [
    "Good", "Moderate", "Unhealthy for sensitive groups",
    "Unhealthy", "Very unhealthy", "Hazardous",
]
box_data = [
    df.loc[df["AQI_Category"] == category, aqi_col].dropna()
    for category in category_order
]
box_data = [values for values in box_data if not values.empty]
box_labels = [
    category for category in category_order
    if not df.loc[df["AQI_Category"] == category].empty
]

plt.figure(figsize=(10, 5))
plt.boxplot(box_data, tick_labels=box_labels)
plt.title("AQI Distribution by AQI Category")
plt.xlabel("AQI Category")
plt.ylabel("AQI")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(FOLDER / "aqi_category_boxplot.png", dpi=150)

# 8. AQI relationship with pollutant measurements
if pollutants:
    fig, axes = plt.subplots(
        1, len(pollutants), figsize=(5 * len(pollutants), 4)
    )
    if len(pollutants) == 1:
        axes = [axes]

    for ax, pollutant in zip(axes, pollutants):
        ax.scatter(df[pollutant], df[aqi_col], alpha=0.7)
        ax.set_title(f"AQI vs {pollutant}")
        ax.set_xlabel(pollutant)
        ax.set_ylabel("AQI")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FOLDER / "aqi_pollutant_scatter.png", dpi=150)

plt.show()

print("\nCharts saved in:", FOLDER)