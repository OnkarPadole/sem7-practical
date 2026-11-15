from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

FOLDER = Path(__file__).resolve().parent
DATA_FILE = FOLDER / "Retail_Sales_Data.csv"

# Create demonstration data if the dataset is not present.
if not DATA_FILE.exists():
    sample = pd.DataFrame({
        "Transaction_Date": [
            "2025-01-05", "2025-01-08", "2025-01-12", "2025-01-15",
            "2025-02-02", "2025-02-06", "2025-02-11", "2025-02-18",
            "2025-03-03", "2025-03-09", "2025-03-14", "2025-03-20",
        ],
        "Region": [
            "North", "South", "East", "West",
            "North", "South", "East", "West",
            "North", "South", "East", "West",
        ],
        "Product_Category": [
            "Electronics", "Clothing", "Groceries", "Electronics",
            "Clothing", "Groceries", "Electronics", "Clothing",
            "Groceries", "Electronics", "Clothing", "Groceries",
        ],
        "Quantity": [2, 5, 8, 1, 4, 10, 3, 6, 12, 2, 5, 9],
        "Sales_Amount": [
            1200, 250, 160, 900, 320, 200, 1800, 480, 240, 1400, 400, 180
        ],
    })
    sample.to_csv(DATA_FILE, index=False)
    print(f"No dataset found. Created sample data: {DATA_FILE}")

# 1. Load data and standardize column names.
df = pd.read_csv(DATA_FILE)
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)

# Match common column-name variations.
aliases = {
    "region": ["region", "sales_region", "area"],
    "category": ["product_category", "category", "product_type"],
    "sales": ["sales_amount", "sales", "revenue", "total_sales"],
    "quantity": ["quantity", "qty", "units_sold"],
    "date": ["transaction_date", "date", "order_date"],
}

def find_column(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

region_col = find_column(aliases["region"])
category_col = find_column(aliases["category"])
sales_col = find_column(aliases["sales"])
quantity_col = find_column(aliases["quantity"])
date_col = find_column(aliases["date"])

required = {
    "region": region_col,
    "product category": category_col,
    "sales amount": sales_col,
}
missing = [name for name, column in required.items() if column is None]
if missing:
    raise ValueError(
        "Dataset is missing required field(s): "
        + ", ".join(missing)
        + ". Expected region, product category, and sales amount columns."
    )

# 2. Explore data.
print("Dataset shape:", df.shape)
print("\nColumns and types:")
print(df.dtypes)
print("\nFirst five rows:")
print(df.head())
print("\nMissing values:")
print(df.isna().sum())

# Clean key fields and convert values.
df[region_col] = df[region_col].astype("string").str.strip()
df[category_col] = df[category_col].astype("string").str.strip()
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

# Parse optional date and quantity columns when present.
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
if quantity_col:
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors="coerce")

# Remove rows without the fields needed for aggregation.
df = df.dropna(subset=[region_col, category_col, sales_col]).copy()
df = df[(df[region_col] != "") & (df[category_col] != "")]

# 4. Total sales by region.
region_sales = (
    df.groupby(region_col)[sales_col]
    .sum()
    .sort_values(ascending=False)
)
print("\nTotal sales by region:")
print(region_sales)

# 6. Top performing region(s).
top_sales = region_sales.max()
top_regions = region_sales[region_sales == top_sales]
print("\nTop-performing region(s):")
for region, total in top_regions.items():
    print(f"{region}: {total:,.2f}")

# 7. Sales by region and product category.
region_category_sales = (
    df.pivot_table(
        index=region_col,
        columns=category_col,
        values=sales_col,
        aggfunc="sum",
        fill_value=0,
    )
)
print("\nSales by region and product category:")
print(region_category_sales)

# Save aggregation tables.
region_sales.rename("total_sales").to_csv(FOLDER / "sales_by_region.csv")
region_category_sales.to_csv(FOLDER / "sales_by_region_category.csv")

# 5. Bar chart and pie chart: sales distribution by region.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

region_sales.plot(kind="bar", ax=axes[0], color="cornflowerblue")
axes[0].set_title("Total Sales by Region")
axes[0].set_xlabel("Region")
axes[0].set_ylabel("Sales Amount")
axes[0].tick_params(axis="x", rotation=35)

region_sales.plot(
    kind="pie",
    ax=axes[1],
    autopct="%1.1f%%",
    startangle=90,
)
axes[1].set_title("Share of Sales by Region")
axes[1].set_ylabel("")

fig.tight_layout()
fig.savefig(FOLDER / "sales_by_region.png", dpi=150)

# 8. Stacked bar chart: category sales comparison by region.
region_category_sales.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    colormap="tab20",
)
plt.title("Sales by Region and Product Category")
plt.xlabel("Region")
plt.ylabel("Sales Amount")
plt.xticks(rotation=35)
plt.legend(title="Product Category", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FOLDER / "sales_by_region_category.png", dpi=150)

plt.show()

print("\nSaved aggregation tables and charts in:", FOLDER)