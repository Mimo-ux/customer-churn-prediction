import pandas as pd
from pathlib import Path

# Step 1: Define the dataset paths

data_path = Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
output_path = Path("data") / "cleaned_churn_data.csv"

# Step 2: Load the dataset

df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# Step 3: Remove duplicate rows

df = df.drop_duplicates()

# Step 4: Remove unnecessary spaces from column names

df.columns = df.columns.str.strip()

# Step 5: Convert TotalCharges to numeric values

df["TotalCharges"] = pd.to_numeric(
df["TotalCharges"],
errors="coerce"
)

# Step 6: Check for missing values

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Step 7: Remove rows with missing values

df = df.dropna()

# Step 8: Save the cleaned dataset

df.to_csv(output_path, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Missing values after cleaning:", df.isnull().sum().sum())
print("\nCleaned dataset saved to:", output_path)
