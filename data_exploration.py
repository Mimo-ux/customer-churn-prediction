import pandas as pd
from pathlib import Path

# Step 1: Locate the dataset

file_path = Path("data") / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Step 2: Check whether the dataset exists

if not file_path.exists():
    print("Dataset not found!")
    print("Expected location:", file_path.resolve())

else:
# Step 3: Load the dataset
    df = pd.read_csv(file_path)


# Step 4: Display the first five rows
    print("\n===== FIRST FIVE ROWS =====")
    print(df.head())

# Step 5: Display the number of rows and columns
print("\n===== DATASET SHAPE =====")
print(df.shape)

# Step 6: Display column names
print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

# Step 7: Display dataset information
print("\n===== DATASET INFORMATION =====")
df.info()

# Step 8: Check missing values
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# Step 9: Check duplicate rows
print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# Step 10: Display customer churn distribution
print("\n===== CHURN DISTRIBUTION =====")
print(df["Churn"].value_counts())

