import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Load the cleaned dataset

data_path = Path("data") / "cleaned_churn_data.csv"
df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)

# Step 2: Remove customerID

df = df.drop(columns=["customerID"])

# Step 3: Separate input features and target

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# Step 4: Convert categorical features into numerical features

X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Step 5: Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# Step 6: Display the results

print("\n===== DATA SPLITTING RESULTS =====")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\n===== TARGET DISTRIBUTION =====")
print("Training churn percentage:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\nTesting churn percentage:")
print((y_test.value_counts(normalize=True) * 100).round(2))

# Step 7: Save the prepared datasets

X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False, header=["Churn"])
y_test.to_csv("data/y_test.csv", index=False, header=["Churn"])

print("\nPrepared datasets saved successfully!")
