import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
roc_auc_score
)

# Step 1: Define the data folder

data_folder = Path("data")

# Step 2: Load the training and testing datasets

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(data_folder / "y_train.csv")["Churn"]
y_test = pd.read_csv(data_folder / "y_test.csv")["Churn"]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 3: Create a machine learning pipeline

model = Pipeline([
("scaler", StandardScaler()),
("classifier", LogisticRegression(
max_iter=3000,
random_state=42
))
])

# Step 4: Train the model

model.fit(X_train, y_train)

print("\nModel training completed successfully!")

# Step 5: Predict churn

y_pred = model.predict(X_test)

# Step 6: Calculate churn probabilities

y_prob = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluate performance

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Step 8: Display the confusion matrix

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# Step 9: Display the classification report

print("\n===== CLASSIFICATION REPORT =====")
print(
classification_report(
y_test,
y_pred,
target_names=["Stayed", "Churned"],
zero_division=0
)
)
