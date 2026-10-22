import pandas as pd
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# Step 1: Load the prepared datasets
data_folder = Path("data")

X_train = pd.read_csv(data_folder / "X_train.csv")
X_test = pd.read_csv(data_folder / "X_test.csv")

y_train = pd.read_csv(data_folder / "y_train.csv")["Churn"]
y_test = pd.read_csv(data_folder / "y_test.csv")["Churn"]

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Step 2: Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Step 3: Train the model
model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# Step 4: Predict customer churn
y_pred = model.predict(X_test)

# Step 5: Calculate churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== RANDOM FOREST PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Step 7: Display the confusion matrix
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# Step 8: Display the classification report
print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"],
        zero_division=0
    )
)

# Step 9: Display the most important features
feature_importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\n===== TOP 10 IMPORTANT FEATURES =====")
print(feature_importance.head(10).round(4))
# Step 10: Save the trained model
model_path = Path("src") / "random_forest_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Model location:", model_path)
