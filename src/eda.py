import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Step 1: Load the cleaned dataset

data_path = Path("data") / "cleaned_churn_data.csv"
df = pd.read_csv(data_path)

# Step 2: Display basic statistics

print("===== BASIC STATISTICS =====")
print(df.describe())

# Step 3: Display the number of customers who stayed or left

print("\n===== CUSTOMER CHURN COUNT =====")
print(df["Churn"].value_counts())

# Step 4: Calculate the churn percentage

churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("\n===== CUSTOMER CHURN PERCENTAGE =====")
print(churn_percentage.round(2))

# Step 5: Compare churn across contract types

print("\n===== CHURN BY CONTRACT TYPE =====")
print(pd.crosstab(df["Contract"], df["Churn"]))

# Step 6: Calculate churn percentage for each contract type

contract_churn = pd.crosstab(
df["Contract"],
df["Churn"],
normalize="index"
) * 100

print("\n===== CHURN RATE BY CONTRACT TYPE =====")
print(contract_churn.round(2))

# Step 7: Plot churn counts by contract type

contract_counts = pd.crosstab(df["Contract"], df["Churn"])

contract_counts.plot(
kind="bar",
figsize=(8, 5)
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

# Step 8: Display the chart

plt.show()
# Step 9: Visualize churn by customer tenure

plt.figure(figsize=(8, 5))

df[df["Churn"] == "Yes"]["tenure"].hist(
    bins=20,
    alpha=0.7,
    label="Churned"
)

df[df["Churn"] == "No"]["tenure"].hist(
    bins=20,
    alpha=0.7,
    label="Stayed"
)

plt.title("Customer Churn by Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.show()


# Step 10: Visualize monthly charges by churn status

plt.figure(figsize=(8, 5))

df.boxplot(
    column="MonthlyCharges",
    by="Churn"
)

plt.title("Monthly Charges by Churn Status")
plt.suptitle("")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()