import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("tapo.churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
print(df.head())
print("\nChurn Count:")
print(df['Churn'].value_counts())
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Count")
plt.savefig("churn_count.png")
plt.show()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.savefig("contract_churn.png")
plt.show()
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Monthly Charges Distribution")
plt.savefig("monthly_charges.png")
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("\n==============================")
print("Model Accuracy:", accuracy)
print("==============================")