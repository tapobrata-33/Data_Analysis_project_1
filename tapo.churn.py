import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your file
df = pd.read_csv("tapo.churn.csv")

# Show data
print(df.head())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna()

# Show churn count
print("\nChurn Count:")
print(df['Churn'].value_counts())

# Graph 1: Churn Count
sns.countplot(x='Churn', data=df)
plt.show()

# Graph 2: Contract vs Churn
sns.countplot(x='Contract', hue='Churn', data=df)
plt.show()

# Graph 3: Monthly Charges
sns.histplot(df['MonthlyCharges'], kde=True)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Convert churn to 0/1
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Select features
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
print("\nModel Accuracy:", model.score(X_test, y_test))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("tapo.churn.csv")

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df = df.dropna()

# Show data
print(df.head())

# Churn count
print("\nChurn Count:")
print(df['Churn'].value_counts())

# Graphs
sns.countplot(x='Churn', data=df)
plt.show()

sns.countplot(x='Contract', hue='Churn', data=df)
plt.show()

sns.histplot(df['MonthlyCharges'], kde=True)
plt.show()

# ===============================
# MACHINE LEARNING PART
# ===============================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Convert Churn to 0/1
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

# Features and target
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
print("\n==============================")
print("Model Accuracy:", model.score(X_test, y_test))
print("==============================")
plt.show()
plt.savefig("graph.png")