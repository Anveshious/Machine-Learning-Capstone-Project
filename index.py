import pandas as pd

# Load dataset
df = pd.read_csv("expected_ctc.csv")

# Check first 5 rows
print(df.head())

# Check data types & missing values
print(df.info())

# Summary statistics
print(df.describe())


#EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of salaries
plt.figure(figsize=(10, 6))
sns.histplot(df['salary'], kde=True, bins=30)
plt.title("Salary Distribution")
plt.show()

# Boxplot for salary
sns.boxplot(x=df['salary'])
plt.title("Salary Outliers")
plt.show()

# Example: Average salary by education level
sns.barplot(x='education', y='salary', data=df)
plt.xticks(rotation=45)
plt.title("Salary by Education Level")
plt.show()

# Numeric features only
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True)
plt.title("Feature Correlations")
plt.show()

# Check missing values
print(df.isnull().sum())

# Option 1: Drop missing rows
df.dropna(inplace=True)

# Option 2: Fill missing values (for numerical)
df.fillna(df.median(), inplace=True)

# One-Hot Encoding for job roles, education, etc.
df = pd.get_dummies(df, columns=['job_role', 'education', 'department'])

from sklearn.model_selection import train_test_split

X = df.drop('salary', axis=1)  # Features
y = df['salary']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)