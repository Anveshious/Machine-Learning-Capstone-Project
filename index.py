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

# One-Hot Encoding for job roles, education, etc.
df = pd.get_dummies(df, columns=['job_role', 'education', 'department'])

from sklearn.model_selection import train_test_split

X = df.drop('salary', axis=1)  # Features
y = df['salary']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")

from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")

from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20]
}

grid_search = GridSearchCV(RandomForestRegressor(), params, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

importances = best_model.feature_importances_
features = X_train.columns

feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print(feature_importance.head(10))

import joblib
import pandas as pd

# Load the saved model
model = joblib.load('salary_predictor.pkl')

# Create new employee data (ADJUST VALUES TO YOUR NEEDS)
new_employee = {
    'experience_years': [5],
    'education': ['Master'],
    'job_role': ['Data Scientist'],
    'projects_completed': [7]
}
new_employee_data = pd.DataFrame(new_employee)

# Preprocess identically to training data
new_employee_data = pd.get_dummies(new_employee_data)
new_employee_data = new_employee_data.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
predicted_salary = model.predict(new_employee_data)
print(f"Predicted Salary: ${predicted_salary[0]:.2f}")