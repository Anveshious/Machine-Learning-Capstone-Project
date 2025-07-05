import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# 1. Load and explore data
df = pd.read_csv("expected_ctc.csv")
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())

# 2. EDA - Focus on Expected_CTC
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['Expected_CTC'], kde=True, bins=30)
plt.title("Expected CTC Distribution")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Expected_CTC'])
plt.title("Expected CTC Outliers")
plt.tight_layout()
plt.show()

# 3. Relationship analysis
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='Education', y='Expected_CTC', data=df)
plt.xticks(rotation=45)
plt.title("Expected CTC by Education Level")

plt.subplot(2, 2, 2)
sns.scatterplot(x='Total_Experience', y='Expected_CTC', data=df)
plt.title("Expected CTC vs Total Experience")

plt.subplot(2, 2, 3)
sns.boxplot(x='Department', y='Expected_CTC', data=df)
plt.xticks(rotation=45)
plt.title("Expected CTC by Department")

plt.subplot(2, 2, 4)
sns.scatterplot(x='Current_CTC', y='Expected_CTC', data=df)
plt.title("Expected CTC vs Current CTC")
plt.tight_layout()
plt.show()

# 4. Data Preprocessing
# Drop irrelevant columns
df = df.drop(['IDX', 'Applicant_ID'], axis=1)

# Handle missing values
print("\nMissing Values Before:")
print(df.isnull().sum())

# Fill missing values
df.fillna({
    'PG_Specialization': 'Not Applicable',
    'University_PG': 'Not Applicable',
    'Passing_Year_Of_PG': 0,
    'PHD_Specialization': 'Not Applicable',
    'University_PHD': 'Not Applicable',
    'Passing_Year_Of_PHD': 0,
    'Last_Appraisal_Rating': df['Last_Appraisal_Rating'].median(),
    'Number_of_Publications': 0,
    'Certifications': 0
}, inplace=True)

print("\nMissing Values After:")
print(df.isnull().sum())

# Feature Engineering
df['Years_Since_Graduation'] = 2023 - df['Passing_Year_Of_Graduation']  # Update year as needed
df['Has_PG'] = np.where(df['PG_Specialization'] != 'Not Applicable', 1, 0)
df['Has_PhD'] = np.where(df['PHD_Specialization'] != 'Not Applicable', 1, 0)
df['Has_International_Degree'] = np.where(df['International_degree_any'] == 'Yes', 1, 0)

# Encode categorical variables
cat_cols = ['Department', 'Role', 'Industry', 'Organization', 'Designation', 
            'Education', 'Graduation_Specialization', 'University_Grad',
            'PG_Specialization', 'University_PG', 'PHD_Specialization',
            'University_PHD', 'Curent_Location', 'Preferred_location']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 5. Prepare data for modeling
X = df.drop('Expected_CTC', axis=1)
y = df['Expected_CTC']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['Total_Experience', 'Total_Experience_in_field_applied', 'Current_CTC',
            'Passing_Year_Of_Graduation', 'Passing_Year_Of_PG', 'Passing_Year_Of_PHD',
            'Last_Appraisal_Rating', 'No_Of_Companies_worked', 'Number_of_Publications',
            'Certifications', 'Years_Since_Graduation']

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Model Training
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# 7. Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                         param_grid, 
                         cv=5,
                         scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)

# 8. Evaluation
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# 9. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# 10. Save Model and Preprocessing Objects
joblib.dump(best_model, 'expected_ctc_predictor.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

# 11. Prediction Example
def predict_expected_ctc(input_data):
    # Load saved objects
    model = joblib.load('expected_ctc_predictor.pkl')
    encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply same transformations
    for col in cat_cols:
        input_df[col] = encoders[col].transform(input_df[col].astype(str))
    
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    # Ensure all columns are present
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    
    # Predict
    prediction = model.predict(input_df)
    return prediction[0]

# Example usage:
sample_input = {
    'Total_Experience': 5,
    'Total_Experience_in_field_applied': 4,
    'Department': 'IT',
    'Role': 'Data Scientist',
    'Industry': 'Technology',
    'Organization': 'Current',
    'Designation': 'Senior',
    'Education': 'Master',
    'Graduation_Specialization': 'Computer Science',
    'University_Grad': 'State University',
    'Passing_Year_Of_Graduation': 2015,
    'PG_Specialization': 'Data Science',
    'University_PG': 'Tech University',
    'Passing_Year_Of_PG': 2017,
    'PHD_Specialization': 'Not Applicable',
    'University_PHD': 'Not Applicable',
    'Passing_Year_Of_PHD': 0,
    'Curent_Location': 'Bangalore',
    'Preferred_location': 'Bangalore',
    'Current_CTC': 15.0,
    'Inhand_Offer': 'No',
    'Last_Appraisal_Rating': 4.0,
    'No_Of_Companies_worked': 2,
    'Number_of_Publications': 3,
    'Certifications': 2,
    'International_degree_any': 'No'
}

predicted_ctc = predict_expected_ctc(sample_input)
print(f"\nPredicted Expected CTC: ₹{predicted_ctc:,.2f} LPA")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Clean and prepare education categories
education_order = sorted(df['Education'].unique())
df['Education'] = df['Education'].str.strip().str