import pandas as pd

# Load dataset
df = pd.read_csv("expected_ctv.csv")

# Check first 5 rows
print(df.head())

# Check data types & missing values
print(df.info())

# Summary statistics
print(df.describe())