import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of salaries
plt.figure(figsize=(10, 6))
sns.histplot(df['salary'], kde=True, bins=30)
plt.title("Salary Distribution")
plt.show()