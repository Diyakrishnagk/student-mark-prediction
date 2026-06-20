# ============================================
# STUDENT MARKS PREDICTION WITH VISUALIZATION
# ============================================

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set style
sns.set(style="darkgrid")

# 2. LOAD DATA
df = pd.read_csv("student_info.csv")

print("Dataset Preview:\n", df.head())

# 3. DATA CLEANING
df['study_hours'] = df['study_hours'].fillna(df['study_hours'].mean())

# ============================================
# 🔥 4. ADVANCED VISUALIZATIONS
# ============================================

# 1. Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x='study_hours', y='student_marks', data=df)
plt.title("Study Hours vs Marks")
plt.show()

# 2. Regression Plot (VERY IMPORTANT)
plt.figure(figsize=(6,4))
sns.regplot(x='study_hours', y='student_marks', data=df, line_kws={"color": "red"})
plt.title("Regression Line")
plt.show()

# 3. Distribution Plot
plt.figure(figsize=(6,4))
sns.histplot(df['student_marks'], kde=True)
plt.title("Marks Distribution")
plt.show()

# 4. Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(y=df['student_marks'])
plt.title("Outlier Detection")
plt.show()

# 5. Correlation Heatmap
plt.figure(figsize=(5,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ============================================
# 5. MODEL BUILDING
# ============================================

X = df[['study_hours']]
y = df['student_marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
# 6. PREDICTION & EVALUATION
# ============================================

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ============================================
# 7. FINAL VISUAL (ACTUAL vs PREDICTED)
# ============================================

plt.figure(figsize=(6,4))
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.scatter(X_test, y_pred, color='red', label="Predicted")
plt.legend()
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Actual vs Predicted")
plt.show()

# ============================================
# 8. USER INPUT
# ============================================

hours = float(input("\nEnter study hours: "))
prediction = model.predict([[hours]])

print(f"Predicted Marks: {prediction[0]:.2f}")