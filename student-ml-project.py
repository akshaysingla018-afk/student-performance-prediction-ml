import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("Student_performance_data _.csv")

# Check data
print(data.head())

# Features and Target
X = data[['StudyTimeWeekly', 'Absences', 'ParentalEducation', 'Tutoring']]
y = data['GPA']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# -------------------------------
# 🔹 Linear Regression Model
# -------------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\n--- Linear Regression ---")
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("R2 Score:", r2_score(y_test, lr_pred))

# Feature Importance (coefficients)
print("Coefficients:", lr_model.coef_)

# -------------------------------
# 🔹 Decision Tree Model
# -------------------------------
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("MAE:", mean_absolute_error(y_test, dt_pred))
print("R2 Score:", r2_score(y_test, dt_pred))

# -------------------------------
# 📊 Plot (Linear Regression)
# -------------------------------
plt.scatter(y_test, lr_pred)
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.title("Actual vs Predicted GPA (Linear Regression)")
plt.show()