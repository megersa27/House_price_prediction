import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Daataa fe'achuu
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Saamuda daataa ilaaluu
print("--- Daataa Jalqabaa (First 5 Rows) ---")
print(df.head())

# Missing values qabaachuu isaa qorachuu
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Features (X) fi Target Variable (y) adda baasuu
X = df.drop(columns=['MedHouseVal'])  # MedHouseVal = Gatii manaa (Target)
y = df['MedHouseVal']

# Daataa leenjii (80%) fi qormaataa (20%)tti qooduu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Normalization / Scaling (Daataa lakkoofsaa wal-qixxeessuu)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelii Linear Regression uumuu fi leenjisuu
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\nModeliin milkaa'inaadhaan leenjifiameera!")

# Daataa qormaataa (X_test) irratti tilmaama gochuu
y_pred = model.predict(X_test_scaled)

# Reddoolee madaallii herreguu
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Bu'aa Madaallii Modelichaa ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R² Score): {r2:.4f}")

# Gatii dhugaa fi gatii tilmaamame wal-bira qabanii ilaaluu
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gatii Dhugaa (Actual Price)')
plt.ylabel('Gatii Tilmaamame (Predicted Price)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()

