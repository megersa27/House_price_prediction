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

# 1. Daataa fe'achuu
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("--- Daataa Jalqabaa (First 5 Rows) ---")
print(df.head())

# 2. Missing values qorachuu fi features/target adda baasuu
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Normalization / Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modelii Leenjisuu
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("\nModeliin milkaa'inaadhaan leenjifiameera!")

# 5. Madaallii (Evaluation)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Bu'aa Madaallii Modelichaa ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R² Score): {r2:.4f}")

# 6. Fakkii (Visualization)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gatii Dhugaa (Actual Price)')
plt.ylabel('Gatii Tilmaamame (Predicted Price)')
plt.title('Actual vs Predicted House Prices')
plt.grid(True)
plt.show()

# ==========================================
# 7. DEETAA HAARAAF TILMAAMA BAASUU (New Prediction)
# ==========================================
print("\n[Hojii irra jira] Deetaan haaraan tilmaamamaa jira...")
new_houses = np.array([
    [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23],  # Mana 1ffaa
    [3.2500, 25.0, 5.0000, 1.1000, 500.0, 3.0000, 34.05, -118.24]   # Mana 2ffaa
])

# Scaling gochuu
new_houses_scaled = scaler.transform(new_houses)

# Tilmaama baasuu
predicted_prices = model.predict(new_houses_scaled)

# Bu'aa print gochuu
print("\n--- Gatiiwwan Mana Haaraa Tilmaamaman ---")
for i, price in enumerate(predicted_prices):
    actual_estimated_price = price * 100000 
    print(f"Mana {i+1}: ${actual_estimated_price:,.2f}")