import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Daataa Iris fe'achuu
iris = load_iris(as_frame=True)
df = iris.frame

print("--- Daataa Iris Jalqabaa (First 5 Rows) ---")
print(df.head())

print("\n--- Gosa Daraaraa (Target Classes) ---")
print(iris.target_names)

# 2. Features (X) fi Target (y) adda baasuu
X = df.drop(columns=['target'])
y = df['target']

# Training fi Testing data qooduu (80% leenjii, 20% qormaata)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling (Data qixxeessuu)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modelii K-Nearest Neighbors (KNN) leenjisuu
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)
print("\nModeliin KNN milkaa'inaadhaan leenjifiameera!")

# 5. Madaallii (Evaluation)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Bu'aa Madaallii Modelichaa ---")
print(f"Accuracy (Sirrummaa): {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ==========================================
# 6. DEETAA HAARAAF RAMADDII BAASUU (New Prediction)
# ==========================================
# Safartuu daraaraa haaraa: [sepal length, sepal width, petal length, petal width]
new_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Fakkeenyaaf gosa Setosa ta'uu danda'a
    [5.9, 3.0, 5.1, 1.8]   # Fakkeenyaaf gosa Virginica ta'uu danda'a
])

# Scaling gochuu
new_flowers_scaled = scaler.transform(new_flowers)

# Tilmaama baasuu
predicted_classes = model.predict(new_flowers_scaled)

# Bu'aa print gochuu
print("\n--- Ramaddii Daraaraa Haaraa ---")
for i, pred_class in enumerate(predicted_classes):
    class_name = iris.target_names[pred_class]
    print(f"Daraaraan {i+1}: Gosa '{class_name}' jedhamuudha.")