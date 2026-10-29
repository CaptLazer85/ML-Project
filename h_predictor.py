import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================
# STEP 1: LOAD DATASET
# ==========================
df = pd.read_csv(r'C:\Users\hp\Desktop\ML project\flat_plate_heat_transfer_dataset.csv')

print("Dataset Loaded Successfully")
print("\nColumns:")
print(df.columns.tolist())

# ==========================
# STEP 2: CREATE KINEMATIC VISCOSITY (nu)
# ==========================
# Using: Re = U*L / nu  =>  nu = U*L / Re

df['Kinematic Viscosity (nu)'] = (
    df['Velocity (U)'] * df['Length (L)']
) / df['Reynolds Number (Re)']

# ==========================
# STEP 3: FEATURES AND TARGET
# ==========================
# Correct ML inputs:
# Velocity, Length, Kinematic Viscosity, Prandtl

X = df[['Velocity (U)',
        'Length (L)',
        'Kinematic Viscosity (nu)',
        'Prandtl Number (Pr)']]

y = df['Heat Transfer Coefficient (h)']

print("\nSelected Features:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

# ==========================
# STEP 4: TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# STEP 5: POLYNOMIAL REGRESSION MODEL
# ==========================
model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

# Train model
model.fit(X_train, y_train)

# ==========================
# STEP 6: PREDICT TEST DATA
# ==========================
y_pred = model.predict(X_test)

# ==========================
# STEP 7: MODEL EVALUATION
# ==========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MODEL PERFORMANCE =====")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ==========================
# STEP 8: PLOT ACTUAL VS PREDICTED
# ==========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    linewidth=2
)

plt.xlabel("Actual Heat Transfer Coefficient (h)")
plt.ylabel("Predicted Heat Transfer Coefficient (h)")
plt.title("Actual vs Predicted Heat Transfer Coefficient")
plt.grid(True)
plt.show()

# ==========================
# STEP 9: SAVE MODEL
# ==========================
joblib.dump(model, "heat_transfer_poly_model.pkl")
print("\nModel saved as heat_transfer_poly_model.pkl")

# ==========================
# STEP 10: USER INPUT FOR NEW PREDICTION
# ==========================
print("\nEnter values for new prediction:")

velocity = float(input("Enter Velocity (U) in m/s: "))
length = float(input("Enter Plate Length (L) in m: "))
nu = float(input("Enter Kinematic Viscosity (nu) in m²/s: "))
pr = float(input("Enter Prandtl Number (Pr): "))

new_data = pd.DataFrame([{
    'Velocity (U)': velocity,
    'Length (L)': length,
    'Kinematic Viscosity (nu)': nu,
    'Prandtl Number (Pr)': pr
}])

prediction = model.predict(new_data)

print(f"\nPredicted Heat Transfer Coefficient = {prediction[0]:.4f} W/m²K")