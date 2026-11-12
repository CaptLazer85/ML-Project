
# FINAL PROJECT CODE: LOG-LINEAR REGRESSION MODEL


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch


# STEP 1: LOAD DATASET

df = pd.read_csv(
    r'C:\Users\91942\Desktop\ML PROJECT FINAL\MACHINE LEARNING MINOR PROJECT\FLUID CLASSIFICATION DATSET.csv'
)

print("Dataset Loaded Successfully")


# STEP 2: ADD 10% NOISE

np.random.seed(42)

noise = np.random.normal(0, 0.10, len(df))

df['Nusselt Number (Nu)'] = (
    df['Nusselt Number (Nu)'] * (1 + noise)
)

df['Nusselt Number (Nu)'] = np.clip(
    df['Nusselt Number (Nu)'],
    1e-6,
    None
)


# STEP 3: FEATURES & TARGET

X = pd.DataFrame({
    'log_Re': np.log(df['Reynolds Number (Re)']),
    'log_Pr': np.log(df['Prandtl Number (Pr)'])
})

y = np.log(df['Nusselt Number (Nu)'])


# STEP 4: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: MODEL

print("\n")
print("LOG-LINEAR REGRESSION MODEL")


model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# Train model
model.fit(X_train, y_train)


# STEP 6: TRAINING PREDICTION

y_train_pred_log = model.predict(X_train)

y_train_actual = np.exp(y_train)
y_train_pred = np.exp(y_train_pred_log)


# STEP 7: TESTING PREDICTION

y_pred_log = model.predict(X_test)

y_actual = np.exp(y_test)
y_pred = np.exp(y_pred_log)


# STEP 8: GENERALIZATION CHECK

print("\n GENERALIZATION CHECK ")

# Training metrics
train_r2 = r2_score(y_train_actual, y_train_pred)
train_mae = mean_absolute_error(y_train_actual, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
train_mape = np.mean(
    np.abs((y_train_actual - y_train_pred) / y_train_actual)
) * 100

print("\n Training Performance ")
print(f"R² Score : {train_r2:.4f}")
print(f"MAE      : {train_mae:.4f}")
print(f"RMSE     : {train_rmse:.4f}")
print(f"MAPE     : {train_mape:.2f}%")

# Testing metrics
test_r2 = r2_score(y_actual, y_pred)
test_mae = mean_absolute_error(y_actual, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
test_mape = np.mean(
    np.abs((y_actual - y_pred) / y_actual)
) * 100

print("\n Testing Performance ")
print(f"R² Score : {test_r2:.4f}")
print(f"MAE      : {test_mae:.4f}")
print(f"RMSE     : {test_rmse:.4f}")
print(f"MAPE     : {test_mape:.2f}%")


# STEP 9: CROSS VALIDATION

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='r2'
)

print("\n CROSS VALIDATION ")
print("R² Scores:", cv_scores)
print(f"Average CV R²: {cv_scores.mean():.4f}")

# STEP 10: FLUID-WISE TEST PERFORMANCE

print("\n FLUID-WISE TEST PERFORMANCE ")

test_indices = X_test.index
results_df = df.loc[test_indices].copy()

results_df['Actual Nu'] = y_actual.values
results_df['Predicted Nu'] = y_pred

for fluid in results_df['Fluid'].unique():

    fluid_data = results_df[
        results_df['Fluid'] == fluid
    ]

    mae_fluid = mean_absolute_error(
        fluid_data['Actual Nu'],
        fluid_data['Predicted Nu']
    )

    rmse_fluid = np.sqrt(mean_squared_error(
        fluid_data['Actual Nu'],
        fluid_data['Predicted Nu']
    ))

    mape_fluid = np.mean(
        np.abs(
            (fluid_data['Actual Nu'] - fluid_data['Predicted Nu'])
            / fluid_data['Actual Nu']
        )
    ) * 100

    r2_fluid = r2_score(
        fluid_data['Actual Nu'],
        fluid_data['Predicted Nu']
    )

    nrmse = (
        rmse_fluid /
        (
            fluid_data['Actual Nu'].max()
            - fluid_data['Actual Nu'].min()
        )
    ) * 100

    print(f"\nFluid: {fluid}")
    print(f"R²    : {r2_fluid:.4f}")
    print(f"MAE   : {mae_fluid:.4f}")
    print(f"RMSE  : {rmse_fluid:.4f}")
    print(f"MAPE  : {mape_fluid:.2f}%")
    print(f"NRMSE : {nrmse:.2f}%")


# STEP 11: FINAL HYPERPLANE EQUATION

scaler = model.named_steps['standardscaler']
regressor = model.named_steps['linearregression']

scaled_coeffs = regressor.coef_
scaled_intercept = regressor.intercept_

coeffs = scaled_coeffs / scaler.scale_

intercept = (
    scaled_intercept
    - np.sum((scaled_coeffs * scaler.mean_) / scaler.scale_)
)

print("\n FINAL LOG-LINEAR HYPERPLANE EQUATION ")
print(
    f"log(Nu) = {intercept:.4f} "
    f"+ ({coeffs[0]:.4f}) * log(Re) "
    f"+ ({coeffs[1]:.4f}) * log(Pr)"
)


# STEP 12: PLOT 1
# ACTUAL vs PREDICTED

plt.figure(figsize=(8, 6))

plt.scatter(
    y_actual,
    y_pred,
    alpha=0.4,
    label='Predicted Data'
)

plt.plot(
    [y_actual.min(), y_actual.max()],
    [y_actual.min(), y_actual.max()],
    'r--',
    linewidth=2,
    label='Ideal Prediction (y = x)'
)

plt.xlabel("Actual Nusselt Number (Nu)")
plt.ylabel("Predicted Nusselt Number (Nu)")
plt.title("MODEL: Actual vs Predicted Nu")
plt.legend()
plt.grid(True)
plt.show()


# STEP 13: PLOT 2
# RESIDUAL PLOT

residuals = y_actual - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.4,
    color='blue',
    label='Residual Points'
)

plt.axhline(
    0,
    color='red',
    linestyle='--',
    linewidth=2,
    label='Zero Error Line'
)

plt.xlabel("Predicted Nusselt Number (Nu)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("MODEL: Residual Plot")
plt.legend()
plt.grid(True)
plt.show()


# STEP 14: PLOT 3
# FLAT HYPERPLANE

log_re_range = np.linspace(
    X['log_Re'].min(),
    X['log_Re'].max(),
    50
)

log_pr_range = np.linspace(
    X['log_Pr'].min(),
    X['log_Pr'].max(),
    50
)

log_Re_grid, log_Pr_grid = np.meshgrid(
    log_re_range,
    log_pr_range
)

grid_data = pd.DataFrame({
    'log_Re': log_Re_grid.ravel(),
    'log_Pr': log_Pr_grid.ravel()
})

log_Nu_pred = model.predict(grid_data)

log_Nu_pred = log_Nu_pred.reshape(
    log_Re_grid.shape
)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(
    log_Re_grid,
    log_Pr_grid,
    log_Nu_pred,
    cmap='viridis',
    alpha=0.8
)

ax.set_xlabel("log(Re)")
ax.set_ylabel("log(Pr)")
ax.set_zlabel("log(Nu)")
ax.set_title("MODEL: Flat Hyperplane")

surface_proxy = Patch(
    facecolor='green',
    edgecolor='black',
    label='Log-Linear Regression Hyperplane'
)

ax.legend(
    handles=[surface_proxy],
    loc='upper left'
)

fig.colorbar(surface, shrink=0.5, aspect=5)

plt.show()


# STEP 15: SAVE MODEL

joblib.dump(
    model,
    "best_log_linear_model.pkl"
)

print("\nModel saved successfully.")


# STEP 16: USER PREDICTION

print("\nEnter values for prediction:")

velocity = float(input("Enter Velocity (U) in m/s: "))
length = float(input("Enter Plate Length (L) in m: "))
nu = float(input("Enter Kinematic Viscosity (nu) in m²/s: "))
pr = float(input("Enter Prandtl Number (Pr): "))
k = float(input("Enter Thermal Conductivity (k) in W/m·K: "))

re = (velocity * length) / nu

print(f"\nCalculated Reynolds Number = {re:.2f}")

if re > 0 and pr > 0:

    new_data = pd.DataFrame({
        'log_Re': [np.log(re)],
        'log_Pr': [np.log(pr)]
    })

    predicted_log_nu = model.predict(new_data)[0]

    predicted_nu = np.exp(predicted_log_nu)

    h = (predicted_nu * k) / length

    print(f"\nPredicted Nusselt Number = {predicted_nu:.4f}")
    print(f"Predicted Heat Transfer Coefficient = {h:.4f} W/m²K")

else:
    print("Invalid input.")
