import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Total samples
N = 10000

# Empty lists
fluid_list = []
U_list = []
L_list = []
nu_list = []
Re_list = []
Pr_list = []
k_list = []
Nu_list = []
h_list = []

while len(U_list) < N:

    # Random fluid class selection
    fluid_type = np.random.choice([
        'Air', 'Water', 'Oil'
    ])

    # Velocity (m/s)
    U = np.random.uniform(0.5, 50)

    # Plate Length (m)
    L = np.random.uniform(0.1, 1.0)

    # Kinematic viscosity (m²/s)
    nu = np.random.uniform(1e-6, 5e-4)

    # Assign realistic Pr and k based on fluid
    if fluid_type == 'Air':
        Pr = np.random.uniform(0.6, 1.0)
        k = np.random.uniform(0.02, 0.08)

    elif fluid_type == 'Water':
        Pr = np.random.uniform(2, 10)
        k = np.random.uniform(0.2, 1.0)

    elif fluid_type == 'Oil':
        Pr = np.random.uniform(10, 500)
        k = np.random.uniform(0.1, 0.2)

    # Reynolds Number
    Re = (U * L) / nu

    # Keep only laminar samples
    if Re < 5e5:

        # Laminar flat plate correlation
        Nu = 0.664 * (Re ** 0.5) * (Pr ** (1 / 3))

        # Heat transfer coefficient
        h = (Nu * k) / L

        # Store values
        fluid_list.append(fluid_type)
        U_list.append(U)
        L_list.append(L)
        nu_list.append(nu)
        Re_list.append(Re)
        Pr_list.append(Pr)
        k_list.append(k)
        Nu_list.append(Nu)
        h_list.append(h)

# Create DataFrame
data = pd.DataFrame({
    'Fluid': fluid_list,
    'Velocity (U)': U_list,
    'Length (L)': L_list,
    'Kinematic Viscosity (nu)': nu_list,
    'Reynolds Number (Re)': Re_list,
    'Prandtl Number (Pr)': Pr_list,
    'Thermal Conductivity (k)': k_list,
    'Nusselt Number (Nu)': Nu_list,
    'Heat Transfer Coefficient (h)': h_list
})

# Save CSV
data.to_csv("TRIAL.csv", index=False)

# Output
print("Dataset generated successfully.")
print("\nFirst 5 Rows:")
print(data.head())

print("\nFluid-wise sample count:")
print(data['Fluid'].value_counts())

print("\nTotal samples:", len(data))
