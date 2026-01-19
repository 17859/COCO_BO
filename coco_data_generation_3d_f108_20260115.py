import numpy as np
import pandas as pd
import cocoex  # Requires: pip install cocoex
from scipy.optimize import minimize, LinearConstraint, Bounds

# ==========================================
# 1. CONFIGURATION
# ==========================================
N_SAMPLES = 2000            # Total size of dataset
DIM = 3                  # Total ingredients (dimensions)
FILENAME = "chemical_sparse_data_f108_3d_20260115_1.csv" #我將FILENAME從f101改成f109_20260115
INSTANCE_ID = 1             # Fix instance ID so geometry (rotation/shift) is consistent
SEED = 42                   # Reproducibility

np.random.seed(SEED)

print(f"--- Setting up COCO Environment (D={DIM}) ---")

# ==========================================
# 2. SETUP ORACLES (God View vs. Agent View)
# ==========================================

# A. God View: Noise-free oracle (Used for Regret Calculation)
# Function indices: 9 = Rosenbrock (Noise-free)
suite_noisefree = cocoex.Suite(
    suite_name="bbob",
    suite_instance=f"instances:{INSTANCE_ID}",
    suite_options=f"function_indices:8 dimensions:{DIM}"
)
problem_noisefree = suite_noisefree.next_problem()

# B. Agent View: Noisy oracle (Used for Training Data)
# Function indices: 109 = Rosenbrock with moderate noise (f109)
suite_noisy = cocoex.Suite(
    suite_name="bbob-noisy",
    suite_instance=f"instances:{INSTANCE_ID}",
    suite_options=f"function_indices:104 dimensions:{DIM}"
)

problem_noisy = None
for p in suite_noisy:
    if "f104" in p.id:
        problem_noisy = p
        break


print('---'*50)
print(f"God View Oracle:   {problem_noisefree}")
print('---'*50)
print('***'*50)
print(f"Agent View Oracle: {problem_noisy}")
print('***'*50)

# ==========================================
# 3. CALCULATE TRUE CONSTRAINED OPTIMUM
# ==========================================
# We must find the minimum of f9 (Rosenbrock) on the SIMPLEX, not the whole domain.
# Math: sum(x_chem) = 1  =>  sum((x_math + 5) / 10) = 1  =>  sum(x_math) = 10 - 5*D

print("\n--- Calculating True Constrained Optimum (Baseline) ---")

target_sum_math = 10.0 - (DIM * 5.0)  # e.g., -15 for D=5

# Define Constraint: sum(x) = target_sum_math
linear_constraint = LinearConstraint(
    np.ones(DIM),         # Vector of 1s
    [target_sum_math],    # Lower bound
    [target_sum_math]     # Upper bound
)

# Define Bounds: COCO [-5, 5] matches Simplex [0, 1]
bounds_math = Bounds([-5.0]*DIM, [5.0]*DIM)

# Optimization Helper
def objective_func(x):
    return problem_noisefree(x)

# Start search from the centroid of the simplex (mapped to math space)
# Centroid: x_chem = 1/D => x_math = 10*(1/D) - 5
start_val = 10.0 * (1.0/DIM) - 5.0
x0 = np.full(DIM, start_val)

# Run SLSQP (Sequential Least SQuares Programming) - handles equality constraints well
res = minimize(
    objective_func,
    x0,
    method='SLSQP',
    bounds=bounds_math,
    constraints=[linear_constraint],
    options={'ftol': 1e-9, 'disp': False}
)

if not res.success:
    print(f"⚠️ Warning: Optimization did not converge fully: {res.message}")

true_optimum = res.fun
true_optimum_x_math = res.x
true_optimum_x_chem = (res.x + 5.0) / 10.0

print(f"Constraint Check (sum x_math): {np.sum(res.x):.4f} (Target: {target_sum_math})")
print(f"True Constrained Minimum (f*): {true_optimum:.6f}")
print(f"Location (Simplex Space): {np.round(true_optimum_x_chem[:5], 3)}... (first 5 dims)")

# ==========================================
# 4. DATA GENERATION LOOP
# ==========================================
print(f"\n--- Generating {N_SAMPLES} Sparse Mixture Samples ---")

X_list = []
y_true_list = []
y_noisy_list = []
k_counts = []

for i in range(N_SAMPLES):
    if i % 500 == 0:
        print(f"  Generating sample {i}/{N_SAMPLES}...")

    # -----------------------------
    # Step A: Generate Chemical Recipe (x_chem)
    # -----------------------------
    # 1. Choose Sparsity k in [2, 5] (adjusted for D=5)
    k = np.random.randint(low=1, high=4)
    k_counts.append(k)
    
    # 2. Choose Active Ingredients
    active_indices = np.random.choice(DIM, size=k, replace=False)
    
    # 3. Sample values (Dirichlet on active set ensures sum=1)
    fractions = np.random.dirichlet(alpha=np.ones(k))
    
    # 4. Construct Sparse Vector
    x_chem = np.zeros(DIM)
    x_chem[active_indices] = fractions
    X_list.append(x_chem)
    
    # -----------------------------
    # Step B: Map to Math Space
    # -----------------------------
    # Map [0, 1] -> [-5, 5]
    x_math = (x_chem * 10.0) - 5.0
    
    # -----------------------------
    # Step C: Evaluate Both Worlds
    # -----------------------------
    # 1. God View (Hidden Truth)
    y_true = problem_noisefree(x_math)
    y_true_list.append(y_true)
    
    # 2. Agent View (Noisy Observation)
    y_noisy = problem_noisy(x_math)
    y_noisy_list.append(y_noisy)

# ==========================================
# 5. SAVE & VALIDATE
# ==========================================
# Construct DataFrame
cols = [f'param_{i+1}' for i in range(DIM)]
df = pd.DataFrame(X_list, columns=cols)

# Add Metadata
df['num_active'] = k_counts
df['energy_observed'] = y_noisy_list  # Training Target
df['energy_true'] = y_true_list       # Validation Metric (Hidden)

# Save to CSV
df.to_csv(FILENAME, index=False)

# Print Stats
best_idx = np.argmin(y_true_list)
best_observed_regret = y_true_list[best_idx] - true_optimum

print(f"\n--- Dataset Created: {FILENAME} ---")
print(f"Shape: {df.shape}")
print(f"True Optimum Baseline (f*): {true_optimum:.6f}")
print(f"Best Sampled True Value:    {y_true_list[best_idx]:.6f}")
print(f"Min Initial Regret in Data: {best_observed_regret:.6f}")
print("-" * 50)
#print("REMINDER: When benchmarking, calculate Regret = energy_true(x) - f*")
#print(f"          where f* = {true_optimum:.6f}")