import numpy as np
import pandas as pd
import cocoex
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import warnings

# Optimization & Stat Imports
from scipy.optimize import minimize, LinearConstraint, Bounds
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.priors import GammaPrior





# Suppress Warnings
warnings.filterwarnings("ignore")

# =====================================================
# 1. Load Dataset (D_0)
# =====================================================
def load_initial_dataset(csv_path='D:/Users/TingYuLin/Desktop/py12/chemical_sparse_data_f108_3d_20260115_1.csv', n_init=None, seed=None):
    """
    Load initial dataset D_0.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find '{csv_path}'. Please run the generation script first.")

    # Extract X
    param_cols = [c for c in df.columns if 'param_' in c]
    if not param_cols:
         param_cols = [f'param_{i+1}' for i in range(5)]
    
    X_init = df[param_cols].values
    y_init_noisy = df['energy_observed'].values
    
    # God View Metadata (Hidden from BO)
    y_init_true = None
    if 'energy_true' in df.columns:
        y_init_true = df['energy_true'].values
        print(f"  ✓ Loaded 'energy_true' metadata (Hidden from BO).")
    
    # Subsample with Seeding Strategy
    if n_init is not None and n_init < len(X_init):
        if seed is not None:
            np.random.seed(seed)
        
        # --- 策略修改：混合采样 (Seeding Strategy) ---
        # CSV 中可能有一些点落在低能量区域（"宝藏"）
        # 我们不能让 Agent 错过这些信息，需要给 Agent 一个"好起点"
        
        # 1. 选出最好的 n_init//2 个点 (Best Samples) - 告诉 Agent 山谷的大致方向
        n_best = n_init // 2
        best_indices = np.argsort(y_init_noisy)[:n_best]  # 能量越低越好（COCO 最小化）
        
        # 2. 选出随机的 n_init - n_best 个点 (Random Samples) - 保持全局视野
        n_random = n_init - n_best
        remaining_indices = np.delete(np.arange(len(X_init)), best_indices)
        random_indices = np.random.choice(remaining_indices, size=n_random, replace=False)
        
        # 3. 合并
        indices = np.concatenate([best_indices, random_indices])
        
        # 应用索引
        X_init = X_init[indices]
        y_init_noisy = y_init_noisy[indices]
        if y_init_true is not None:
            y_init_true = y_init_true[indices]
            
        print(f"  Using {n_best} BEST samples + {n_random} RANDOM samples (Seeding Strategy).")
        
    return X_init, y_init_noisy, y_init_true

# =====================================================
# 2. Setup COCO Oracles 
# =====================================================
def setup_coco_oracles(instance_id=1):
    """
    Setup Oracles and calculate the TRUE CONSTRAINED OPTIMUM.
    """
    DIM = 3
    
    # 1. Initialize Oracles
    # 1. 设置 Noisy (Agent View) - Rosenbrock with moderate noise (f104)
    # Note: In bbob-noisy suite, function_indices are 1-indexed (1-30), where 4 corresponds to f104 (4 + 100 = 104)
    suite_noisy = cocoex.Suite("bbob-noisy", f"instances:{instance_id}", f"function_indices:4 dimensions:{DIM}")
    problem_noisy = suite_noisy.next_problem()
    
    # 2. 设置 Noise-free (God View) - Rosenbrock Original (f8)
    suite_noisefree = cocoex.Suite("bbob", f"instances:{instance_id}", f"function_indices:8 dimensions:{DIM}")
    problem_noisefree = suite_noisefree.next_problem()
    
    # 2. Calculate True Optimum ON THE SIMPLEX
    print("  Calculating True Constrained Optimum (Simplex)...")
    
    # Map Simplex Constraint sum(x)=1 to COCO space [-5, 5]
    # sum(x_coco) = 10*sum(x_simplex) - 5*DIM = 10 - 25 = -15 (for D=5)
    target_sum_coco = 10.0 - (5.0 * DIM) 
    
    constraint = LinearConstraint(np.ones(DIM), [target_sum_coco], [target_sum_coco])
    bounds = Bounds([-5.0]*DIM, [5.0]*DIM)
    
    # Start from center
    x0_simplex = np.ones(DIM) / DIM
    x0_coco = (x0_simplex * 10.0) - 5.0
    
    res = minimize(
        lambda x: problem_noisefree(x),
        x0_coco,
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint],
        options={'ftol': 1e-9}
    )
    
    true_optimum = res.fun
    print(f"  True Constrained Optimum f*: {true_optimum:.6f}")
    
    x_simplex = (res.x + 5.0) / 10.0
    print('best res.x = ',res.x)
    print('best_simplex_x',x_simplex)
    print('///'*50)
    print('problem_noisy = ',problem_noisy)
    print('problem_noisefree = ',problem_noisefree)
    print('true_optimum = ',true_optimum)
    print('///'*50)


    return problem_noisy, problem_noisefree, true_optimum




# =====================================================
# 3. Helpers: Mappings
# =====================================================
def simplex_to_coco(x_simplex):
    return 10.0 * np.array(x_simplex) - 5.0

def oracle_noisy(x_simplex, problem):
    x_coco = simplex_to_coco(x_simplex)
    if x_coco.ndim == 1: return float(problem(x_coco))
    return np.array([problem(x) for x in x_coco])

def oracle_noisefree(x_simplex, problem):
    x_coco = simplex_to_coco(x_simplex)
    if x_coco.ndim == 1: return float(problem(x_coco))
    return np.array([problem(x) for x in x_coco])

# =====================================================
# 4. Optimizer (Acquisition Function) - Using BoTorch optimize_acqf with Warm-Start
# =====================================================
def optimize_acquisition_simplex(acquisition_fn, train_X=None, train_Y=None, n_candidates=2000, seed=42):
    """
    Optimize ACQ function using BoTorch's optimize_acqf with warm-start initialization.
    
    Args:
        acquisition_fn: BoTorch acquisition function
        train_X: Current training inputs (for warm-start)
        train_Y: Current training outputs (for warm-start)
        n_candidates: Number of raw samples for initial conditions (legacy param, kept for compatibility)
        seed: Random seed
    
    Returns:
        best_x: Optimal point (numpy array)
        best_value: Optimal acquisition value
    """
    DIM = 3
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Define bounds: Simplex [0, 1]^D
    bounds = torch.stack([
        torch.zeros(DIM, dtype=torch.float64),
        torch.ones(DIM, dtype=torch.float64)
    ])
    
    # Define equality constraint: sum(x) = 1
    # BoTorch format: [(indices, coefficients, rhs)]
    equality_constraints = [(
        torch.arange(DIM, dtype=torch.long),  # All dimensions
        torch.ones(DIM, dtype=torch.float64),  # Coefficients (all 1s)
        1.0  # Right-hand side: sum = 1
    )]
    
    # Warm-start: Use best known point as one of the initial conditions
    initial_conditions = None
    if train_X is not None and train_Y is not None and len(train_X) > 0:
        # Find best observed point (for minimization: argmin)
        best_idx = train_Y.argmin()  # COCO minimizes, so we want minimum
        x_best_known = train_X[best_idx].clone().unsqueeze(0)  # Shape: (1, D)
        
        # Generate initial conditions using BoTorch
        # Note: This may fail for custom functions (not BoTorch acquisition objects)
        try:
            initial_conditions = gen_batch_initial_conditions(
                acq_function=acquisition_fn,
                bounds=bounds,
                q=1,
                num_restarts=10,
                raw_samples=100,
            )[0]  # Shape: (10, 1, D)
            
            # Replace first initial condition with best known point (Warm Start)
            initial_conditions[0] = x_best_known.unsqueeze(0)  # Shape: (1, 1, D)
            
            # 【关键修复】: 必须开启梯度，否则优化器无法工作
            initial_conditions.requires_grad_(True)
        except Exception:
            # For custom functions, manually create initial conditions
            # Generate random simplex points for remaining restarts
            num_restarts = 10
            initial_conditions = torch.zeros((num_restarts, 1, DIM), dtype=torch.float64)
            initial_conditions[0] = x_best_known.unsqueeze(0)  # Warm-start with best point
            
            # Generate random simplex points for rest
            for r in range(1, num_restarts):
                k = np.random.randint(1, 4)  # k ~ 2 to 5 for D=5
                idx = np.random.choice(DIM, k, replace=False)
                vals = np.random.dirichlet(np.ones(k))
                x_random = np.zeros(DIM)
                x_random[idx] = vals
                initial_conditions[r, 0] = torch.tensor(x_random, dtype=torch.float64)
            
            # 【关键修复】: 必须开启梯度，否则优化器无法工作
            initial_conditions.requires_grad_(True)
    
    # Optimize acquisition function
    try:
        best_predicted_x, best_acq_value = optimize_acqf(
            acq_function=acquisition_fn,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            equality_constraints=equality_constraints,
            batch_initial_conditions=initial_conditions,
            options={"maxiter": 200}  # Limit iterations for speed
        )
        
        # Extract result (best_predicted_x shape: (1, 1, D))
        best_x = best_predicted_x.squeeze().detach().cpu().numpy()  # Shape: (D,)
        best_value = best_acq_value.item()
        
        return best_x, best_value
        
    except Exception as e:
        # Fallback to random shooting if optimize_acqf fails
        print(f"  ⚠️ optimize_acqf failed ({e}), falling back to random shooting...")
        np.random.seed(seed)
        best_value = -np.inf
        best_x = None
        
        batch_size = 100
        n_batches = n_candidates // batch_size
        
        for b in range(n_batches):
            ks = np.random.randint(1, 4, size=batch_size)
            X_cand = np.zeros((batch_size, DIM))
            for i in range(batch_size):
                k = ks[i]
                idx = np.random.choice(DIM, k, replace=False)
                vals = np.random.dirichlet(np.ones(k))
                X_cand[i, idx] = vals
                
            X_tensor = torch.tensor(X_cand, dtype=torch.float64).unsqueeze(1)
            with torch.no_grad():
                acq_vals = acquisition_fn(X_tensor)
                if acq_vals.ndim > 1:
                    acq_vals = acq_vals.squeeze(-1)
            
            max_val, max_idx = torch.max(acq_vals, dim=0)
            if max_val.item() > best_value:
                best_value = max_val.item()
                best_x = X_cand[max_idx.item()]
        
        return best_x, best_value

# =====================================================
# 5. Main BO Loop (Twin-World) - FIXED VERSION
# =====================================================
def run_bo_experiment(csv_path, n_init=100, n_iter=50, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"\n{'='*60}\nRunning BO Benchmark (Seed {seed}) - FIXED VERSION\n{'='*60}")
    
    # 1. Setup
    X_init, y_init_noisy, y_init_true = load_initial_dataset(csv_path, n_init, seed)
    problem_noisy, problem_noisefree, true_optimum = setup_coco_oracles()
    
    # 2. Tensors
    train_X = torch.tensor(X_init, dtype=torch.float64)
    train_Y = torch.tensor(y_init_noisy, dtype=torch.float64).unsqueeze(-1)
    
    # 3. Metrics
    log_regret_history = []
    best_found_history = []  # Track best observed value (Agent View)
    
    # Calculate initial regret
    if y_init_true is None:
         y_init_true = oracle_noisefree(X_init, problem_noisefree)
    best_initial_true = np.max(y_init_true) #假裝用最大的來找
    initial_log_regret = np.log10(max(best_initial_true - true_optimum, 1e-10)) #假裝最大的來找
    log_regret_history.append(initial_log_regret) 
    
    # Track initial best found (minimum value since COCO minimizes)
    best_found_history.append(train_Y.max().item())
    
    print(f"Initial Best True: {best_initial_true:.4f} | Log Regret: {initial_log_regret:.4f}")
    
    # Store log space statistics for consistent conversion
    log_space_stats = None
    
    # --- Loop ---
    gp = None # Placeholder
    for i in range(n_iter):
        # A. Fit GP with Log Space Modeling
        # ==========================================
        # 【关键修改】将最小化能量转换为最大化效用（负能量）
        # 对于 Rosenbrock 这种有"山谷"的函数，最大化效用 = 往山谷里走
        y_min = train_Y.min().item()
        y_max = train_Y.max().item()
        # 使用相对偏移：1% of the range 或至少 1e-6
        offset = max(1e-6, 0.01 * (y_max - y_min))
        
        # 【关键修改】加上负号！现在的含义：值越大 = 能量越低 = 越好 (Utility)
        train_Y_log = -torch.log(train_Y + offset)  # Utility = -log(Energy)
        
        # 【修复2】: 存储统计量用于后续转换（确保一致性）
        y_log_mean = train_Y_log.mean().item()
        y_log_std = train_Y_log.std().item()
        log_space_stats = {
            'mean': y_log_mean,
            'std': y_log_std,
            'offset': offset
        }
        
        # Standardize log space data (现在是 Utility 空间)
        train_Y_modeling = standardize(train_Y_log)
        
        # gp = SingleTaskGP(train_X, train_Y_modeling)


        covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=train_X.shape[-1],  # ARD：每一維一個 lengthscale（強烈建議）
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )

        gp = SingleTaskGP(
            train_X,
            train_Y_modeling,
            covar_module=covar_module,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # B. Optimize ACQ (LogEI) with Warm-Start
        # ==========================================
        # 【关键修改】因为我们取了负号，现在是 Maximize Utility 问题
        # LogEI 计算的是"超过 best_f 的改进"，对于最大化： 
        # - 我们想要找到比当前最大值更大的值（更高的效用 = 更低的能量）
        # - 所以 best_f 应该是当前最大值（在标准化后的效用空间）
        best_f_log_std = train_Y_modeling.max().item()  # ✅ 改回 max()，因为现在是最大化问题
        EI = LogExpectedImprovement(gp, best_f=best_f_log_std)
        new_x, _ = optimize_acquisition_simplex(EI, train_X=train_X, train_Y=train_Y_modeling, seed=seed+i)
        
        # C. Agent View (Lab)
        new_y_noisy = oracle_noisy(new_x, problem_noisy)
        
        # D. Update Model Data
        new_x_tensor = torch.tensor(new_x, dtype=torch.float64).unsqueeze(0)  # Shape: (1, 5)

        new_y_tensor = torch.tensor([[new_y_noisy]], dtype=torch.float64)  # Shape: (1, 1)


        # print('**'*50)
        # print('new_x_tensor = ',new_x_tensor)
        # print('new_x_tensor.sum() = ',new_x_tensor.sum())
        # print('**'*50)


        train_X = torch.cat([train_X, new_x_tensor])
        train_Y = torch.cat([train_Y, new_y_tensor])
        
        # E. God View (Inference Regret)
        # ==========================================
        # Ask model for best prediction (maximize utility = minimize energy)
        def neg_mean_acq_fn(x_tensor):
            """Acquisition function: returns posterior mean of utility (to maximize)"""

            posterior = gp.posterior(x_tensor)  # x_tensor shape: (B, 1, D)
            mean = posterior.mean.squeeze(-1)  # Shape: (B, 1) -> (B,)
            # GP 预测的是 Utility (-LogY)
            # 我们想找 Energy 最小的点 => 即 Utility 最大的点
            # optimize_acqf 默认是 Maximize，所以直接返回 mean 即可，不需要取负号！
            return mean  # ✅ 去掉负号，直接返回均值（最大化效用）
        
        # For inference regret, use warm-start with current training data
        x_hat, _ = optimize_acquisition_simplex(neg_mean_acq_fn, train_X=train_X, train_Y=train_Y_modeling, n_candidates=3000, seed=seed+i+1000)
        y_hat_true = oracle_noisefree(x_hat, problem_noisefree)

        print('000'*50)
        print('x_hat = ',x_hat)
        print('y_hat_true = ',y_hat_true)
        print('000'*50)
        
        regret = y_hat_true - true_optimum
        log_regret = np.log10(max(regret, 1e-10))
        log_regret_history.append(log_regret)
        
        # Track best found (minimum value since COCO minimizes)
        best_found_history.append(train_Y.min().item())


        
        print(f"Iter {i+1}/{n_iter}: Regret={regret:.6f} | LogRegret={log_regret:.4f}")

    # ==========================================
    # F. FINAL DIAGNOSTICS (For Parity Plot)
    # ==========================================
    print("\nRunning Final Model Diagnostics...")
    
    # 1. Generate Test Set
    n_test = 100
    X_test_simplex = []
    for _ in range(n_test):
        k = np.random.randint(1, 4)  # k ~ 2 to 5 for D=5
        idx = np.random.choice(3, k, replace=False)
        vals = np.random.dirichlet(np.ones(k))
        x = np.zeros(3); x[idx] = vals
        X_test_simplex.append(x)
    
    X_test = torch.tensor(np.array(X_test_simplex), dtype=torch.float64)
    
    # 2. Get Predictions (Agent View)
    # ==========================================
    gp.eval()
    with torch.no_grad():
        posterior = gp.posterior(X_test)
        pred_means_std = posterior.mean.squeeze()
        pred_vars_std = posterior.variance.squeeze()
        
        # 【修复4】: 使用存储的统计量进行转换（确保一致性）
        if log_space_stats is None:
            # Fallback: recalculate if somehow stats weren't stored
            # 注意：这里也要使用负号，与训练时保持一致
            offset = max(1e-6, 0.01 * (train_Y.max().item() - train_Y.min().item()))
            train_Y_log = -torch.log(train_Y + offset)
            y_log_mean = train_Y_log.mean().item()
            y_log_std = train_Y_log.std().item()
        else:
            y_log_mean = log_space_stats['mean']
            y_log_std = log_space_stats['std']
            offset = log_space_stats['offset']
        
        # 反标准化（得到的是 -log(y) 的均值和方差）
        pred_means_neg_log = pred_means_std * y_log_std + y_log_mean  # 这是 -log(Y) 的均值
        
        # 【关键修改】反转负号（变回 log(y)）
        pred_means_log = -pred_means_neg_log  # ✅ 把负号乘回去，得到 log(Y) 的均值
        
        # 方差不变（Var(-X) = Var(X)）
        pred_vars_log = pred_vars_std * (y_log_std ** 2)
        
        # 【修复5】: 数值稳定的转换回原始空间
        # 使用 clamp 防止 exp 溢出
        pred_vars_log_clamped = torch.clamp(pred_vars_log, max=10.0)  # exp(10) ≈ 22026, 防止溢出
        
        # 转换回原始空间（Log-Normal 分布）
        # E[Y] = exp(mu + 0.5*sigma^2)
        pred_means = torch.exp(pred_means_log + 0.5 * pred_vars_log_clamped) - offset
        
        # 标准差：Std[Y] = E[Y] * sqrt(exp(sigma^2) - 1)
        # 使用数值稳定的计算
        exp_var_minus_one = torch.exp(pred_vars_log_clamped) - 1.0
        exp_var_minus_one = torch.clamp(exp_var_minus_one, min=0.0)  # 确保非负
        pred_std_devs = torch.exp(pred_means_log) * torch.sqrt(exp_var_minus_one)
        
        # 检查是否有 NaN 或 Inf
        if torch.any(torch.isnan(pred_means)) or torch.any(torch.isinf(pred_means)):
            print("  ⚠️ Warning: Some predictions are NaN or Inf. Using median instead of mean.")
            # Fallback: 使用中位数（更稳定）
            pred_means = torch.exp(pred_means_log) - offset
            pred_std_devs = torch.zeros_like(pred_means)

    # 3. Get Truth (God View)
    true_values = oracle_noisefree(X_test_simplex, problem_noisefree)

    return {
        "regret_history": log_regret_history,
        "best_found_history": best_found_history,  # Best observed values (Agent View)
        "true_optimum": true_optimum,  # True optimum for plotting
        "test_preds": pred_means.numpy(),
        "test_stds": pred_std_devs.numpy(),
        "test_true": true_values,
        "final_noise_est": gp.likelihood.noise.item() if gp is not None else 0.0
    }

# =====================================================
# 6. Visualization (Professional Style with Smoothing)
# =====================================================
def get_monotonic_trace(log_regret_history):
    """
    Convert jagged trace to smooth "Best Found So Far" curve.
    Takes the minimum value seen up to each step (cumulative minimum).
    
    Args:
        log_regret_history: Array of log regret values (can be noisy/jumpy)
    
    Returns:
        Smooth monotonic trace showing best performance so far
    """
    return np.minimum.accumulate(log_regret_history)


def visualize_results(results):
    """
    Visualize BO results with professional smoothing.
    Plots both raw (instantaneous) and smooth (best found so far) traces.
    """
    # Unpack results
    raw_regret = np.array(results["regret_history"])
    best_found = results["best_found_history"]  # Best observed values (for comparison)
    true_optimum = results["true_optimum"]  # True optimum value
    y_pred = results["test_preds"]
    y_true = results["test_true"]
    y_std = results["test_stds"]
    
    # 1. SMOOTHING: Calculate "Best Found So Far" (Monotonic Regret)
    # This hides the exploration jumps and shows true convergence
    smooth_regret = get_monotonic_trace(raw_regret)
    
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1]) 

    # --- PLOT 1: Optimization Convergence (Log Inference Regret) ---
    ax1 = plt.subplot(gs[0])
    iterations = np.arange(len(raw_regret))
    
    # Plot Raw (faint) to show "Thinking/Exploration"
    ax1.plot(iterations, raw_regret, 'b--', alpha=0.3, linewidth=1, 
             label='Instantaneous (Exploration)(Inference Regret)')
    
    # Plot Smooth (bold) to show "Result" - Cumulative Minimum
    ax1.plot(iterations, smooth_regret, 'b-o', linewidth=2.5, markersize=6, 
             label='Best Found So Far (Monotonic) (simple regret)')
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Log Inference Regret", fontsize=12)
    ax1.set_title("Optimization Convergence (5D Mixture)(oracle_noisefree)", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Target Line (optional - can be adjusted based on problem)
    ax1.axhline(0, color='r', linestyle=':', linewidth=2, label='Target (1e-1)')
    ax1.legend()

    # --- PLOT 2: Parity Plot (Model Calibration) ---
    ax2 = plt.subplot(gs[1])
    
    # Filter out invalid values for plotting
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true) & (y_pred > 0) & (y_true > 0)
    y_pred_clean = y_pred[valid_mask]
    y_true_clean = y_true[valid_mask]
    y_std_clean = y_std[valid_mask] if len(y_std) > 0 else None
    
    if len(y_pred_clean) > 0:
        if y_std_clean is not None and len(y_std_clean) > 0:
            ax2.errorbar(y_true_clean, y_pred_clean, yerr=y_std_clean, fmt='o', color='purple', 
                         alpha=0.5, ecolor='gray', elinewidth=1, capsize=2, label='Predictions')
        else:
            ax2.scatter(y_true_clean, y_pred_clean, color='purple', alpha=0.5, label='Predictions')
        
        # Perfect Model Line
        min_val = min(y_true_clean.min(), y_pred_clean.min())
        max_val = max(y_true_clean.max(), y_pred_clean.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Model')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No valid predictions to plot', 
                transform=ax2.transAxes, ha='center', va='center')
    
    ax2.set_xlabel("True Value (God View)", fontsize=12)
    ax2.set_ylabel("Predicted Value (Agent View)", fontsize=12)
    ax2.set_title("Model Reliability ", fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the plot
    output_filename = "bo_results_fixed.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    CSV_FILE = 'D:/Users/TingYuLin/Desktop/py12/chemical_sparse_data_f108_3d_20260115_1.csv'
    try:
        results = run_bo_experiment(CSV_FILE, n_init=100, n_iter=500)
        visualize_results(results)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
