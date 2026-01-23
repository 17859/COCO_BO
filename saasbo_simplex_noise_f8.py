import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models import SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.optim import optimize_acqf

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel

from torch.distributions import Dirichlet

# ======================================================
# 0. Environment
# ======================================================
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
device = "cpu"

# ======================================================
# 1. Noisefree Rosenbrock f8 (God view)
# ======================================================
def objective_noisefree(X):
    """
    X : (N, D) in [0,1]
    return utility = -energy (N,1)
    """
    Z = 10.0 * X - 5.0
    energy = torch.sum(
        100.0 * (Z[:, 1:] - Z[:, :-1]**2)**2 +
        (Z[:, :-1] - 1.0)**2,
        dim=1
    )
    return -energy.unsqueeze(-1)

# ======================================================
# 2. Noisy Rosenbrock (Agent view, reproducible)
# ======================================================
def make_noisy_rosenbrock_f8(noise_std=1, seed=42, device="cpu"):
    rng = torch.Generator(device=device)
    # rng.manual_seed(seed)

    def objective(X):
        Z = 10.0 * X - 5.0
        energy = torch.sum(
            100.0 * (Z[:, 1:] - Z[:, :-1]**2)**2 +
            (Z[:, :-1] - 1.0)**2,
            dim=1
        )
        noise = noise_std * torch.randn(
            energy.shape,
            generator=rng,
            device=energy.device
        )
        return -(energy + noise).unsqueeze(-1)

    return objective

# ======================================================
# 3. Simplex + k-sparse utilities
# ======================================================
def sparsify_x(x, k):
    values, indices = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, indices, values)
    return out / out.sum(dim=-1, keepdim=True)

def sample_sparse_simplex(n, d, k):
    samples = []
    for _ in range(n):
        idx = torch.randperm(d)[:k]
        val = Dirichlet(torch.ones(k)).sample()
        x = torch.zeros(d)
        x[idx] = val
        samples.append(x)
    return torch.stack(samples)

# ======================================================
# 4. Problem setup
# ======================================================
dim = 10
k_sparse = 5

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])

equality_constraints = [
    (torch.arange(dim), torch.ones(dim), 1.0)
]

N_INITIAL = 15
N_ITER = 50

# ⚠️ 建議你之後用 SLSQP 算真正的 simplex optimum
GLOBAL_MAXIMUM = -406267.8

# ======================================================
# 5. Initialize dataset (Twin-world)
# ======================================================
noisy_objective = make_noisy_rosenbrock_f8(
    noise_std=1,
    seed=42,
    device=device
)

train_X = sample_sparse_simplex(N_INITIAL, dim, k_sparse)
train_Y = noisy_objective(train_X)          # 🔴 noisy only

# ======================================================
# 6. Regret containers
# ======================================================
simple_regrets = []
# inference_regrets = []
rmses = []
cumulative_regrets = [0.0]

# ======================================================
# 7. Plot setup
# ======================================================
plt.ion()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axs

# ======================================================
# 8. BO Loop
# ======================================================
print("開始 SAASBO (Simplex + COCO f8 Hellish) 優化循環...")

for i in range(N_ITER):
    # --------------------------------------------------
    # A. Fit GP (on noisy data)
    # --------------------------------------------------
    model = SaasFullyBayesianSingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
    fit_fully_bayesian_model_nuts(model, warmup_steps=128, num_samples=128, thinning=8)


    # # --------------------------------------------------
    # # B. Inference regret (God view)
    # # --------------------------------------------------
    # pm = PosteriorMean(model)
    # inferred_x, _ = optimize_acqf(
    #     pm,
    #     bounds=bounds,
    #     q=1,
    #     num_restarts=5,
    #     raw_samples=20,
    #     equality_constraints=equality_constraints
    # )
    # inferred_x = sparsify_x(inferred_x, k_sparse)

    # inferred_true = objective_noisefree(inferred_x).item()
    # inf_reg = max(GLOBAL_MAXIMUM - inferred_true, 1e-6)
    # inference_regrets.append(inf_reg)


    # --------------------------------------------------
    # B. RMSE (God view, model accuracy)
    # --------------------------------------------------
    with torch.no_grad():
        # PosteriorMean expects shape: (batch, q=1, d)
        X_eval = train_X.unsqueeze(1)        # [N, 1, d]

        posterior_mean = PosteriorMean(model)(X_eval).squeeze(-1)
        true_y = objective_noisefree(train_X).squeeze(-1)

        rmse = torch.sqrt(
            torch.mean((posterior_mean - true_y) ** 2)
        ).item()

    rmses.append(rmse)


    # --------------------------------------------------
    # C. Simple regret (best observed true value)
    # --------------------------------------------------
    best_true = objective_noisefree(train_X).max().item()
    simp_reg = max(GLOBAL_MAXIMUM - best_true, 1e-6)
    simple_regrets.append(simp_reg)

    # --------------------------------------------------
    # D. Visualization (FIXED AXES)
    # --------------------------------------------------
    iters = list(range(1, len(simple_regrets) + 1))

    # --- D1. Isotropic relevance ---
    ls = model.covar_module.base_kernel.lengthscale.median(dim=0).values.squeeze()
    relevance = 1.0 / ls

    ax1.clear()
    ax1.bar(range(dim), relevance.detach().cpu().numpy(), color='gray', alpha=0.5)
    ax1.set_title(f"Iteration {i+1}: Relevance (1/Lengthscale)| saasbo")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Relevance")
    ax1.set_ylim(0, 10.0)                # 🔒 固定 y 軸
    ax1.set_xlim(-0.5, dim - 0.5)       # 🔒 固定 x 軸

    # --- D2. Simple Regret ---
    ax2.clear()
    ax2.plot(
        iters,
        simple_regrets,
        linestyle='-',
        marker='o',
        markersize=4,
        label='Simple Regret'
    )
    ax2.set_yscale("log")
    ax2.set_title("Simple Regret (Best Observed)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Regret")
    ax2.set_xlim(1, N_ITER)             # 🔒 固定 x 軸
    ax2.set_ylim(1e2, 1e6)              # 🔒 固定 y 軸
    ax2.legend()

    # # --- D3. Inference Regret ---
    # ax3.clear()
    # ax3.plot(
    #     iters,
    #     inference_regrets,
    #     linestyle='-',
    #     marker='s',
    #     markersize=4,
    #     color='orange',
    #     label='Inference Regret'
    # )
    # ax3.set_yscale("log")
    # ax3.set_title("Inference Regret (Model Belief)")
    # ax3.set_xlabel("Iteration")
    # ax3.set_ylabel("Regret")
    # ax3.set_xlim(1, N_ITER)             # 🔒 固定 x 軸
    # ax3.set_ylim(1e2, 1e6)              # 🔒 固定 y 軸
    # ax3.legend()

    # --- D3. RMSE (Model Accuracy) ---
    ax3.clear()
    ax3.plot(
        iters,
        rmses,
        linestyle='-',
        marker='s',
        markersize=4,
        color='purple',
        label='RMSE'
    )
    ax3.set_yscale("log")
    ax3.set_title("RMSE (Posterior Mean vs True f)")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMSE")
    ax3.set_xlim(1, N_ITER)
    ax3.legend()


    # --- D4. Cumulative Regret ---
    ax4.clear()
    ax4.plot(
        range(len(cumulative_regrets)),
        cumulative_regrets,
        linestyle='-',
        marker='^',
        markersize=4,
        color='green',
        label='Cumulative Regret'
    )
    ax4.set_title("Cumulative Regret")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Regret")
    ax4.set_xlim(0, N_ITER)             # 🔒 固定 x 軸
    ax4.set_ylim(0, 4e6)                # 🔒 固定 y 軸
    ax4.legend()

    plt.tight_layout()
    plt.pause(0.3)



    # --------------------------------------------------
    # E. Acquisition (LogEI, noisy world)
    # --------------------------------------------------
    acq = LogExpectedImprovement(model, train_Y.max())
    new_row, _ = optimize_acqf(
        acq,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
        equality_constraints=equality_constraints
    )



    new_x = sparsify_x(new_row, k_sparse)

    # noisy observation → training
    new_y_noisy = noisy_objective(new_x)
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, new_y_noisy])

    # true value → regret
    new_y_true = objective_noisefree(new_x).item()
    inst_reg = max(GLOBAL_MAXIMUM - new_y_true, 0.0)
    cumulative_regrets.append(cumulative_regrets[-1] + inst_reg)

    print(
        f"Iter {i+1:02d} | "
        f"BO Recommended x (Raw) |{new_row} "
        f"BO Recommended x  |{new_x} "
        f"Best(noisy)={train_Y.max().item():.2f} | "
        f"SimpReg={simp_reg:.2e} | "
        f"RMSE={rmse:.2e}"
        # f"InfReg={inf_reg:.2e}"
    )


plt.ioff()
plt.show()
