import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
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
# 1. Objective (God view)
# ======================================================
def objective_noisefree(X):
    Z = 10.0 * X - 5.0
    energy = torch.sum(
        100.0 * (Z[:, 1:] - Z[:, :-1] ** 2) ** 2 +
        (Z[:, :-1] - 1.0) ** 2,
        dim=1
    )
    return -energy.unsqueeze(-1)

# def make_noisy_objective(noise_std=1.0):
#     def f(X):
#         return objective_noisefree(X) + noise_std * torch.randn(X.size(0), 1)
#     return f


def make_noisy_objective(noise_std=1.0):
    def f(X):
        return objective_noisefree(X) 
    return f

# ======================================================
# 2. Simplex utils
# ======================================================
def sparsify_x(x, k):
    v, idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, idx, v)
    return out / out.sum(dim=-1, keepdim=True)

def sample_sparse_simplex(n, d, k):
    xs = []
    for _ in range(n):
        idx = torch.randperm(d)[:k]
        v = Dirichlet(torch.ones(k)).sample()
        x = torch.zeros(d)
        x[idx] = v
        xs.append(x)
    return torch.stack(xs)

# ======================================================
# 3. Setup
# ======================================================
dim = 10
k_sparse = 5
N_INIT = 15
N_ITER = 50
GLOBAL_MAX =  -406267.8

bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
eq_constraints = [(torch.arange(dim), torch.ones(dim), 1.0)]

noisy_f = make_noisy_objective()

# shared initial data
X0 = sample_sparse_simplex(N_INIT, dim, k_sparse)
Y0 = noisy_f(X0)

# ======================================================
# 3.5 Fixed evaluation set (OUT-OF-SAMPLE)
# ======================================================
N_EVAL = 10        # 可自行調大，例如 500 / 1000
X_eval = sample_sparse_simplex(N_EVAL, dim, k_sparse)

with torch.no_grad():
    Y_eval = objective_noisefree(X_eval).squeeze()   # God view, fixed


# ======================================================
# 4. Containers
# ======================================================
methods = ["Isotropic", "ARD", "SAASBO"]
colors = {"Isotropic": "tab:blue", "ARD": "tab:red", "SAASBO": "tab:green"}

simple_regret = {m: [] for m in methods}
rmse = {m: [] for m in methods}
cum_regret = {m: [0.0] for m in methods}
relevance = {m: None for m in methods}

datasets = {
    m: {"X": X0.clone(), "Y": Y0.clone()}
    for m in methods
}

# ======================================================
# 5. Plot
# ======================================================
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
(ax1, ax2), (ax3, ax4) = axs

# ======================================================
# 6. BO Loop
# ======================================================
for it in range(N_ITER):

    print(f"\n================ Iteration {it+1:02d} ================")

    for m in methods:
        X = datasets[m]["X"]
        Y = datasets[m]["Y"]

        # ---------- model ----------
        if m == "Isotropic":
            model = SingleTaskGP(
                X, Y,
                outcome_transform=Standardize(1),
                covar_module=ScaleKernel(MaternKernel(nu=2.5))
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            ls = model.covar_module.base_kernel.lengthscale.item()
            relevance[m] = torch.ones(dim) / ls

        elif m == "ARD":
            model = SingleTaskGP(
                X, Y,
                outcome_transform=Standardize(1),
                covar_module=ScaleKernel(
                    MaternKernel(nu=2.5, ard_num_dims=dim)
                )
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            relevance[m] = 1.0 / model.covar_module.base_kernel.lengthscale.view(-1)

        else:  # SAASBO
            model = SaasFullyBayesianSingleTaskGP(X, Y, outcome_transform=Standardize(1))
            fit_fully_bayesian_model_nuts(model, 128, 128)
            relevance[m] = 1.0 / model.covar_module.base_kernel.lengthscale.median(0).values


        # # ---------- metrics ----------
        # with torch.no_grad():
        #     pm = PosteriorMean(model)(X.unsqueeze(1)).squeeze()
        #     true = objective_noisefree(X).squeeze()
        #     curr_rmse = torch.sqrt(((pm - true) ** 2).mean()).item()
        #     rmse[m].append(curr_rmse)

        with torch.no_grad():
            pm_eval = PosteriorMean(model)(X_eval.unsqueeze(1)).squeeze()
            curr_rmse = torch.sqrt(
                torch.mean((pm_eval - Y_eval) ** 2)
            ).item()
            rmse[m].append(curr_rmse)


        with torch.no_grad():
            best_true = objective_noisefree(X).max().item()

        curr_simple_reg = max(GLOBAL_MAX - best_true, 1e-6)
        simple_regret[m].append(curr_simple_reg)

        # ---------- BO step ----------
        acq = LogExpectedImprovement(model, Y.max())
        x_raw, _ = optimize_acqf(
            acq, 
            bounds, 
            q=1,
            num_restarts=5, 
            raw_samples=20,
            equality_constraints=eq_constraints
        )
        x_new = sparsify_x(x_raw, k_sparse)
        y_new = noisy_f(x_new)

        datasets[m]["X"] = torch.cat([X, x_new])
        datasets[m]["Y"] = torch.cat([Y, y_new])

        true_new = objective_noisefree(x_new).item()
        curr_cum_reg = cum_regret[m][-1] + max(GLOBAL_MAX - true_new, 0.0)
        cum_regret[m].append(curr_cum_reg)

        best_noisy = datasets[m]["Y"].max().item()

        # ---------- PRINT ----------
        print(
            f"Iter {it+1:02d} | Method={m:<8} | "
            f"Best(noisy)={best_noisy: .2f} | "
            f"SimpleReg={curr_simple_reg: .2e} | "
            f"CumReg={curr_cum_reg: .2e} | "
            f"RMSE={curr_rmse: .2e}\n"
            f"BO Recommended x (Raw): {x_raw.detach().cpu().numpy()}\n"
            f"BO Recommended x     : {x_new.detach().cpu().numpy()}"
        )

    # ==================================================
    # Visualization (FIXED AXES)
    # ==================================================

    # --- D1. Isotropic relevance ---
    ax1.clear()
    for m in methods:
        x_pos = (
            torch.arange(dim).detach().cpu().numpy()
            + 0.25 * methods.index(m)
        )

        rel = relevance[m]
        if torch.is_tensor(rel):
            rel = rel.detach().cpu().numpy()
        rel = rel.reshape(-1)

        ax1.bar(x_pos, rel, width=0.25, label=m)

    ax1.set_title("Comparison: Final Relevance")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("1 / Lengthscale")
    ax1.set_ylim(0, 10.0)                # 🔒 固定 y 軸
    ax1.set_xlim(-0.5, dim - 0.5)       # 🔒 固定 x 軸
    ax1.legend()


    # --- D2. Simple Regret ---
    ax2.clear()
    for m in methods:
        ax2.plot(
            simple_regret[m],
            label=m,
            color=colors[m],
            marker='o',
            markersize=4,
            linewidth=1.5
        )

    ax2.set_yscale("log")
    ax2.set_title("Simple Regret (Best Observed)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Regret")
    ax2.set_xlim(1, N_ITER)             # 🔒 固定 x 軸
    ax2.set_ylim(1e2, 1e6)              # 🔒 固定 y 軸
    ax2.legend()


    # --- D3. RMSE (Model Accuracy) ---
    ax3.clear()
    for m in methods:
        ax3.plot(
            rmse[m],
            label=m,
            color=colors[m],
            marker='s',        # 用方形點，和 regret 區分
            markersize=4,
            linewidth=1.5
        )

    ax3.set_yscale("log")
    ax3.set_title("Surrogate Model RMSE")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("RMSE")
    ax3.set_xlim(1, N_ITER)
    ax3.set_ylim(1e2, 1e6)             # 🔒 固定 y 軸
    ax3.legend()



    # --- D4. Cumulative Regret ---
    ax4.clear()
    for m in methods:
        ax4.plot(
            cum_regret[m],
            label=m,
            color=colors[m],
            marker='^',        # 三角形，視覺上很好分
            markersize=4,
            linewidth=1.5
        )

    ax4.set_title("Cumulative Regret")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Regret")
    ax4.set_xlim(0, N_ITER)             # 🔒 固定 x 軸
    ax4.set_ylim(0, 4e6)                # 🔒 固定 y 軸
    ax4.legend()

    plt.tight_layout()
    plt.pause(0.1)




plt.show()
