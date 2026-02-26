from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.means import Mean
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel

import torch
import itertools
import time
import psutil
import os
import logging
import json
from functools import wraps
from contextlib import redirect_stdout



# ============================================================
# Optimized Mean Function: IDXSFastScheffeMean 
# ============================================================
class IdxsFastScheffeMean(Mean):
    """
    High-performance Scheffé polynomial mean function.

    Mathematical form:

        m(z) =
            Σ β_i z_i
          + Σ β_ij z_i z_j
          + Σ β_ijk z_i z_j z_k
          + Σ β_ijkl z_i z_j z_k z_l

    depending on order.

    Design goals:
    - No torch.pow
    - No mask matrix
    - No large broadcast tensor
    - No Python inner loops in forward
    - Optimized for optimize_acqf gradient loops
    """

    def __init__(self, input_dim, order=2):

        """
        Parameters
        ----------
        input_dim : int
            Number of mixture components (dimension d)

        order : int
            Polynomial order (1 ~ 4 supported)
        """


        super().__init__()
        self.input_dim = input_dim
        self.order = order

        num_params = 0
        
        # --------------------------------------------------
        # 1st-order terms (linear effects)
        # --------------------------------------------------
        # z1, z2, ..., zd
        self.register_buffer("idx1", torch.arange(input_dim))
        num_params += input_dim


        # --------------------------------------------------
        # 2nd-order interaction terms
        # --------------------------------------------------
        # zi * zj for i < j
        if order >= 2:
            idx2_i, idx2_j = torch.triu_indices(input_dim, input_dim, offset=1)
            self.register_buffer("idx2_i", idx2_i)
            self.register_buffer("idx2_j", idx2_j)
            num_params += len(idx2_i)

        # --------------------------------------------------
        # 3rd-order interaction terms
        # --------------------------------------------------
        # zi * zj * zk
        if order >= 3:
            comb3 = torch.combinations(torch.arange(input_dim), r=3)
            self.register_buffer("idx3", comb3)
            num_params += comb3.shape[0]

        # --------------------------------------------------
        # 4th-order interaction terms
        # --------------------------------------------------
        # zi * zj * zk * zl

        if order >= 4:
            comb4 = torch.combinations(torch.arange(input_dim), r=4)
            self.register_buffer("idx4", comb4)
            num_params += comb4.shape[0]

        # --------------------------------------------------
        # Trainable coefficients β
        # --------------------------------------------------
        # Total parameters = Σ C(d, k)

        self.beta = torch.nn.Parameter(torch.zeros(num_params))



    def forward(self, X):
        """
        Compute Scheffé mean.

        Parameters
        ----------
        X : Tensor
            Shape (..., n, d)
            where d = input_dim

        Returns
        -------
        mean : Tensor
            Shape (..., n)
        """


        features = []

        # -------------------------
        # 1st order
        # -------------------------
        features.append(X)

        # -------------------------
        # 2nd order
        # -------------------------
        if self.order >= 2:
            F2 = X[..., self.idx2_i] * X[..., self.idx2_j]
            features.append(F2)

        # -------------------------
        # 3rd order
        # -------------------------
        if self.order >= 3:
            i, j, k = self.idx3.unbind(dim=1)
            F3 = X[..., i] * X[..., j] * X[..., k]
            features.append(F3)

        # -------------------------
        # 4th order
        # -------------------------
        if self.order >= 4:
            i, j, k, l = self.idx4.unbind(dim=1)
            F4 = X[..., i] * X[..., j] * X[..., k] * X[..., l]
            features.append(F4)

        # -------------------------
        # Concatenate all polynomial terms
        # -------------------------
        F = torch.cat(features, dim=-1)

        # Linear combination with β
        return torch.matmul(F, self.beta)


# ============================================================
# Monitoring Utilities
# ============================================================

logger = logging.getLogger("BO_Monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)



def monitor(runtime=True, memory=True, condition=False):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "debug", False):
                return func(self, *args, **kwargs)
            if not hasattr(self, 'performance_history'):
                self.performance_history = []
            
            start_time = time.perf_counter()
            process = psutil.Process(os.getpid())
            
            result = func(self, *args, **kwargs)
            
            elapsed = time.perf_counter() - start_time
            cpu_mem = process.memory_info().rss / 1024**2
            
            entry = {
                "iteration": getattr(self, "current_iter", "N/A"),
                "function": func.__name__,
                "runtime_sec": round(elapsed, 4),
                "cpu_memory_mb": round(cpu_mem, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            if torch.cuda.is_available():
                entry["gpu_alloc_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)
            
            self.performance_history.append(entry)
            logger.info(f"Finished {func.__name__} - {entry}")
            return result
        return wrapper
    return decorator



# ============================================================
# Scheffé Trend GP Emulator IDX ver.
# ============================================================
class IdxsScheffeTrendGPEmulator:
    """
    Universal Kriging model:

        f(x) = Scheffé polynomial trend + GP residual
    """

    def __init__(
        self,
        device        = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype         = torch.double,
        kernel        = 'rbf',
        use_ard       = False,
        matern_nu     = 2.5,
        scheffe_order = 2,
        debug         = False,
    ):
        self.device = device
        self.dtype = dtype
        self.kernel = kernel
        self.use_ard = use_ard
        self.matern_nu = matern_nu
        self.scheffe_order = scheffe_order
        self.debug = debug
        self.performance_history = []
        self.current_iter = 0
        self.model = None

    def _build_kernel(self, input_dim):
        ard_dims = input_dim if self.use_ard else None
        if self.kernel == "matern":
            base_kernel = MaternKernel(nu=self.matern_nu, ard_num_dims=ard_dims)
        else:
            base_kernel = RBFKernel(ard_num_dims=ard_dims)
        return ScaleKernel(base_kernel)

    @monitor(runtime=True, memory=True, condition=True)
    def fit(self, train_x: torch.Tensor, train_obj: torch.Tensor):
        """
        Fit multi-objective GP model.
        """
        train_x = train_x.to(device=self.device, dtype=self.dtype)
        train_obj = train_obj.to(device=self.device, dtype=self.dtype)

        if train_obj.ndim == 1:
            train_obj = train_obj.unsqueeze(-1)

        input_dim = train_x.shape[-1]
        num_objectives = train_obj.shape[-1]
        models = []

        for i in range(num_objectives):
            covar_module = self._build_kernel(input_dim)

            mean_module = IdxsFastScheffeMean(
                input_dim=input_dim,
                order=self.scheffe_order,
            ).to(device=self.device, dtype=self.dtype) 

            train_y = train_obj[..., i:i + 1]

            gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                mean_module=mean_module,
                outcome_transform=Standardize(m=1),
                covar_module=covar_module,
            )
            models.append(gp)

        self.model = ModelListGP(*models).to(device=self.device, dtype=self.dtype)
        self.mll = SumMarginalLogLikelihood(self.model.likelihood, self.model)

        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                fit_gpytorch_mll(self.mll)

        return self.model

    def save_performance_to_json(self, folder_path, filename="performance_report.json"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_path = os.path.join(folder_path, filename)
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save JSON: {str(e)}")

    @torch.no_grad()
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        posterior = self.model.posterior(X)
        return posterior.mean, posterior.variance