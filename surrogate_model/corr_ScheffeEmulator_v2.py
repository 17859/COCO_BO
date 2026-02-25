import torch
import itertools
import torch
import time
import psutil
import os
from functools import wraps
from contextlib import redirect_stdout
import logging
import json


from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from gpytorch.means import Mean
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood



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
# Monitoring Utilities & Decorator
# ============================================================
logger = logging.getLogger("BO_Monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)




# ============================================================
# Decorator
# ============================================================
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




class CorrelationIdxsScheffeTrendGPEmulator:
    """
    Correlated Multi-Objective Gaussian Process Emulator
    ====================================================

    This class builds a multi-objective GP surrogate model:

        y_k(z) = m_k(z) + g_k(z)

    where:
        m_k(z) = Scheffé polynomial trend
        g_k(z) ~ GP(0, K_θ)

    The correlation between objectives is modeled via
    KroneckerMultiTaskGP.

    Parameters
    ----------
    device : torch.device
        Device for computation (CPU or GPU).

    dtype : torch.dtype
        Floating point precision.

    scheffe_order : int
        Polynomial order of Scheffé mean.

    debug : bool
        Enable runtime/memory diagnostics.
    """
    def __init__(
        self,
        device        = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype         = torch.double,
        scheffe_order = 2,
        debug         = False,   # Debug switch

    ):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.mll = None

        self.scheffe_order = scheffe_order
        self.debug = debug

        self.performance_history = []
        self.current_iter = 0
        self.model = None


    @monitor(runtime=True, memory=True, condition=False)
    def fit(self, train_x: torch.Tensor, train_obj: torch.Tensor):
        """
        Fit the multi-task GP model.

        Parameters
        ----------
        train_x : torch.Tensor
            Shape (N, d)

        train_obj : torch.Tensor
            Shape (N, k)
            k = number of objectives
        """

        # If single objective (N,), convert to (N, 1)
        if train_obj.ndim == 1:
            train_obj = train_obj.unsqueeze(-1)

        input_dim = train_x.shape[-1]
        num_objectives = train_obj.shape[-1]


        # Define Scheffé polynomial mean
        mean_module = IdxsFastScheffeMean(
            input_dim=input_dim,
            order=self.scheffe_order,
        )  


        # Construct correlated multi-task GP
        self.model = KroneckerMultiTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            outcome_transform=Standardize(m=num_objectives),
            mean_module = mean_module
        )

        # Exact marginal log likelihood
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                fit_gpytorch_mll(self.mll)

        return self.model
    
    def save_performance_to_json(self, folder_path, filename="performance_report.json"):
        """
        將監控紀錄儲存至指定資料夾路徑。
        """
        # 1. 確保資料夾存在，若不存在則自動建立
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            logger.info(f"Created directory: {folder_path}")

        # 2. 合併完整路徑
        full_path = os.path.join(folder_path, filename)

        # 3. 寫入 JSON
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved performance metrics to: {full_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {str(e)}")

            
    
    @torch.no_grad()
    def predict(self, X ):
        """
        Posterior prediction.

        Parameters
        ----------
        X : torch.Tensor or array-like
            Shape (n, d)

        Returns
        -------
        mean : torch.Tensor
            Posterior predictive mean, shape (n, k)

        var : torch.Tensor
            Posterior predictive variance, shape (n, k)
        """

        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        posterior = self.model.posterior(X)
        mean = posterior.mean    # (n, k)
        var = posterior.variance # (n, k)

        return mean, var
