from botorch.models import SingleTaskGP, ModelListGP
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
from gpytorch.mlls import SumMarginalLogLikelihood
from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel,
)


# =========================================================
# Monitoring Utilities
# ============================================================

def _log_memory():
    """Log CPU & GPU memory usage."""

    # CPU
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / 1024**2
    print(f"[CPU] Memory: {cpu_mem:.2f} MB")

    # GPU
    if torch.cuda.is_available():
        gpu_alloc = torch.cuda.memory_allocated() / 1024**2
        gpu_res = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU] Alloc: {gpu_alloc:.2f} MB | Reserved: {gpu_res:.2f} MB")


def _log_condition_number(model_list_gp):
    """Log condition number of kernel matrix for each objective."""

    for i, model in enumerate(model_list_gp.models):

        train_x = model.train_inputs[0]

        with torch.no_grad():
            K = model.covar_module(train_x).evaluate().cpu()

            eigvals = torch.linalg.eigvalsh(K)

            cond = eigvals.max() / eigvals.min()

        print(f"[Objective {i}] Condition #: {cond.item():.3e}")

        if cond > 1e10:
            print("⚠ WARNING: Kernel matrix is ill-conditioned!")


# --- Logging 配置 ---
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

            # 確保類別中有存儲紀錄的容器
            if not hasattr(self, 'performance_history'):
                self.performance_history = []

            # 紀錄開始資訊
            start_time = time.perf_counter()
            process = psutil.Process(os.getpid())
            
            # 執行函數
            result = func(self, *args, **kwargs)
            
            # 計算指標
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            cpu_mem = process.memory_info().rss / 1024**2
            
            entry = {
                "iteration": getattr(self, "current_iter", "N/A"), # 追蹤當前疊代次數
                "function": func.__name__,
                "runtime_sec": round(elapsed, 4),
                "cpu_memory_mb": round(cpu_mem, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            if torch.cuda.is_available():
                entry["gpu_alloc_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 2)

            # 如果需要紀錄條件數 (僅針對 fit 後的模型)
            if condition and getattr(self, "model", None) is not None:
                cond_list = []
                for i, m in enumerate(self.model.models):
                    with torch.no_grad():
                        K = m.covar_module(m.train_inputs[0]).evaluate()
                        eigvals = torch.linalg.eigvalsh(K)
                        cond = (eigvals.max() / eigvals.min()).item()
                        cond_list.append(cond)
                entry["condition_numbers"] = cond_list

            # 儲存到物件歷史紀錄
            self.performance_history.append(entry)
            
            # 同時印出日誌供參考
            logger.info(f"Finished {func.__name__} - {entry}")
            
            return result
        return wrapper
    return decorator




class BaselineGPEmulator:
    """
    Baseline Multi-Objective GP Emulator
    =====================================

    This class implements a multi-objective Gaussian Process (GP)
    surrogate model using independent SingleTaskGP models.

    Mathematical Form:

        y_k(z) = m_k(z) + g_k(z)

        m_k(z) : constant mean (default in SingleTaskGP)
        g_k(z) ~ GP(0, K_{θ_k})

    Each objective is modeled with an independent GP.

    Example:
        >>> import torch
        >>> from baseline_gp_emulator import BaselineGPEmulator
        >>>
        >>> # Generate training data
        >>> train_X = torch.rand(20, 3, dtype=torch.double)
        >>> train_Y = torch.rand(20, 2, dtype=torch.double)
        >>>
        >>> # Create emulator
        >>>
        >>> # method1 : RBF is defult
        >>> emulator = BaselineGPEmulator()  #defult = RBF
        >>>
        >>> # method2 : matern
        >>> emulator = BaselineGPEmulator(
        ...     kernel="matern"
        ... )
        >>>
        >>> # Fit model
        >>> model = emulator.fit(train_X, train_Y)
        >>>
        >>> # Make predictions
        >>> test_X = torch.rand(5, 3, dtype=torch.double)
        >>> mean, var = emulator.predict(test_X)
        >>>
        >>> mean.shape
        torch.Size([5, 2])
        >>> var.shape
        torch.Size([5, 2])
    """

    def __init__(
        self,
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype   = torch.double,
        kernel    = "rbf",      # "rbf" or "matern"
        use_ard   = False,       # Whether to use Automatic Relevance Determination
        matern_nu = 2.5,         # Smoothness parameter for Matern kernel
        debug=False,   # Debug switch
        
    
    ):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.mll = None

        self.kernel = kernel
        self.use_ard = use_ard
        self.matern_nu = matern_nu
        self.debug = debug
        self.performance_history = []
        self.current_iter = 0


    def _build_kernel(self, input_dim):

        """
        Construct covariance kernel based on configuration.

        If ARD is enabled, each input dimension has its own
        independent lengthscale parameter.
        """


        ard_dims = input_dim if self.use_ard else None

        if self.kernel == "matern":
            base_kernel = MaternKernel(
                nu=self.matern_nu,
                ard_num_dims=ard_dims,
            )
        else:  # default: RBF
            base_kernel = RBFKernel(
                ard_num_dims=ard_dims,
            )

        # ScaleKernel allows the model to learn an output scale parameter
        return ScaleKernel(base_kernel)


    @monitor(runtime=True, memory=True, condition=True)
    def fit(self, train_x: torch.Tensor, train_obj: torch.Tensor):
        """
        Fit the multi-objective GP surrogate model.

        Parameters
        ----------
        train_x : torch.Tensor
            Training inputs of shape (N, d)

        train_obj : torch.Tensor
            Training objectives of shape (N, k)

        Returns
        -------
        ModelListGP
            A ModelListGP containing independent SingleTaskGP models
            for each objective.
        """

        # If single objective (N,), convert to (N, 1)
        if train_obj.ndim == 1:
            train_obj = train_obj.unsqueeze(-1)

        input_dim = train_x.shape[-1]
        num_objectives = train_obj.shape[-1]

        models = []
        
        # Build one independent GP per objective
        for i in range(num_objectives):
            covar_module = self._build_kernel(input_dim)

            train_y = train_obj[..., i:i + 1]  # (N, 1)

            gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                outcome_transform=Standardize(m=1),
                covar_module=covar_module,
            )
            models.append(gp)

        # Combine independent GPs into ModelListGP
        self.model = ModelListGP(*models)

        # Define marginal log likelihood for multi-model case
        self.mll = SumMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                fit_gpytorch_mll(self.mll)


        # Maximize marginal log likelihood
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
        Predict using the trained GP surrogate.

        Parameters
        ----------
        X : torch.Tensor or array-like
            Input locations of shape (n, d)

        Returns
        -------
        mean : torch.Tensor
            Posterior predictive mean of shape (n, k)

        var : torch.Tensor
            Posterior predictive variance of shape (n, k)
        """

        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        posterior = self.model.posterior(X)
        mean = posterior.mean    # (n, k)
        var = posterior.variance # (n, k)

        return mean, var




