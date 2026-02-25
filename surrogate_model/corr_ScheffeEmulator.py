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



class ScheffeMean(Mean):
    """
    優化後的 Scheffé 混合多項式均值函數。
    利用張量廣播與掩碼矩陣消除 Python 迴圈，提升 GPU 運算效率。
    """
    def __init__(self, input_dim, order=1):
        super().__init__()
        self.input_dim = input_dim
        self.order = order

        if order < 1:
            raise ValueError("ScheffeMean requires order >= 1")

        # 1. 產生所有組合索引
        basis_indices = []
        for k in range(1, order + 1):
            basis_indices.extend(list(itertools.combinations(range(input_dim), k)))
        
        self.p = len(basis_indices)

        # 2. 建立掩碼矩陣 (Mask Matrix)
        # mask shape: (p, d), 代表 p 個項中哪些維度需要被相乘
        mask = torch.zeros(self.p, input_dim)
        for i, idx_tuple in enumerate(basis_indices):
            mask[i, list(idx_tuple)] = 1
        
        # 註冊為 buffer，會自動隨模型搬移至 GPU/CPU 並處理 dtype
        self.register_buffer("mask", mask)

        # 3. 學習參數 β
        self.register_parameter(
            name="beta",
            parameter=torch.nn.Parameter(torch.zeros(self.p)),
        )

    def forward(self, X):
        """
        X shape: (..., n, d)
        Returns: (..., n)
        """
        # 增加維度以便廣播: X -> (..., n, 1, d)
        X_expanded = X.unsqueeze(-2) 

        # 利用 X^1 = X, X^0 = 1 的特性計算各項乘積
        # F_elements: (..., n, p, d)
        F_elements = torch.pow(X_expanded, self.mask)
        
        # 對最後一個維度求積，得到每一項的基函數值 (..., n, p)
        F = torch.prod(F_elements, dim=-1)

        # 與權重 beta 做線性組合 (..., n)
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




class CorrelationScheffeTrendGPEmulator:
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
        mean_module = ScheffeMean(
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
