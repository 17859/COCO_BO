from botorch.models import KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize


import logging
import json
import torch
import time
import psutil
import os
from functools import wraps
from contextlib import redirect_stdout
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize


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



class CorrelationBaselineGPEmulator:
    """
    Correlation Baseline Multi-Objective GP Emulator
    =================================================

    This implementation uses KroneckerMultiTaskGP to jointly model
    multiple objectives while learning correlations between tasks.

    Mathematical Form:

        Y = f(X) + ε
        f ~ MultiTaskGP(m, K_x ⊗ K_t)

    where:

        K_x : covariance function over input space
        K_t : task covariance matrix modeling inter-task correlation

    Note:
        KroneckerMultiTaskGP requires all tasks to share
        the same training inputs (fully observed design).
    """


    def __init__(
        self,
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype   = torch.double,
        debug   = False,
    ):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.mll = None
        self.debug = debug
        self.performance_history = []
        self.current_iter = 0



    @monitor(runtime=True, memory=True, condition=True)
    def fit(self, train_x: torch.Tensor, train_obj: torch.Tensor):
        """
        Fit a Kronecker Multi-Task GP model.

        Important:
            KroneckerMultiTaskGP assumes that all tasks are observed
            at the same set of training input locations.
        """

        # Convert single-objective case (N,) to (N, 1)
        if train_obj.ndim == 1:
            train_obj = train_obj.unsqueeze(-1)

        num_tasks = train_obj.shape[-1]

        # Construct the Kronecker Multi-Task GP
        # The model automatically learns the task covariance matrix
        self.model = KroneckerMultiTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            outcome_transform=Standardize(m=num_tasks),
        )

        # Multi-task GP uses ExactMarginalLogLikelihood
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Train model while suppressing optimization output
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
    def predict(self, X: torch.Tensor):
        """
        Perform prediction using the trained model.

        Returns:
            posterior.mean      : shape (n, num_tasks)
            posterior.variance  : shape (n, num_tasks)
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        X = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        
        # Switch to evaluation mode
        self.model.eval()
        self.model.likelihood.eval()
        
        posterior = self.model.posterior(X)
        
        # mean shape: (n, num_tasks)
        # variance shape: (n, num_tasks)
        return posterior.mean, posterior.variance