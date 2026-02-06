import sys
import os
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from botorch.models.transforms.outcome import Standardize
from botorch.models import MultiTaskGP, KroneckerMultiTaskGP
from botorch.models import SingleTaskGP, ModelListGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood



from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.transforms import unnormalize, standardize

from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.utils import get_acquisition_function
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.monte_carlo import qExpectedImprovement


from botorch.acquisition.multi_objective.logei import (
    qLogExpectedHypervolumeImprovement,
    qLogNoisyExpectedHypervolumeImprovement
)

from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from pprint import pprint
from typing import Optional
from contextlib import redirect_stdout
import joblib

import warnings
warnings.filterwarnings("ignore")



# 設定設備與型別
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.set_default_dtype(dtype)
print('==='*5)
print('device = ',device)
print('==='*5)


# 計算超體積
def get_current_hv(train_Y, ref_point):
    # 1. 取得 Pareto Front
    pareto_y = train_Y[is_non_dominated(train_Y)]
    
    # 2. 初始化計算器 (注意大小寫，通常是 Hypervolume)
    hv_obj = Hypervolume(ref_point=ref_point)
    
    # 3. 計算並回傳
    return hv_obj.compute(pareto_y)


def run_bo_experiment(seed, initial_x, initial_y, n_iter, oracle, ref_point, x_name, y_name):
    print(f"\n>>> Starting Experiment with Seed: {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_x = initial_x.clone() #[0,100]
    train_obj = initial_y.clone()
    D = train_x.shape[-1]
    current_hvs = []
    
    t = tqdm(range(n_iter), desc=f"Seed {seed}", ncols=100)
    for i in t:
        std_train_x = (train_x / 100.0).to(device=device, dtype=dtype)


        # 1. 定義與擬合模型
        model = KroneckerMultiTaskGP(
            train_X=std_train_x,
            train_Y=train_obj,
            outcome_transform=Standardize(m=train_obj.shape[-1])
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                fit_gpytorch_mll(mll)

        # 2. 定義採集函數
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model          = model,
            ref_point      = ref_point,
            X_baseline     = std_train_x,
            sampler        = sampler,
            prune_baseline = True,
        )

        # 3. 設定範圍與約束 (Sum = 1.0)
        bounds = torch.zeros(2, D, device=device, dtype=dtype)
        bounds[1] = 1.0
        constraints = [(
            torch.arange(D, device=device), 
            torch.ones(D, dtype=dtype, device=device), 
            torch.tensor([1.0], device=device, dtype=dtype)
        )]

        # 4. 優化採集函數 (增加 raw_samples 減少警告)
        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            equality_constraints=constraints,
            q=1,
            num_restarts=10,
            raw_samples=128, # 建議增加
            options={"batch_limit": 5, "maxiter": 100}
        )


        #candidate in  [0,1]
        #new_x in [0,100]
        # 5. 更新資料

        new_y = evaluate_oracle(candidate, oracle)  # (n,2)
        
        new_x = candidate*100
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_y])

        # 6. 計算 HV
        current_hv = get_current_hv(train_obj, ref_point)

        current_hvs.append(current_hv)
        t.set_postfix(hv=f"{current_hv:.4f}")

    # --- 儲存結果 CSV ---


    df_x = pd.DataFrame(train_x.cpu().numpy(), columns=x_name)
    df_y = pd.DataFrame(train_obj.cpu().numpy(), columns=y_name)

    # 3. 合併 X 與 Y
    df_final = pd.concat([df_x, df_y], axis=1)


    # 4. 命名並儲存
    file_name = f"x_sparse_{seed}.csv"
    save_dir = "D:/Users/TingYuLin/Desktop/py12/MOBO/out_data"
    save_path = os.path.join(save_dir, file_name)
    df_final.to_csv(save_path, index=False)
    
    return current_hvs



def preprocess_data(df, target_cols=["SPGR", "TE"], device="cuda", dtype=torch.double):
    """
    將 DataFrame 轉換為 BoTorch 所需的 train_x 與 train_y Tensor。
    
    參數:
        df: pd.DataFrame, 輸入的原始資料
        target_cols: list, 作為目標變數的欄位名稱
        device: 運算設備 (cpu 或 cuda)
        dtype: 數值型態 (建議使用 torch.double 以維持數值穩定性)
    """
    # 1. 取得目標變數 (train_y)
    y_name = target_cols
    train_y = torch.tensor(df[y_name].values, device=device, dtype=dtype)
    
    # 2. 取得特徵變數 (train_x): 排除 target 欄位的所有欄位
    x_name = [col for col in df.columns if col not in y_name]
    train_x = torch.tensor(df[x_name].values, device=device, dtype=dtype)
    
    return train_x, train_y, x_name, y_name


def load_beta_oracle(beta_csv_path,
                    feature_cols,
                    target_cols,
                    device=device,
                    dtype=dtype) :
    """
    Read beta1.csv and build a fast oracle:
      - intercept: (m,)
      - beta_lin: (m, q)
      - pairs: (K, 2)  (indices for i<j)
      - beta_inter: (m, K)
    """
    df = pd.read_csv(beta_csv_path)

    # Optional: only keep active coefficients (if any are False)
    if 'active' in df.columns:
        df = df[df['active'].astype(bool)].copy()

    # ---- intercept ----
    inter = df[df['type'].str.lower().eq('intercept_correction')]
    if inter.empty:
        raise ValueError("No row with type == 'intercept_correction' found in beta CSV.")
    intercept = torch.tensor(inter.iloc[0][target_cols].values.astype(float),
                             device=device, dtype=dtype)  # (m,)

    # map material -> index
    mat_to_idx = {m: i for i, m in enumerate(feature_cols)}

    # ---- linear terms ----
    lin_df = df[df['type'].str.lower().eq('linear')].copy()
    lin_df['mat'] = lin_df['feature'].str.extract(r"x\[(.+?)\]")
    lin_df['idx'] = lin_df['mat'].map(mat_to_idx)
    if lin_df['idx'].isna().any():
        missing = lin_df[lin_df['idx'].isna()]['mat'].unique().tolist()
        raise ValueError(f"Linear beta has unknown materials: {missing}")

    beta_lin = torch.zeros((len(target_cols), len(feature_cols)), device=device, dtype=dtype)
    for _, row in lin_df.iterrows():
        beta_lin[:, int(row['idx'])] = torch.tensor(row[target_cols].values.astype(float),
                                                    device=device, dtype=dtype)

    # ---- interaction terms ----
    int_df = df[df['type'].str.lower().eq('interaction')].copy()
    mats = int_df['feature'].str.extract(r"x\[(.+?)\]\*x\[(.+?)\]")
    int_df['mat_i'] = mats[0]
    int_df['mat_j'] = mats[1]
    int_df['i'] = int_df['mat_i'].map(mat_to_idx)
    int_df['j'] = int_df['mat_j'].map(mat_to_idx)
    if int_df[['i', 'j']].isna().any().any():
        missing_i = int_df[int_df['i'].isna()]['mat_i'].unique().tolist()
        missing_j = int_df[int_df['j'].isna()]['mat_j'].unique().tolist()
        raise ValueError(f"Interaction beta has unknown materials: {missing_i + missing_j}")

    pairs = torch.tensor(int_df[['i', 'j']].values.astype(int),
                         device=device, dtype=torch.long)  # (K,2)
    beta_inter = torch.tensor(int_df[target_cols].values.astype(float),
                              device=device, dtype=dtype).T  # (m,K)

    return {
        "intercept": intercept,          # (m,)
        "beta_lin": beta_lin,            # (m,q)
        "pairs": pairs,                  # (K,2)
        "beta_inter": beta_inter,        # (m,K)
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    }


def evaluate_oracle(X_tensor, oracle):
    """
    X_tensor: (n, q) in fraction scale (0~1), with sum(x)=1
    return:   (n, m) for [SPGR, TE]
    """
    intercept = oracle["intercept"]      # (m,)
    beta_lin = oracle["beta_lin"]        # (m,q)
    pairs = oracle["pairs"]              # (K,2)
    beta_inter = oracle["beta_inter"]    # (m,K)

    # linear: (n,q) @ (q,m) -> (n,m)
    lin_term = X_tensor @ beta_lin.T

    # interactions: build (n,K) of x_i * x_j
    cross = X_tensor[:, pairs[:, 0]] * X_tensor[:, pairs[:, 1]]  # (n,K)
    inter_term = cross @ beta_inter.T  # (n,m)

    Y = intercept.unsqueeze(0) + lin_term + inter_term
    return Y






if __name__ == "__main__":

    # ==========================================
    # 主程式執行流程
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64

    BETA_CSV = "D:/Users/TingYuLin/Desktop/py12/MOBO/beta1.csv"
    FEATURE_COLS = ['AA001', 'AA002', 'AA004', 'AA005', 'AA006', 'AW001', 'AW003', 'AW004', 'AW005', 'AW014', 'AX010', 'AX015', 'AX019', 'AX020', 'AX029', 'AX032', 'AX137', 'CM1002', 'CM1007', 'CM1008', 'CP2002', 'FR001', 'FR002', 'FR006', 'FR007', 'FR008', 'FR015', 'FR058', 'GF001', 'GF006', 'GF013', 'GF014', 'GF016', 'GF020', 'MF001', 'MF005', 'MF006', 'MF007', 'PR002', 'PR007', 'PR009', 'PR016', 'PR020', 'PR022', 'PR024', 'SS004', 'SS010']
    TARGET_COLS = ['SPGR', 'TE']

    DATA_PATH_39 = "D:/Users/TingYuLin/Desktop/py12/MOBO/synthetic_data_sparse_seed_39.csv"
    DATA_PATH_40 = "D:/Users/TingYuLin/Desktop/py12/MOBO/synthetic_data_sparse_seed_40.csv"
    DATA_PATH_41 = "D:/Users/TingYuLin/Desktop/py12/MOBO/synthetic_data_sparse_seed_41.csv"


    data_39 = pd.read_csv(DATA_PATH_39)
    data_40 = pd.read_csv(DATA_PATH_40)
    data_41 = pd.read_csv(DATA_PATH_41)

    data_seed = [data_39, data_40, data_41]

    N_ITER = 50
    MODEL_PATH = "D:/Users/TingYuLin/Desktop/py12/MOBO/oracle_model.pt"

    oracle = load_beta_oracle(BETA_CSV, FEATURE_COLS, TARGET_COLS, device=device, dtype=dtype)
    ref_point = torch.tensor([1.3653894, 2.848232], device=device, dtype=dtype)

    all_seed_hvs = []
    seed = 39
    for seed_df in data_seed:
        
        train_x, train_obj, x_name, y_name = preprocess_data(
            df = seed_df, 
            target_cols=["SPGR", "TE"], 
            device="cuda", 
            dtype=torch.double)
        

        hv_history = run_bo_experiment(seed        = seed, 
                                       initial_x   = train_x, 
                                       initial_y   = train_obj, 
                                       n_iter      = N_ITER, 
                                       ref_point   = ref_point, 
                                       oracle      = oracle,
                                       x_name      = x_name, 
                                       y_name      = y_name)
        all_seed_hvs.append(hv_history)
        seed +=1

    # ==========================================
    # 繪圖階段
    # ==========================================
    all_seed_hvs = np.array(all_seed_hvs) # Shape: (num_seeds, n_iter)
    mean_hvs = np.mean(all_seed_hvs, axis=0)
    std_hvs = np.std(all_seed_hvs, axis=0)
    iters = np.arange(1, N_ITER + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iters, mean_hvs, label="Mean Hypervolume", color='blue', linewidth=2)
    # 繪製 95% 置信區間 (1.96 * std / sqrt(n))
    ci = 1.96 * std_hvs / np.sqrt(len(data_seed))
    plt.fill_between(iters, mean_hvs - ci, mean_hvs + ci, color='blue', alpha=0.2, label="95% CI")

    plt.xlabel("Iteration")
    plt.ylabel("Hypervolume")
    plt.title("Aggregate MOBO Performance (sparse(2-14)、無 constrain)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("mobo_performance_final.png")
    plt.show()