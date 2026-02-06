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






def filter_data_by_groups(X, Y):
    """
    根據大類欄位加總範圍與 Y 值大於 0 篩選資料。
    
    Args:
        X (pd.DataFrame): 原始特徵資料
        Y (pd.DataFrame): 原始目標資料
        group_ranges (dict): 格式為 {'大類名稱': [min, max]} 的字典
        
    Returns:
        tuple: (filtered_x, filtered_y)
    """
    #定義範圍
    group_ranges = {
        'AA': [0, 1], 'AW': [0, 1], 'AX': [0, 10], 'CM': [0, 5],
        'CP': [0, 1], 'FR': [0, 20], 'GF': [0, 51], 'MF': [0, 20],
        'PR': [30, 90], 'SS': [0, 10]
    }


    # 1. 取得所有大類名稱
    categories = list(group_ranges.keys())

    # 2. 建立各類加總的暫存 DataFrame
    sums_df = pd.DataFrame(index=X.index)

    for cat in categories:
        # 找出所有以該大類開頭的欄位 (例如 AA001, AA002...)
        cat_cols = [c for c in X.columns if c.startswith(cat)]
        
        if cat_cols:
            # 計算該類橫向總和
            sums_df[cat] = X[cat_cols].sum(axis=1)
        else:
            # 若無此大類則補 0，避免後續比對出錯
            sums_df[cat] = 0.0

    # 3. 根據 group_ranges 進行範圍篩選 (Masking)
    range_mask = pd.Series([True] * len(X), index=X.index)
    for cat, (lower, upper) in group_ranges.items():
        condition = sums_df[cat].between(lower, upper)
        range_mask &= condition

    # 4. Y > 0 篩選 (所有目標都必須大於 0)
    y_positive_mask = (Y > 0).all(axis=1)

    # 5. 合併所有條件
    final_mask = range_mask & y_positive_mask

    # 6. 套用篩選
    filtered_x = X[final_mask].copy()
    filtered_y = Y[final_mask].copy()

    print(f"原始資料筆數: {len(X)}")
    print(f"篩選後筆數: {len(filtered_x)}")
    print(f"篩選後筆數: {len(filtered_y)}")

    return filtered_x, filtered_y




def prepare_multitask_data_long(train_x, train_y):
    """
    將數據轉換為 MultiTaskGP 所需的 Long Format。
    
    Args:
        train_x: [n, d] Tensor (你的 noise_train_x)
        train_y: [n, m] Tensor (你的 train_y)
        
    Returns:
        tuple: (long_x, long_y)
            long_x: [n*m, d+1] (最後一欄是任務索引 0, 1, 2...)
            long_y: [n*m, 1]
    """
    n, d = train_x.shape
    num_tasks = train_y.shape[-1]
    device = train_x.device
    dtype = train_x.dtype

    # 1. 擴展 X: 重複 X 矩陣 num_tasks 次
    # shape: [n*m, d]
    long_x_base = train_x.repeat(num_tasks, 1)

    # 2. 建立 Task Index 欄位: [0,0...1,1...2,2...]
    # shape: [n*m, 1]
    task_indices = torch.arange(num_tasks, device=device, dtype=dtype).repeat_interleave(n).unsqueeze(-1)

    # 3. 合併 X 與 Task Index
    # shape: [n*m, d+1]
    long_x = torch.cat([long_x_base, task_indices], dim=-1)

    # 4. 拉平 Y
    # shape: [n*m, 1]
    # 使用 flatten() 或 view 確保順序與 task_indices 匹配 (先排完 task 0 再排 task 1)
    long_y = train_y.transpose(0, 1).reshape(-1, 1)

    return long_x, long_y







def run_bo_iteration(filteredxy_x, filteredxy_y, ref_point, seed = 42, n_iterations = 3, proxy_model = '', model_type ='modellist', method='qNoisyLogHVEI'):
    """
    執行單次 BO 疊代：訓練模型並產出下一個建議點。
    
    Args:
        train_x (Tensor): 已經標準化 (0-1) 的訓練特徵
        train_y (Tensor): 原始目標值
        ref_point (Tensor): 超體積參考點
        method (str): 採集函數類型 'qLogEHVI' 或 'qNoisyLogHVEI'
        
    Returns:
        tuple: (candidate, model, acq_func)
    """

    X_train, X_test, y_train, y_test = train_test_split(filteredxy_x, filteredxy_y, test_size=0.9,  random_state=seed)
    train_x = torch.tensor(np.array(X_train), device=device,dtype=dtype)
    train_y = y_train.to(device=device, dtype=dtype)

    # test_y  = y_test
    # test_x  = torch.tensor(np.array(X_test), device=device,dtype=dtype)
    # 紀錄初始狀態的 HV
    initial_hv = get_current_hv(train_y, ref_point)
    hv_history = [initial_hv]



    for n in range(n_iterations):

        torch.manual_seed(seed)
        noise = torch.randn_like(train_x) * 1e-6 # 加入小雜訊以免程式一值跳出 NumericalWarning: A not p.d 加上noise
        std_train_x = (train_x / 100.0).to(device=device, dtype=dtype)
        noise_train_x = std_train_x + noise


        # print('------'*10)
        # print(f'-----開始進行 surrgate model 訓練  {n} -----')
        # print('------'*10)
        # --- 模型建立區 ---
        if model_type == 'ModelListGP':
            models = []
            num_objectives = train_y.shape[-1]
            for j in range(num_objectives):
                models.append(SingleTaskGP(
                    noise_train_x, 
                    train_y[:, j:j+1]
                ))
            
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            current_X_baseline = noise_train_x # Baseline 維度為 D


        elif  model_type == 'MultiTaskGP':
            mt_x, mt_y = prepare_multitask_data_long(noise_train_x, train_y)
            model = MultiTaskGP(mt_x, 
                                mt_y, 
                                task_feature=-1,
                                outcome_transform=Standardize(m=1)  # ⭐ 明確指定
                                )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            current_X_baseline = noise_train_x


        
        elif  model_type == 'KroneckerMultiTaskGP':
            model = KroneckerMultiTaskGP(noise_train_x, 
                                         train_y,
                                         outcome_transform=Standardize(m=train_y.shape[-1])) # 用這個 model 就可以不用 #, task_feature=-1
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            current_X_baseline = noise_train_x


        # --- 訓練模型 ---
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                fit_gpytorch_mll(mll)


        # --- 採集函數 ---
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        
        if method == 'qLogEHVI':
            partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
            acq_func = qLogExpectedHypervolumeImprovement(
                model        = model, 
                ref_point    = ref_point, 
                partitioning = partitioning,
                sampler      = sampler
            )

        elif method == 'qparEGO':
            # 1. 隨機產生權重向量 (Weights)，確保權重總和為 1 且與目標數量一致
            num_obj = train_y.shape[-1]
            weights = torch.randn(num_obj, device=device, dtype=dtype).abs()
            weights = weights / weights.sum()

            scalarization = get_chebyshev_scalarization(weights=weights, Y=train_y)     
            mc_objective = GenericMCObjective(scalarization)
            acq_func = qExpectedImprovement(
                model=model,
                best_f=scalarization(train_y).max(), # 找出目前純量化後的最佳值
                objective=mc_objective,
                sampler=sampler
            )


        else: # qNoisyLogHVEI
            acq_func = qLogNoisyExpectedHypervolumeImprovement(
                model          = model,
                ref_point      = ref_point,
                X_baseline     = current_X_baseline,
                prune_baseline = True,
                sampler        = sampler
            )

        


        # 優化採集函數，尋找下一個候選點
        D = train_x.shape[-1]
        bounds = torch.stack([torch.zeros(D, device=device, dtype=dtype), 
                            torch.ones(D, device=device, dtype=dtype)])
        
        # Set constraints
        constraints = [
            (
                torch.arange(D, device=device), # indices: X 的哪些維度要參與計算
                torch.ones(D, dtype=dtype, device=device), # coefficients: 這些維度的係數
                torch.tensor([1.0], device=device, dtype=dtype) # rhs: 等號右邊的值 (Sum = 1.0)
            )
        ]

        # print('+++++'*10)
        # print(f'+++ 開始進行 optimize_acqf  {n} ++++++')
        # print('+++++'*10)

        candidate, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            equality_constraints=constraints,
            q=1,
            num_restarts=5,
            raw_samples=20
        )


        # 從 oracle function 用 candidate 取得新的資料
        new_x = pd.DataFrame((candidate*100).cpu().numpy(), columns=data_info['oracle_model']['data_cols'])  # new_x in [0,100]
        new_y = proxy_model.predict(new_x)# 這裡給的 new_y 是尚未標準化的狀態 # new_x in 原始範圍
        new_y = torch.tensor(new_y, device=device) 

        # concate 舊的資料
        train_x = torch.concat([train_x, candidate*100])
        train_y = torch.concat([train_y, new_y])


        current_hv = get_current_hv(train_y, ref_point)
        hv_history.append(current_hv)
        print(f"Iteration {n+1}/{n_iterations} - HV: {current_hv:.4f}")


    return candidate, model, acq_func, hv_history, train_x, train_y






def run_experiment(filteredxy_x, filteredxy_y, ref_point, proxy_model, seeds, n_iterations, model_type, method,  x_name, y_name):
    """
    執行特定組合在多個 Seed 下的實驗，並記錄所有 HV 歷程。
    """
    all_hvs = []
    
    for s in seeds:
        print(f"\n>>>>>> 正在執行: {model_type} + {method} | Seed: {s}")
        # 這裡調用你修改後的 run_bo_iteration
        # 確保 run_bo_iteration 會回傳一個 list，包含初始 HV 到第 n 次迭代的 HV
        _, _, _, hv_history,  new_x_train, new_y_train = run_bo_iteration(
            filteredxy_x, 
            filteredxy_y, 
            ref_point, 
            seed=s, 
            n_iterations=n_iterations, 
            proxy_model=proxy_model, 
            model_type=model_type, 
            method=method
        )
        all_hvs.append(hv_history)

        save_dir = "D:/Users/TingYuLin/Desktop/py12/MOBO/result_data"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        if model_type == 'MultiTaskGP':
            pass

        else:
            df_x = pd.DataFrame(new_x_train.cpu().numpy(), columns=x_name)
            df_y = pd.DataFrame(new_y_train.cpu().numpy(), columns=y_name)
            # 3. 合併 X 與 Y
            df_final = pd.concat([df_x, df_y], axis=1)

            # 4. 命名並儲存
            file_name = f"{s}_{model_type}_{method}.csv"
            save_path = os.path.join(save_dir, file_name)
            df_final.to_csv(save_path, index=False)
            print(f"已儲存數據至: {save_path} (資料筆數: {len(df_final)})")


    return np.array(all_hvs) # Shape: [num_seeds, n_iterations + 1]





if __name__ == "__main__":
    seeds = [42,43,44]
    n_iterations = 300    # 迭代次數建議多一點（如 20-30）圖表才好看


    # 1. 載入資料
    data_info_path = "D:/Users/TingYuLin/Desktop/py12/MOBO/interactive_term/data.pkl"
    data_info = joblib.load(data_info_path)


    proxy_model = data_info['oracle_model']['model_info']['PIPE']

    total_x = data_info['initial_data']['X']
    total_y = data_info['initial_data']['Y']

    y_name = data_info['oracle_model']['target_cols']
    x_name = data_info['oracle_model']['data_cols']

    # 2. 篩選資料集 找到滿足 y>0 和 x>各大類群組上下界
    filteredxy_x, filteredxy_y = filter_data_by_groups(total_x, total_y)

    # 確保 filtered_y 是 Tensor，並放到正確的 device 上
    if isinstance(filteredxy_y, np.ndarray):
        filteredxy_y = torch.from_numpy(filteredxy_y).to(device=device, dtype=dtype)
    elif isinstance(filteredxy_y, pd.DataFrame):
        filteredxy_y = torch.tensor(filteredxy_y.values, device=device, dtype=dtype)

    ref_point = torch.min(filteredxy_y, dim=0).values - 0.1 * torch.abs(torch.min(filteredxy_y, dim=0).values)
    print('ref_point = ',ref_point)


    configs = {
            "ModelListGP + qLogEHVI": ("ModelListGP", "qLogEHVI"),
            "ModelListGP + qNoisyLogHVEI": ("ModelListGP", "qNoisyLogHVEI"),
            # "ModelListGP + qparEGO": ("ModelListGP", "qparEGO"),

            "KroneckerGP + qNoisyLogHVEI": ("KroneckerMultiTaskGP", "qNoisyLogHVEI"),
            "KroneckerGP + qLogEHVI": ("KroneckerMultiTaskGP", "qLogEHVI"),
            # "KroneckerGP + qparEGO": ("KroneckerMultiTaskGP", "qparEGO"),

            "MultiTaskGP + qNoisyLogHVEI": ("MultiTaskGP", "qNoisyLogHVEI"),
            "MultiTaskGP + qLogEHVI": ("MultiTaskGP", "qLogEHVI"),
            # "MultiTaskGP + qLogEHVI": ("MultiTaskGP", "qparEGO"),
        }


    results = {}

    # 3. 跑實驗
    for label, (m_type, method) in configs.items():
        hv_data = run_experiment(filteredxy_x, filteredxy_y, ref_point, proxy_model, seeds, n_iterations, m_type, method, x_name, y_name)
        results[label] = hv_data


    # 4. 繪圖
    plt.figure(figsize=(12, 7))
    iters = np.arange(n_iterations + 1)
    
    # 定義一些顏色，確保不同模型容易分辨
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (label, data) in enumerate(results.items()):
        # data shape: [num_seeds, n_iterations + 1]
        
        # 計算統計量
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # 如果 seed 數量較多，也可以用 95% 信心區間 (1.96 * std / sqrt(n))
        # ci = 1.96 * (std / np.sqrt(len(seeds))) 
        
        current_color = colors[i % len(colors)]
        
        # 畫平均線 (標記點點以便辨識迭代位置)
        plt.plot(iters, mean, label=label, color=current_color, linewidth=2.5, marker='o', markersize=4)
        
        # 畫變異範圍 (陰影)
        plt.fill_between(
            iters, 
            mean - std, 
            mean + std, 
            color=current_color, 
            alpha=0.15,      # 陰影透明度
            edgecolor=None
        )

    # 圖表美化
    plt.title("MOBO Hypervolume Convergence (Mean ± 1 Std Dev)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Hypervolume (HV)", fontsize=13)
    
    # 設定 X 軸刻度為整數
    plt.xticks(iters)
    
    # 加入網格，但使用較淡的顏色
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 圖例放在外面或右下角
    plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
    
    plt.tight_layout()
    
    # 儲存高解析度圖片
    plt.savefig("mobo_comparison_high_res.png", dpi=300)
    plt.show()




