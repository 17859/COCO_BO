import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from botorch.models.deterministic import GenericDeterministicModel
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.penalized import L0Approximation
from botorch.optim import (
    Homotopy,
    HomotopyParameter,
    LogLinearHomotopySchedule,
    optimize_acqf_homotopy,
)
from botorch.models import SaasFullyBayesianSingleTaskGP, ModelList
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.utils.sampling import draw_sobol_samples
from sklearn.model_selection import train_test_split

import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


# ======================================================
# 0. 基本設定
# ======================================================

F_BEST = 150.0  # oracle optimum (given)
SEED = 42
tkwargs = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}
torch.manual_seed(SEED)
np.random.seed(SEED)

print('tkwargs = ',tkwargs)


# ======================================================
# 1. 資料讀取
# ======================================================
data = pd.read_csv('D:/Users/TingYuLin/Desktop/py12/20260127.csv')
data_new = data.drop(columns=['MI',	'MV','SPGR','ASH','TE','TM','FS','FM','IS'])
data_cleaned = data_new.dropna()


# 準備特徵 X 與目標 y
X = data_cleaned.drop(columns=['TS'])
y = data_cleaned['TS']

# 切分訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
param_names = list(X_train.columns) 
N_DIM = X_train.shape[1]



# ======================================================
# 2. XGBoost oracle
# ======================================================
model_xgb = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",      # 使用直方圖演算法
    device="cuda",           # 關鍵：指定 XGBoost 使用 CUDA
    objective='reg:squarederror',
    random_state=42
)

# 訓練模型，加入 early_stopping 防止過擬合
model_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

preds_xgb = model_xgb.predict(X_test)
def calculate_metrics(y_true, y_pred, model_name):
    # Global RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Spearman Rho (Rank Consistency)
    # spearmanr 回傳 (correlation, p-value)
    rho, _ = spearmanr(y_true, y_pred)
    
    return {
        "Model": model_name,
        "Global RMSE": round(rmse, 4),
        "Spearman Rho": round(rho, 4)
    }

# 執行計算
xgb_metrics = calculate_metrics(y_test, preds_xgb, "XGBoost")
results = pd.DataFrame([xgb_metrics])
print('-----'*50)
print(results)
print('-----'*50)


# ======================================================
# 3. 初始資料 (Sobol)
# ======================================================
N_BATCH = 20         # 疊代次數 (視計算資源調整)
BATCH_SIZE = 1       # 每次產生的候選點數量 (q)
N_INIT = 125*2       # 初始隨機點數量 (建議設為 2*N_DIM)


# 設定搜尋邊界 (0-1) 與 SEBO 目標點
bounds = torch.stack([torch.zeros(N_DIM, **tkwargs), torch.ones(N_DIM, **tkwargs)])
target_point = torch.zeros(N_DIM, **tkwargs)


# def get_xgb_predictions(X_tensor, model_xgb, feature_names):
#     X_real = 100.0 * X_tensor  
#     df = pd.DataFrame(
#         X_real.cpu().numpy(),
#         columns=feature_names,
#     )
#     preds = model_xgb.predict(df)
#     return torch.tensor(preds, **tkwargs).unsqueeze(-1)

def get_xgb_predictions(X_tensor, model_xgb):
    X_real = 100.0 * X_tensor
    X_np = X_real.detach().cpu().numpy()
    preds = model_xgb.predict(X_np)
    return torch.tensor(preds, **tkwargs).unsqueeze(-1)



# ==========================================
# 4. BO + SEBO 主迴圈
# ==========================================
# 使用 Sobol 序列進行初始空間填充
train_X = draw_sobol_samples(bounds=bounds, n=N_INIT, q=1).squeeze(1)
train_X = train_X / train_X.sum(dim=-1, keepdim=True)

# train_Y = get_xgb_predictions(train_X, model_xgb, param_names)
train_Y = get_xgb_predictions(train_X, model_xgb)


y_history = []           
simple_regret = []
cumulative_regret = []
cum_reg = 0.0
best_so_far = -np.inf



print(f"Starting BoTorch-only SEBO optimization...")

for i in range(N_BATCH):

    # (A) 對 Y 進行標準化 (這對 SAASBO 和多目標優化非常重要)
    std = train_Y.std().clamp_min(1e-8)
    train_Y_std = (train_Y - train_Y.mean()) / std

    # (B) 建立 SAASBO 模型 (代理目標 Y)
    saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y_std)
    fit_fully_bayesian_model_nuts(saas_gp, warmup_steps=128, num_samples=128, thinning=8)


    # (C) 建立懲罰項模型 (L0 Norm)
    l0_penalty = L0Approximation(target_point=target_point,a=0.1)
    penalty_model = GenericDeterministicModel(f=lambda x: -l0_penalty(x))
    
    # 合併為多目標模型
    model_list = ModelList(saas_gp, penalty_model)

    # (D) 設定 MOO 參考點
    # 參考點代表你能接受的「最差值」。
    # 第 1 維是標準化後的 Y（建議設大一點），第 2 維是稀疏度（上限是 N_DIM）
    # Y：越大越好（設在偏低位置）
    y_ref = train_Y_std.min().item() - 0.5
    # Sparsity：active ≤ 10
    sparsity_ref = -10.0
    

    ref_point = torch.tensor([y_ref, sparsity_ref],**tkwargs)

    # (E) 定義 qLogNEHVI 採集函數
    acq_f = qLogNoisyExpectedHypervolumeImprovement(
        model=model_list,
        ref_point=ref_point,
        X_baseline=train_X,
        prune_baseline=True,
    )


    # (F) 同倫優化設定 (讓 L0 從平滑過渡到真實)Homotopy
    homotopy_schedule = LogLinearHomotopySchedule(start=0.1, end=1e-4, num_steps=30)
    homotopy = Homotopy(
        homotopy_parameters=[
            HomotopyParameter(parameter=l0_penalty.a, schedule=homotopy_schedule)
        ]
    )

    simplex_constraint = [
        (
            torch.arange(N_DIM, device=tkwargs["device"]),      # 所有特徵的索引
            torch.ones(N_DIM).to(**tkwargs), # 係數全部為 1
            1.0                       # 總和為 1
        )
    ]


    # ---- Optimize acquisition ----
    # (G) 執行同倫優化
    new_x, _ = optimize_acqf_homotopy(
        acq_function=acq_f,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=512,
        homotopy=homotopy,
        equality_constraints=simplex_constraint 
    )

    # (H) 後處理與資料更新
    new_y = get_xgb_predictions(new_x, model_xgb)


    # ---- Update dataset ----
    train_X = torch.cat([train_X, new_x])
    train_Y = torch.cat([train_Y, new_y])


    # ---- Regret tracking ----
    y_val = new_y.item()
    y_history.append(y_val)
    best_so_far = max(best_so_far, y_val)
    sr = F_BEST - best_so_far
    cr = F_BEST - y_val
    cum_reg += cr

    simple_regret.append(sr)
    cumulative_regret.append(cum_reg)

    # 計算統計量
    sparsity = (new_x > 1e-3).sum().item()
    print(
        f"Iter {i+1:02d} | "
        f"Y={y_val:.4f} | "
        f"Sparsity={sparsity}/{N_DIM} | "
        f"SimpleRegret={sr:.4f}"
    )



# ======================================================
# 5. Regret 圖
# ======================================================
plt.figure()
plt.plot(simple_regret, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Simple Regret")
plt.title("Simple Regret (f* = 150)")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(cumulative_regret, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Cumulative Regret")
plt.title("Cumulative Regret (f* = 150)")
plt.grid(True)
plt.show()




# ==========================================
# 6. 最佳解
# ==========================================
# 找到預測值最低的點 (如果你是最大化目標)
best_idx = torch.argmax(train_Y)
best_X = 100.0 * train_X[best_idx]


print("\n=== Best Found Solution ===")
print(f"Best Y = {train_Y[best_idx].item():.6f}")
print(f"Non-zero dims = {(best_X > 1e-3).sum().item()}")
best_df = pd.DataFrame(
    best_X.cpu().numpy().reshape(1, -1),
    columns=param_names,
)
print(best_df)