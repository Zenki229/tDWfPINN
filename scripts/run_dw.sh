#!/bin/bash

# ==========================================
# 全局基础配置
# ==========================================
PLOT_BACKEND='matplotlib'
export PYTHONPATH=$(pwd)
PROJECT_NAME='DW-table'

# ==========================================
# 1. 定义参数列表
# ==========================================

# Alpha 列表 (对应表格的大行)
ALPHA_LIST=(1.25 1.50 1.75)

# k 和 lambda 的组合列表 (对应表格的列)
# 格式: "k值:lambda值"
# 表格中共有四种组合: (k=1,λ=1), (k=2,λ=4), (k=4,λ=4), (k=6,λ=6)
CONDITIONS=(
    "1:1"
    "2:4"
    "4:4"
    "6:6"
)

# 方法列表
METHODS=("GJ-I" "GJ-II" "MC-I" "MC-II")

# ==========================================
# 2. 定义 Num (M) 列表 (严格对应表格内容)
# ==========================================
# 表格中 MC 方法只测了 80 和 640
MC_NUMS=(80 640)
# 表格中 GJ 方法只测了 16 和 80
GJ_NUMS=(16 80)


# ==========================================
# 3. 开始多层循环
# 顺序建议: Alpha -> Condition -> Method -> Num
# ==========================================

for alpha in "${ALPHA_LIST[@]}"; do
    
    for cond in "${CONDITIONS[@]}"; do
        
        # 解析 k 和 lambda
        IFS=':' read -r k_val lambda_val <<< "$cond"

        for method in "${METHODS[@]}"; do
            
            # --- A. 根据 Method 选择 Num 列表 ---
            if [[ "$method" == "MC-I" || "$method" == "MC-II" ]]; then
                CURRENT_NUMS=("${MC_NUMS[@]}")
            else
                CURRENT_NUMS=("${GJ_NUMS[@]}")
            fi

            # --- B. 遍历 M 值 ---
            for num in "${CURRENT_NUMS[@]}"; do
                
                # 构建实验名称 (方便在 wandb 中查找)
                # 格式: Method_M_Alpha_k_lambda
                exp_name="${method}_M${num}_AL${alpha}_k${k_val}_lam${lambda_val}"
                
                echo "----------------------------------------------------------------"
                echo "Running Task:"
                echo "  Method: ${method}, M: ${num}"
                echo "  Physics: Alpha=${alpha}, k=${k_val}, Lambda=${lambda_val}"
                echo "  Experiment: ${exp_name}"
                echo "----------------------------------------------------------------"

                # 运行 Python 脚本
                python src/train.py \
                    pde=dw_forward \
                    pde.alpha=${alpha} \
                    pde.method=${method} \
                    pde.lambda_val=${lambda_val} \
                    pde.k=${k_val} \
                    pde.monte_carlo_params.nums=$num \
                    pde.gauss_jacobi_params.nums=$num \
                    trainer.rad.use=False \
                    trainer.max_steps=3 \
                    trainer.batch_size.domain=5000 \
                    trainer.batch_size.boundary=1000 \
                    trainer.batch_size.initial=1000 \
                    optimizer.epochs=5000 \
                    wandb.mode='online' \
                    wandb.project=$PROJECT_NAME \
                    wandb.name=$exp_name \
                    plot=${PLOT_BACKEND}

                # 错误检测
                if [ $? -ne 0 ]; then
                    echo ">>> WARNING: Task ${exp_name} failed. Continuing..."
                fi

            done # End Num
        done # End Method
    done # End Condition
done # End Alpha

echo "All experiments completed."