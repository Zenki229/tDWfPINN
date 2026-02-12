#!/bin/bash

# ==========================================
# 全局基础配置
# ==========================================
PLOT_BACKEND='matplotlib'
export PYTHONPATH=$(pwd)
PROJECT_NAME='Burgers-table'

# ==========================================
# 1. 定义参数列表
# ==========================================

# Alpha 列表
ALPHA_LIST=(1.1 1.8)

# 方法列表
METHODS=("MC-II" "GJ-II")

# Num (M) 值
NUM=80

# ==========================================
# 2. 开始多层循环
# ==========================================

for alpha in "${ALPHA_LIST[@]}"; do
    
    # 根据 alpha 设置数据文件
    DATAFILE=""
    if [ "$alpha" == "1.1" ]; then
        DATAFILE="/root/tDWfPINN/data/burgers_110.npz"
    elif [ "$alpha" == "1.8" ]; then
        DATAFILE="/root/tDWfPINN/data/burgers_180.npz"
    else
        echo "Unknown alpha value: $alpha"
        exit 1
    fi

    for method in "${METHODS[@]}"; do
        
        # 构建实验名称
        exp_name="burgers_${method}_M${NUM}_AL${alpha}"
        
        echo "----------------------------------------------------------------"
        echo "Running Task:"
        echo "  Method: ${method}, M: ${NUM}"
        echo "  Physics: Alpha=${alpha}"
        echo "  Experiment: ${exp_name}"
        echo "----------------------------------------------------------------"

        # 准备参数
        COMMON_ARGS=(
            pde=burgers
            pde.alpha=${alpha}
            pde.datafile=${DATAFILE}
            pde.method=${method}
            trainer.max_steps=20
            optimizer.epochs=10000
            optimizer.lr=1e-4
            trainer.rad.use=true
            trainer.rad.ratio=0.3
            trainer.batch_size.domain=1500
            trainer.batch_size.boundary=500
            trainer.batch_size.initial=500
            wandb.project=${PROJECT_NAME}
            wandb.name=${exp_name}
            plot=${PLOT_BACKEND}
        )

        # 根据方法添加特定的 num 参数
        if [[ "$method" == "MC-II" ]]; then
            METHOD_ARGS=(pde.monte_carlo_params.nums=${NUM})
        elif [[ "$method" == "GJ-II" ]]; then
            METHOD_ARGS=(pde.gauss_jacobi_params.nums=${NUM})
        else
            METHOD_ARGS=()
        fi

        # 运行 Python 脚本
        python src/train.py "${COMMON_ARGS[@]}" "${METHOD_ARGS[@]}"

        # 错误检测
        if [ $? -ne 0 ]; then
            echo ">>> WARNING: Task ${exp_name} failed. Continuing..."
        fi

    done # End Method
done # End Alpha

echo "All experiments completed."
