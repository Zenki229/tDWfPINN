#!/bin/bash

# 基础配置
ALPHA=1.50
PLOT_BACKEND='matplotlib'
export PYTHONPATH=$(pwd)
project_name='EG1-small-batch'

# 定义两组不同的 Num (M) 列表
# 对应表格左侧 (MC方法)
MC_NUMS=(80 160 320 640 1280)
# 对应表格右侧 (GJ方法)
GJ_NUMS=(16 32 48 64 80)

# 定义要运行的方法列表
METHODS=("MC-I" "MC-II" "GJ-I" "GJ-II")

# 循环遍历方法
for method in "${METHODS[@]}"; do
    
    # 根据 method 的名字判断使用哪一组 num 列表
    if [[ "$method" == "MC-I" || "$method" == "MC-II" ]]; then
        CURRENT_NUMS=("${MC_NUMS[@]}")
    else
        # 假设剩下的就是 GJ-I 和 GJ-II
        CURRENT_NUMS=("${GJ_NUMS[@]}")
    fi

    # 循环遍历对应的 num (M)
    for num in "${CURRENT_NUMS[@]}"; do
        
        exp_name="${method}_${num}"
        
        echo "=========================================================="
        echo "Starting Task: Method=${method}, Num=${num}"
        echo "Project Name: ${project_name}"
        echo "=========================================================="

        # 运行 Python 脚本
        # 注意：为了防止某个任务显存溢出(OoM)导致整个脚本停止，
        # 这里没有用 set -e，如果 python 报错，bash 会继续执行下一个循环
        python src/train.py \
            pde=dw_eg1 \
            pde.alpha=${ALPHA} \
            pde.method=${method} \
            pde.monte_carlo_params.nums=$num \
            pde.gauss_jacobi_params.nums=$num \
            trainer.rad.use=False \
            trainer.max_steps=10 \
            trainer.batch_size.domain=5000 \
            trainer.batch_size.boundary=1000 \
            trainer.batch_size.initial=1000 \
            optimizer.epochs=5000 \
            wandb.mode='online' \
            wandb.project=$project_name \
            wandb.name=$exp_name
            plot=${PLOT_BACKEND}

        # 检查上一个命令的退出状态
        if [ $? -ne 0 ]; then
            echo "Warning: Task [Method=${method}, Num=${num}] failed (possibly OoM). Continuing to next task..."
        else
            echo "Task [Method=${method}, Num=${num}] completed successfully."
        fi
        
        # 可选：如果你希望每个任务之间稍微停顿一下释放资源
        # sleep 5 

    done
done