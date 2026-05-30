#!/bin/bash
# 启动 PPO 训练（后台，无缓冲输出）
# 用法: bash run_training.sh [额外参数]

cd "$(dirname "$0")"

# 使用 -u 保证无缓冲输出，实时写入日志
nohup python3 -u main.py \
    --mode train \
    --task dismantle \
    --train_data_dir dataset/data/raw/train \
    --max_train_graphs 500 \
    --subgraph_size 100 \
    --max_iters 300 \
    --num_episodes 20 \
    --device cpu \
    --n_nodes 20 \
    --batch_size 64 \
    --hidden_dim 64 \
    --num_heads 4 \
    "$@" \
    > train_real_data.log 2>&1 &

echo "Training started with PID $!"
echo "Log: train_real_data.log"
