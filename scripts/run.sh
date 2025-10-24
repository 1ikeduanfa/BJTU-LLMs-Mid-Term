#!/bin/bash
# --------------------------------------------------
# 此脚本用于按照指定超参在数据集 IWSLT2017 上训练
# --------------------------------------------------

set -e

echo "Step 1: 下载语spacy库中的语言模型"
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm

# 确保结果和模型目录存在
mkdir -p results 
mkdir -p models

# 定义输出路径
PLOT_PATH="results/train_loss_curve.png" 
MODEL_PATH="models/iwslt_transformer.pt"

echo "Step 2: 开始训练模型"

# 运行训练的精确命令 
# 超参数基于
python -m src.train
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_ff 512 \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 15 \
    --seed 42 \
    --save_path $MODEL_PATH \
    --plot_path $PLOT_PATH

echo "Training complete. Model saved to $MODEL_PATH"
echo "Training plot saved to $PLOT_PATH"