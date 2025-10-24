# 从零实现 Transformer (Encoder-Decoder架构)

本项目使用 PyTorch 框架从零开始实现了一个标准的 Transformer 模型（包含编码器-解码器架构）。这是《大模型基础与应用》课程的期中作业。具体实现了 Transformer 的各种核心组件，包括多头自注意力、位置编码、残差连接与层归一化等。

模型在 **Multi30k (英语-德语)** 机器翻译任务上进行了训练和验证。

## 主要介绍

* **编码器-解码器架构:** 完整实现了编码器和解码器。
* **Transformer 核心组件:**
    * 缩放点积注意力
    * 多头注意力
    * 逐位置前馈网络
    * 残差连接与层归一化
    * 正弦位置编码
    * 输入/输出词嵌入
* **掩码机制:** 实现了源序列填充掩码 (Source Padding Mask) 和目标序列填充及未来掩码 (Future Mask)。
* **训练流程:** 包含必要的训练组件：
    * AdamW 优化器
    * 学习率调度器
    * 梯度裁剪
    * 交叉熵损失函数
    * 模型保存与加载
    * 训练曲线可视化

## 环境设置

### 先决条件

* Python (测试版本为 3.10)
* Conda 虚拟环境

### 安装步骤

1.  **克隆仓库:**
    ```bash
    git@github.com:1ikeduanfa/BJTU-LLMs-Mid-Term.git
    cd BJTU-LLMs-Mid-Term
    ```

2.  **创建虚拟环境:**
    ```bash
    conda create -n transformer python=3.10
    conda activate transformer
    ```

3.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```
   

4.  **下载 SpaCy 语言模型:** 数据预处理的分词步骤需要用到。
    ```bash
    python -m spacy download en_core_web_sm
    python -m spacy download de_core_news_sm
    ```

## 训练步骤

### 数据准备

`Multi30k (EN-DE)` 数据集会在首次运行训练脚本时由 `torchtext` 自动下载。安装步骤中下载的 SpaCy 模型将用于分词。

### 模型训练

最简单的开始训练（使用默认超参数）的方式是运行提供的 Bash 脚本：

```bash
sh scripts/run.sh

或者，你可以直接运行 train.py 脚本，并按需指定参数。run.sh 脚本中用于确保可复现性的精确命令（包含随机种子）如下：
python src/train.py \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 2 \
    --d_ff 512 \
    --batch_size 32 \
    --lr 3e-4 \
    --epochs 15 \
    --seed 42 \
    --save_path "models/multi30k_transformer.pt" \
    --plot_path "results/train_loss_curve.png"