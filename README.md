# 从零实现 Transformer (Encoder-Decoder)

本项目使用 PyTorch 从零开始实现了一个标准的 Transformer 模型（包含编码器-解码器架构）。这是《大模型基础与应用》课程的期中作业。项目旨在深入理解 Transformer 的核心组件，包括多头自注意力、位置编码、残差连接与层归一化等。

模型在 **Multi30k (英语-德语)** 机器翻译任务上进行了训练和验证。

## 主要特性

* **编码器-解码器架构:** 完整实现了编码器和解码器堆栈。
* **Transformer 核心组件:**
    * 缩放点积注意力 (Scaled Dot-Product Attention)
    * 多头注意力 (Multi-Head Attention)
    * 逐位置前馈网络 (Position-wise Feed-Forward Networks)
    * 残差连接与层归一化 (Add & Norm)
    * 正弦位置编码 (Sinusoidal Positional Encoding)
    * 输入/输出词嵌入 (Input/Output Embeddings)
* **掩码机制:** 实现了源序列填充掩码 (Source Padding Mask) 和目标序列填充及未来掩码 (Target Padding/Future Mask)。
* **训练流程:** 包含必要的训练组件：
    * AdamW 优化器
    * 学习率调度器 (`ReduceLROnPlateau`)
    * 梯度裁剪 (Gradient Clipping)
    * 交叉熵损失函数（忽略填充标记）
    * 模型保存与加载
    * 训练曲线可视化
* **可复现性:** 使用固定的随机种子，并提供精确的运行命令。

## 代码结构
仓库结构组织如下：
transformer-from-scratch/ |-- src/ # 源代码 | |-- model.py # Transformer, Encoder, Decoder 核心架构 | |-- modules.py # MultiHeadAttention, FFN, AddNorm, PositionalEncoding 等子模块 | |-- train.py # 训练与评估主脚本 | |-- data.py # 数据加载、预处理、词表 (使用 Multi30k) | -- utils.py # 辅助工具 (例如，绘图) |-- results/ # 存放输出的图表 | |-- train_loss_curve.png # 示例：标准模型的训练曲线图 | -- ablation_loss_curve.png # 示例：标准模型与消融实验的对比图 |-- models/ # 存放训练好的模型检查点 | -- multi30k_transformer.pt # 示例：保存的模型文件 |-- scripts/ | -- run.sh # 便于运行训练的 Bash 脚本 |-- requirements.txt # Python 依赖项列表 `-- README.md # 本文件

## 环境设置

### 先决条件

* Python (测试版本为 3.10)
* Conda 或其他虚拟环境管理器 (推荐)

### 安装步骤

1.  **克隆仓库:**
    ```bash
    git clone [https://github.com/your-username/transformer-from-scratch.git](https://github.com/your-username/transformer-from-scratch.git) # <-- 替换为你的仓库 URL
    cd transformer-from-scratch
    ```

2.  **(可选，但推荐) 创建虚拟环境:**
    ```bash
    conda create -n transformer python=3.10
    conda activate transformer
    # 或者使用 venv: python -m venv venv && source venv/bin/activate
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

## 如何使用

### 数据准备

`Multi30k (EN-DE)` 数据集会在首次运行训练脚本时由 `torchtext` 自动下载。安装步骤中下载的 SpaCy 模型将用于分词。

### 模型训练

最简单的开始训练（使用默认超参数）的方式是运行提供的 Bash 脚本：

```bash
bash scripts/run.sh

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