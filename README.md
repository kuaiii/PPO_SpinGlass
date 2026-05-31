# PPO_SpinGlass

基于自旋玻璃能量景观与深度强化学习的网络拓扑优化引擎。

将网络拆解（Dismantling）、构造（Construction）与重构（Rewiring）统一于**自旋玻璃能量景观**框架下，使用 **PPO（Proximal Policy Optimization）** 进行求解。核心思想是将拓扑操作视为自旋组态翻转，TGNN 学习耦合场，策略网络在能量景观中搜索最优轨迹。

## 核心特性

- **三种任务**：拆解（Dismantle）、构造（Construct）、重构（Rewiring）
- **物理驱动**：自旋玻璃哈密顿量作为状态能量，局部能量变化指导奖励塑形
- **GNN 策略**：3 层 GAT 编码器 + 注意力耦合层，学习节点嵌入与耦合矩阵
- **课程学习**：从 n=20 逐步扩展到 n=1000
- **双轨制训练**：规模扩展（稀疏大图）+ 密度适应（中小图全密度谱）

---

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 关键依赖版本
Python >= 3.8
PyTorch >= 2.0
torch-geometric >= 2.3
networkx >= 2.8
numpy >= 1.21
scipy >= 1.7
```

### CPU 训练重要提示

在 CPU 上训练小图 GNN 时，**PyTorch 默认多线程会导致严重性能退化**。代码已自动检测并在 CPU 模式下设置 `torch.set_num_threads(1)`，可获得 **120 倍加速**。

| 线程数 | GAT forward (n=20) |
|--------|-------------------|
| 8 (默认) | **400ms** |
| 1 | **2.5ms** |

---

## 快速开始

### 1. 运行验证测试

```bash
python3 main.py --mode test
```

### 2. 训练拆解模型

#### 基础训练（合成图，n=20）

```bash
python3 main.py --mode train --task dismantle \
    --n_nodes 20 --max_iters 300 \
    --num_episodes 20 --device cpu
```

#### 在真实图数据集上训练

```bash
python3 main.py --mode train --task dismantle \
    --train_data_dir dataset/data/raw/train \
    --max_train_graphs 500 --subgraph_size 100 \
    --n_nodes 20 --max_iters 300 \
    --num_episodes 20 --device cpu
```

#### 双轨制训练（推荐，用于 n=1000 大图）

**轨道 1：规模扩展（稀疏大图 n=100→1000）**

```bash
python3 main.py --mode train --task dismantle \
    --train_data_dir dataset/synth_training/sparse_large \
    --subgraph_size 1000 --max_iters 300 \
    --num_episodes 5 --device cpu \
    --n_nodes 100 --curriculum \
    --max_nodes 1000 --node_increment 50 --increment_every 10
```

**轨道 2：密度适应（中小图 n≤200，密度 0.001~0.5）**

```bash
# 先生成训练数据
python3 generate_training_graphs.py --sparse_num 800 --dense_num 500

# 然后训练
python3 main.py --mode train --task dismantle \
    --train_data_dir dataset/synth_training/dense_small \
    --subgraph_size 200 --max_iters 200 \
    --num_episodes 20 --device cpu \
    --n_nodes 100 --no-curriculum --random_graph_mix 0.7
```

### 3. 评估模型

```bash
python3 main.py --mode eval --task dismantle --device cpu --n_nodes 40
```

### 4. Benchmark 对比

与 network_dismantling 中的 18+ 种算法进行对比：

```bash
python3 compare_dismantling.py \
    --test_dir dataset/small_test \
    --checkpoint checkpoint_dismantle.pt \
    --max_graphs 20 --device cpu \
    --output_dir comparison_results
```

---

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 运行模式：train/eval/test | train |
| `--task` | 任务类型：dismantle/construct/rewiring | dismantle |
| `--n_nodes` | 初始图节点数（课程学习起点） | 20 |
| `--max_nodes` | 课程学习最大节点数 | 100 |
| `--node_increment` | 每次增加的节点数 | 10 |
| `--increment_every` | 每隔多少 iteration 增量 | 100 |
| `--max_iters` | 最大训练轮数 | 200 |
| `--num_episodes` | 每轮收集的 episode 数 | 20 |
| `--batch_size` | PPO batch size | 64 |
| `--train_data_dir` | 训练图数据目录（.gml 文件） | None |
| `--subgraph_size` | 大图子采样目标大小 | 100 |
| `--random_graph_mix` | 随机图混合比例（0~1） | 0.0 |
| `--hidden_dim` | GNN 隐藏维度 | 64 |
| `--num_heads` | GAT 注意力头数 | 4 |
| `--lr_policy` | 策略网络学习率 | 3e-4 |
| `--lr_value` | 值函数学习率 | 1e-3 |
| `--device` | 计算设备：auto/cpu/cuda | auto |

---

## 训练数据准备

### 已有数据集

| 数据集 | 文件数 | 规模 | 说明 |
|--------|--------|------|------|
| `dataset/data/raw/train/` | 4154 | n=7~500 | 混合 BA/ER/WS/真实网络 |
| `dataset/small_test/` | 125 | n=10~49 | 小规模测试集 |
| `dataset/test_converted/` | 69 | n=20~1.4M | 真实世界网络（.gt 转换）|
| `dataset/test_synth_converted/` | 93 | n=1k~100k | 合成网络（ER/SBM/PowerLaw）|
| `dataset/test_lfr_converted/` | 101 | n=16384 | LFR 基准图 |

### 生成合成训练数据

```bash
python3 generate_training_graphs.py \
    --output_dir dataset/synth_training \
    --sparse_num 800 --dense_num 500
```

生成两类数据：
- `sparse_large/`：n=100~1000，稀疏图（密度≤0.02）
- `dense_small/`：n=20~200，全密度谱（0.001~0.5）

---

## 性能优化

### 已实现的关键优化

1. **增量 spin state 更新**：环境 step 时不再全量重建 (n,n) dense 矩阵，只更新被移除节点的行/列
2. **增量 triadic cache**：稀疏图时增量删除相关三角形和公共邻居；稠密图（>5000 三角形）回退到全量重建
3. **AttentionCoupling top_k**：默认 top_k=50，降低稠密图的 softmax 开销
4. **Batch training**：PPO update 中每个 mini-batch 累加 loss 后做一次 backward+step
5. **CPU 单线程**：`torch.set_num_threads(1)`，小图 GNN 加速 120 倍

### 训练速度参考（CPU）

| 场景 | 每 iteration | 每 step |
|------|-------------|---------|
| n=20 稀疏 | ~20s | ~3ms |
| n=100 稀疏 | ~25s | ~5ms |
| n=1000 稀疏 (m≈4000) | ~15min | ~45ms |
| n=200 中等密度 | ~25s | ~40ms |

---

## Benchmark 结果

在 10 张真实网络（n=10~49）上的对比：

| 排名 | 方法 | Avg AUC | 说明 |
|------|------|---------|------|
| 🥇 1 | EGND | 3.30 | 专门 dismantling 算法 |
| 🥈 2 | GND | 3.31 | 专门 dismantling 算法 |
| 🥉 3 | **PPO_SpinGlass** | **3.35** | **本模型** |
| 4 | CI_L1 | 3.49 | Collective Influence |
| 5 | CI_L2 | 3.51 | Collective Influence |
| 6 | CoreHD | 3.71 | Coreness-based |
| ... | ... | ... | ... |
| 18 | random | 7.39 | 随机基线 |

**相比仅在小合成图上训练的模型**：
- 排名从 9/12 提升至 **3/18**
- AUC 从 4.23 降至 **3.35**（提升 21%）
- 与第 1 名差距从 +0.93 缩小至 **+0.05**

---

## 项目结构

```
.
├── main.py                      # 主入口（训练/评估/测试）
├── train_ppo.py                 # PPO 训练器
├── compare_dismantling.py       # Benchmark 对比脚本
├── generate_training_graphs.py  # 合成数据生成
├── eval.py                      # 评估函数
├── models/                      # 模型定义
│   ├── encoder.py               # TGNN GAT 编码器
│   ├── coupling.py              # 注意力耦合层
│   ├── policy.py                # 策略头（拆解/构造）
│   └── value.py                 # 值函数头
├── envs/                        # 环境定义
│   └── topology_env.py          # Dismantle/Construct/Rewiring 环境
├── utils/                       # 工具函数
│   ├── spin_glass.py            # 自旋玻璃物理计算
│   ├── graph_metrics.py         # 图度量与 PyG 转换
│   └── graph_loader.py          # 图数据加载
├── network_dismantling/         # 传统 dismantling 算法库
├── dataset/                     # 数据集目录
├── tests/                       # 验证测试
└── requirements.txt             # 依赖
```

---

## 引用

本项目基于自旋玻璃能量景观理论，结合图神经网络与深度强化学习，用于网络拓扑优化。
