# 基于自旋玻璃能量景观与深度强化学习的网络拓扑优化引擎

## 项目概述

本项目将网络拆解（Dismantling）、构造（Construction）与重构（Rewiring）统一于**自旋玻璃能量景观**框架下，使用 **PPO（Proximal Policy Optimization）** 进行求解。核心思想是将拓扑操作视为自旋组态翻转，TGNN 学习耦合场 $J_{ij}$，策略网络在能量景观中搜索最优轨迹：拆解对应沿梯度上升最快的路径逃离势阱，构造对应沿梯度下降最快的路径弛豫到基态。

---

## 一、核心理论公式（必须严格实现）

### 1.1 自旋组态编码

给定图 $G_t=(V,E_t)$，潜在边集为完全图 $K_n$ 的边集 $\mathcal{E}$。自旋变量定义为：

$$
s_{ij}^{(t)} = 2A_{ij}^{(t)} - 1 \in \{+1, -1\}
$$

其中 $A_{ij}^{(t)}$ 为时刻 $t$ 的邻接矩阵。

### 1.2 哈密顿量（状态能量）

$$
\mathcal{H}(G_t) = -\sum_{(i,j)\in\mathcal{E}} J_{ij} s_{ij}^{(t)} - \gamma \sum_{(i,j,k)\in\Delta_t} s_{ij}^{(t)}s_{jk}^{(t)}s_{ki}^{(t)} - h \sum_{(i,j)\in\mathcal{E}} s_{ij}^{(t)}
$$

其中：
- $J_{ij}$ 由 TGNN 注意力输出：$J_{ij} = \text{Attn}(\mathbf{z}_i, \mathbf{z}_j) = \frac{(\mathbf{W}_Q\mathbf{z}_i)^\top(\mathbf{W}_K\mathbf{z}_j)}{\sqrt{d_k}}$
- $\gamma > 0$ 为三角闭合系数（超参数，默认 0.5）
- $h$ 为外部场：**构造任务 $h=+1.0$，拆解任务 $h=-1.0$，重构任务 $h=0.0$**

> **注意符号**：外部场项为 $-h\sum s_{ij}$。当 $h>0$ 且 $s_{ij}=+1$（连接）时，贡献为 $-h$（能量降低），系统偏好连接。

### 1.3 局部能量变化（奖励塑形核心）

**移除边 $(i,j)$**（$s_{ij}: +1 \to -1$）：

$$
\Delta\mathcal{H}_{ij}^{\text{remove}} = +2J_{ij} + 2\gamma \sum_{k\in\mathcal{N}(i)\cap\mathcal{N}(j)} s_{jk}s_{ki} + 2h
$$

**添加边 $(i,j)$**（$s_{ij}: -1 \to +1$）：

$$
\Delta\mathcal{H}_{ij}^{\text{add}} = -2J_{ij} - 2\gamma \sum_{k\in\mathcal{N}(i)\cap\mathcal{N}(j)} s_{jk}s_{ki} - 2h
$$

**Kawasaki 交换**（固定边数）：

$$
\Delta\mathcal{H}^{\text{swap}} = \Delta\mathcal{H}_{kl}^{\text{add}} + \Delta\mathcal{H}_{ij}^{\text{remove}}
$$

### 1.4 局部分子场（节点重要性）

$$
h_i = \sum_{j\in\mathcal{N}(i)} J_{ij} s_{ij}
$$

---

## 二、深度强化学习 MDP 形式化

| 要素 | 拆解 (Dismantling) | 构造 (Construction) | 重构 (Rewiring) |
|:---|:---|:---|:---|
| **状态 $s_t$** | 当前图 $G_t$，TGNN 嵌入 $\mathbf{Z}_t$，局部分子场 $\{h_i(t)\}$，当前步数 $t$ | 同上 | 同上，额外记录边预算 $|E_t|=|E_0|$ |
| **动作 $a_t$** | 选择节点 $v \in V_t^{\text{alive}}$ 移除（级联删除关联边） | 选择非边 $(i,j) \in \bar{E}_t$ 添加 | 选择边对：移除 $e_{\text{out}}\in E_t$，添加 $e_{\text{in}}\notin E_t$ |
| **转移** | $G_{t+1} = G_t \setminus \{v\}$ | $G_{t+1} = G_t \cup \{(i,j)\}$ | $G_{t+1} = (G_t \setminus e_{\text{out}}) \cup e_{\text{in}}$ |
| **终止** | $\sigma(G_t) \le \theta_\sigma n$ 或 $t=k_{\max}$ | $\lambda_2 \ge \theta_\lambda$ 或 $t=b_{\max}$ | $\lambda_2 \ge \theta_\lambda$ 或 $t=s_{\max}$ |
| **奖励 $r_t$** | $-\Delta\sigma + \alpha \Delta\mathcal{H} - c_{\text{step}}$ | $\Delta\lambda_2 - \alpha \Delta\mathcal{H} - c_{\text{step}}$ | $\Delta\lambda_2 - \alpha \Delta\mathcal{H} - c_{\text{step}}$ |

---

## 三、网络架构设计（PyTorch Geometric）

### 3.1 共享 TGNN 编码器 `TGNNEncoder`

- **输入**：`Data(x=node_features, edge_index=edge_index, edge_attr=optional)`
- **层数**：3 层 `GATConv`（Graph Attention），隐藏维度 64，头数 4
- **输出**：节点嵌入 $\mathbf{Z} \in \mathbb{R}^{n \times 64}$

### 3.2 注意力耦合层 `AttentionCoupling`

- **输入**：$\mathbf{Z}$
- **计算**：$\mathbf{J} = \text{softmax}(\mathbf{Z}\mathbf{W}_Q (\mathbf{Z}\mathbf{W}_K)^\top / \sqrt{64})$，对角线置 0
- **输出**：$J_{ij}$ 矩阵（$n \le 200$ 可用稠密，更大需稀疏化取 Top-K）

### 3.3 策略头与值函数头

#### 拆解策略头 `DismantlePolicyHead`
- **输入**：$\mathbf{Z}$，局部场 $\mathbf{h} \in \mathbb{R}^n$
- **拼接特征**：$[\mathbf{z}_i; h_i; \text{deg}(i)]$
- **网络**：2 层 MLP $\to$ 节点 logits $\phi \in \mathbb{R}^n$
- **动作掩码**：已移除节点 mask = $-\infty$
- **输出**：Categorical 分布，采样节点

#### 构造策略头 `ConstructPolicyHead`
采用**两阶段注意力采样**解决 $O(n^2)$ 动作空间问题：
1. **物理预筛选**：计算所有非边的 $\Delta\mathcal{H}_{ij}^{\text{add}}$，取 Top-K=100 作为候选集 $\mathcal{C}$
2. **策略精炼**：对候选边 $(i,j)\in\mathcal{C}$，边特征 $[\mathbf{z}_i; \mathbf{z}_j; J_{ij}; \Delta\mathcal{H}_{ij}]$，MLP $\to$ 边 logits $\psi \in \mathbb{R}^{K}$
- **动作掩码**：已存在边自动排除
- **输出**：在 $\mathcal{C}$ 上的 Categorical 分布

#### 值函数头 `ValueHead`
- **全局 readout**：`global_mean_pool(Z)` + 图级特征（当前边数、LCC、$\lambda_2$、$\mathcal{H}$）
- **网络**：2 层 MLP $\to$ 标量 $V(s)$

---

## 四、环境设计

### 4.1 拆解环境 `DismantleEnv`

- `reset(G: nx.Graph)`：初始化状态，计算初始 $\mathcal{H}_0$, $\sigma_0$
- `step(node_idx: int)`：
  1. 移除节点及其所有边
  2. 计算新 LCC $\sigma_{t+1}$，新能量 $\mathcal{H}_{t+1}$
  3. 奖励：$r_t = (\sigma_t - \sigma_{t+1})/n + 0.1 \cdot (\mathcal{H}_{t+1} - \mathcal{H}_t) - 0.01$
  4. 终止：$\sigma \le 0.1n$（成功，额外 +10）或节点耗尽
- `observation()`：返回 PyG Data 对象（当前图状态）

### 4.2 构造环境 `ConstructEnv`

- `reset(G: nx.Graph, budget: int)`：初始化，记录最大添加边数 $b_{\max}$
- `step(edge: Tuple[int,int])`：
  1. 添加边
  2. 计算新 $\lambda_2$，新能量
  3. 奖励：$r_t = (\lambda_{2,t+1} - \lambda_{2,t}) - 0.05 \cdot (\mathcal{H}_{t+1} - \mathcal{H}_t) - 0.01$
  4. 终止：$\lambda_2 \ge \theta_\lambda$（成功，额外 +10）或预算耗尽
- 动作空间掩码：只允许当前不存在的边

### 4.3 重构环境 `RewiringEnv`

- `reset(G: nx.Graph, max_swaps: int)`：严格保持边数不变
- `step(swap: Tuple[edge_out, edge_in])`：
  1. 先移除 $e_{\text{out}}$，再添加 $e_{\text{in}}$
  2. 验证 $|E|$ 不变，否则拒绝
  3. 奖励同构造，使用 Kawasaki 能量变化 $\Delta\mathcal{H}^{\text{swap}}$
- 动作空间：通过构造头采样 $e_{\text{in}}$，拆解头采样 $e_{\text{out}}$，或联合采样

---

## 五、PPO 训练器 `SpinGlassPPOTrainer`

### 5.1 超参数

- 学习率：$3\times 10^{-4}$（策略），$1\times 10^{-3}$（值函数）
- $\gamma = 0.99$，GAE $\lambda = 0.95$
- Clip $\epsilon = 0.2$
- 每轮收集 $N=20$ 个 episode，训练 4 epochs，batch size 64

### 5.2 训练循环

```python
for iteration in range(max_iters):
    # 收集轨迹
    trajectories = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_buffer = []
        done = False
        while not done:
            with torch.no_grad():
                Z = encoder(obs)
                J = coupling(Z)
                h = local_field(J, obs.edge_index)
                # 根据任务选择头
                if task == 'dismantle':
                    action, log_prob = dismantle_head.sample(Z, h, mask)
                elif task == 'construct':
                    candidates = physical_topk(J, obs, gamma, h_field, K=100)
                    action, log_prob = construct_head.sample(Z, J, candidates)
                v = value_head(Z, obs)

            obs_next, reward, done, info = env.step(action)
            episode_buffer.append((obs, action, reward, v, log_prob, mask))
            obs = obs_next

        # 计算 GAE 和 Returns
        advantages, returns = compute_gae(episode_buffer, gamma=0.99, lam=0.95)
        trajectories.extend(episode_buffer)

    # PPO 更新
    for epoch in range(K_epochs):
        for batch in dataloader(trajectories, batch_size=64):
            # 重新计算当前策略的 log_prob 和 value
            new_log_prob, entropy = policy(batch.obs, batch.action)
            new_v = value(batch.obs)

            ratio = torch.exp(new_log_prob - batch.old_log_prob)
            surr1 = ratio * batch.advantage
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch.advantage
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            value_loss = F.mse_loss(new_v, batch.returns)

            loss = policy_loss + 0.5 * value_loss
            loss.backward()
            optimizer.step()
```

### 5.3 联合训练对齐损失

每 5 个 iteration，采样一对拆解和构造的完整轨迹，计算：

```python
# 取两个任务的中间状态
Z_d = encoder(G_d_mid)
Z_c = encoder(G_c_mid)
loss_align = F.mse_loss(Z_d.mean(dim=0), Z_c.mean(dim=0))  # 简化版嵌入对齐
```

加入总损失：`total_loss += 0.1 * loss_align`

---

## 六、关键实现细节

1. **$\lambda_2$ 计算**：使用 `scipy.sparse.linalg.eigsh` 或 `torch.lobpcg` 计算 Laplacian 第二小特征值。对于小图可用稠密特征值分解。
2. **三角项缓存**：预计算 `common_neighbors` 字典或 `edge_index` 的三角环列表，避免每步 $O(n^3)$ 枚举。能量变化公式中的 $\sum_{k\in\mathcal{N}(i)\cap\mathcal{N}(j)} s_{jk}s_{ki}$ 可通过节点交集快速计算。
3. **动作掩码**：使用 `torch.where(mask, logits, -1e9)` 实现 Categorical 采样掩码，确保无效动作概率为 0。
4. **课程学习**：训练初期 $n=20$，每 100 iteration 增加 $n$ 至 50, 100。图生成器使用 `nx.barabasi_albert_graph` 和 `nx.watts_strogatz_graph` 混合。
5. **奖励归一化**：每个 episode 内对奖励做 Z-score 归一化，或维护运行均值方差（RunningMeanStd）。

---

## 七、文件结构与依赖

### 7.1 依赖库

```text
torch >= 2.0
torch-geometric >= 2.3
networkx >= 2.8
numpy >= 1.21
scipy >= 1.7
matplotlib >= 3.5
tensorboard
```

### 7.2 文件结构

```
/project_root
├── models/
│   ├── encoder.py       # TGNNEncoder
│   ├── coupling.py      # AttentionCoupling
│   ├── policy.py        # DismantlePolicyHead, ConstructPolicyHead
│   └── value.py         # ValueHead
├── envs/
│   └── topology_env.py  # DismantleEnv, ConstructEnv, RewiringEnv
├── utils/
│   ├── spin_glass.py    # Hamiltonian, delta_remove, delta_add, local_field
│   └── graph_metrics.py # LCC, lambda2, triadic_term
├── train_ppo.py         # SpinGlassPPOTrainer 主循环
└── eval.py              # 评估与可视化
```

---

## 八、测试验证要求

1. **能量一致性测试**：随机图 $n=15$，随机执行 5 步动作，验证 `env.H` 的增量严格等于 `delta_*` 公式计算值（误差 < 1e-6）。
2. **PPO 收敛测试**：在 $n=20$ 星型图上训练拆解策略，验证 200 iteration 内学会优先移除中心节点，且 LCC AUC 优于随机策略 > 30%。
3. **守恒测试**：重构环境运行 10 步，每步断言 `len(env.edges) == env.initial_edge_count`。
4. **物理引导测试**：固定随机 $J$，比较 $h=+1$ 与 $h=-1$ 时策略的动作分布差异（构造倾向高 $J_{ij}$ 边，拆解倾向高局部场节点）。

---

## 九、算法选型说明：PPO 而非 GRPO

| 维度 | PPO | GRPO |
|:---|:---|:---|
| **值函数必要性** | ✅ 需要 $V(s)$，物理上精确对应**能量景观高度**（剩余势能/自由能估计） | ❌ 无值函数，仅用组内相对排名估计优势 |
| **状态空间特性** | 图结构状态有明确度量（$\lambda_2$, LCC, $\mathcal{H}$），值函数可监督预训练 | 长文本生成任务中状态价值难以定义，但图状态价值可解析 |
| **Episode 长度** | 短（$O(n)$ 步），适合带基线的方差缩减 | 长序列生成优势区，短 episode 采样效率低 |
| **动作空间** | 支持大规模离散动作（Gumbel-Softmax + 掩码） | 主要用于文本 token 采样，图组合动作适配差 |
| **物理可解释性** | 优势函数 $A(s,a) = Q(s,a)-V(s)$ 可解释为"该动作相对于平均能量变化的超额收益" | 缺乏显式状态价值，难以注入物理先验 |

**PPO 的物理对应**：值函数 $V_\theta(s_t)$ 学习的是从当前组态到终止的**期望累积能量变化**。策略更新中的 clip 机制防止在能量景观的陡峭区域发生策略崩塌（对应物理中的相变点附近梯度爆炸）。

---

## 十、Code Plan Mode Prompt（可直接使用）

```markdown
# 任务：基于自旋玻璃能量景观与PPO的网络拓扑DRL优化引擎

## 角色设定
你是一位精通深度强化学习、图神经网络与统计物理的算法工程师。请基于以下理论框架，编写一个基于 PyTorch + PyTorch Geometric (PyG) 的完整可训练代码。要求模块化设计，支持拆解、构造、重构三种任务，并严格遵循自旋玻璃能量景观的物理约束。

## 核心理论公式（必须严格实现）
- 自旋组态：$s_{ij} = 2A_{ij} - 1 \in \{+1, -1\}$
- 哈密顿量：$\mathcal{H} = -\sum J_{ij}s_{ij} - \gamma \sum_{\Delta} s_{ij}s_{jk}s_{ki} - h\sum s_{ij}$
- 局部能量变化：$\Delta\mathcal{H}_{ij}^{\text{remove}} = +2J_{ij} + 2\gamma\sum_{k} s_{jk}s_{ki} + 2h$
- 局部能量变化：$\Delta\mathcal{H}_{ij}^{\text{add}} = -2J_{ij} - 2\gamma\sum_{k} s_{jk}s_{ki} - 2h$
- 局部分子场：$h_i = \sum_{j} J_{ij} s_{ij}$

## 网络架构
1. TGNNEncoder：3层 GATConv，隐藏维度64，输出节点嵌入 Z
2. AttentionCoupling：从 Z 计算 J 矩阵（Attn 机制）
3. DismantlePolicyHead：输入 [z_i; h_i; deg(i)]，输出节点移除概率
4. ConstructPolicyHead：两阶段设计（物理Top-K预筛选 + 策略精炼），输出边添加概率
5. ValueHead：全局 readout + 图级特征，输出状态价值 V(s)

## 环境设计
- DismantleEnv：节点移除，奖励 = -Δσ + αΔH - c_step，终止 σ ≤ 0.1n
- ConstructEnv：边添加，奖励 = Δλ₂ - αΔH - c_step，终止 λ₂ ≥ θ_λ
- RewiringEnv：边交换，保持 |E| 不变，使用 Kawasaki ΔH_swap

## PPO 训练器
- 使用 GAE 计算优势
- 策略损失：clip + entropy bonus
- 值函数损失：MSE
- 联合对齐损失：每5轮计算拆解/构造嵌入对齐

## 关键实现细节
- λ₂ 使用 sparse eigsh 或 lobpcg
- 三角项通过 common_neighbors 缓存加速
- 动作掩码使用 torch.where(mask, logits, -1e9)
- 课程学习：n 从20逐步增加到100
- 奖励使用 RunningMeanStd 归一化

## 测试验证
1. 能量一致性测试（误差 < 1e-6）
2. PPO 收敛测试（星型图优先移除中心节点，AUC 提升 > 30%）
3. 重构守恒测试（边数严格不变）
4. 物理引导测试（h=+1 vs h=-1 动作分布差异）

## 输出要求
输出完整可运行的 Python 项目（多文件），包含所有模块、PPO 训练循环、以及 if __name__ == "__main__" 中的测试验证代码。所有类和方法必须带类型注解与 docstring。
```
