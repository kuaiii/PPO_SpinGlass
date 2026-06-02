# PPO_SpinGlass 算法实现详解与效果诊断

> 本文档基于项目源码逐行解析，旨在揭示模型实现细节、训练配置与测试表现之间的关联，定位效果不佳的根因。

---

## 1. 算法总览

PPO_SpinGlass 将**网络拆解（Network Dismantling）**问题建模为基于**自旋玻璃能量景观**的马尔可夫决策过程（MDP），使用 **PPO（Proximal Policy Optimization）** 求解。核心思路是：

- 将图的拓扑状态编码为自旋组态矩阵 $s \in \{+1, -1\}^{n \times n}$；
- 用 TGNN（3 层 GATConv）学习节点嵌入 $Z$，并通过注意力机制生成耦合矩阵 $J$；
- 策略网络在能量景观中搜索最优拆解轨迹，每一步移除一个节点，直到最大连通分量（LCC）低于阈值。

**当前 checkpoint 的训练配置（来自 `train_real_data.log` 与 `checkpoint_dismantle.pt`）：**

| 配置项 | 实际值 |
|--------|--------|
| 任务 | dismantle |
| 训练迭代数 | 300 iterations |
| 课程学习起点 | n_nodes = 20 |
| 课程学习终点 | n_nodes = 40（iteration 200 时升至 40，此后维持） |
| 每轮 episode 数 | 20 |
| PPO batch_size | 64 |
| PPO K_epochs | 2 |
| 学习率（统一） | 3e-4（策略与值函数共用 Adam） |
| GAT 隐藏维度 | 64，4 头，3 层 |
| 训练数据来源 | **无真实数据**，回退到随机生成 BA/WS（n≤40） |

---

## 2. 自旋玻璃物理框架

### 2.1 自旋组态编码

对任意图 $G=(V,E)$，定义完全图 $K_n$ 上的自旋变量：

$$
s_{ij} = 2A_{ij} - 1 \in \{+1, -1\}
$$

- $s_{ij}=+1$：边 $(i,j)$ 存在；
- $s_{ij}=-1$：边 $(i,j)$ 不存在。

**代码实现**（`utils/spin_glass.py:273-297`）：

```python
def adjacency_to_spin_dense(G: nx.Graph, n_total: Optional[int] = None) -> torch.Tensor:
    if n_total is None:
        n_total = G.number_of_nodes()
    adj = np.zeros((n_total, n_total), dtype=np.float32)
    for u, v in G.edges():
        adj[u_int, v_int] = 1.0
        adj[v_int, u_int] = 1.0
    s = torch.from_numpy(spin_config(adj))  # 2*adj - 1
    return s
```

**注意**：在拆解环境中，节点被移除后，对应的行/列被设为 $-1$（`topology_env.py:107-108`），表示该节点与所有其他节点的连接均为“不存在”。

### 2.2 哈密顿量（状态能量）

$$
\mathcal{H}(G) = -\sum_{(i,j)} J_{ij} s_{ij} - \gamma \sum_{(i,j,k) \in \Delta} s_{ij}s_{jk}s_{ki} - h \sum_{(i,j)} s_{ij}
$$

- $\gamma = 0.5$：三角闭合系数；
- $h = -1.0$：拆解任务的外部场（构造任务为 $+1.0$）。

**代码实现**（`utils/spin_glass.py:29-76`）：

```python
def hamiltonian(J, s, gamma, h_field, triangles):
    n = J.shape[0]
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=J.device)
    term1 = -torch.sum(J[row_idx, col_idx] * s[row_idx, col_idx])
    term2 = torch.tensor(0.0, ...)
    if triangles and gamma != 0.0:
        vals = []
        for i, j, k in triangles:
            vals.append(s[i, j] * s[j, k] * s[k, i])
        term2 = -2.0 * gamma * torch.stack(vals).sum()
    term3 = -h_field * torch.sum(s[row_idx, col_idx])
    return term1 + term2 + term3
```

**关键细节**：三角项使用了系数 $-2\gamma$ 而非 $-\gamma$，作者注释说明这是为了与单条边增删时的能量变化公式严格匹配（见 `delta_remove`/`delta_add`）。

### 2.3 局部分子场

节点 $i$ 的局部分子场定义为：

$$
h_i = \sum_{j \in \mathcal{N}(i)} J_{ij} s_{ij}
$$

**代码实现**（`utils/spin_glass.py:200-225`）：

```python
def local_field(J, s, node, neighbors=None):
    if neighbors is None:
        neighbors = list(range(J.shape[0]))
    idx = torch.tensor(neighbors, dtype=torch.long, device=J.device)
    return torch.sum(J[node, idx] * s[node, idx])
```

在拆解环境中，`get_local_fields` 仅对**存活节点**计算局部分子场（`topology_env.py:240-259`），已移除节点位置为 0。

---

## 3. 网络架构详解

整体架构由 4 个模块组成：

```
Observation (PyG Data)
    ↓
TGNNEncoder  →  Z (n, hidden_dim)
    ↓
AttentionCoupling  →  J (n, n)
    ↓
DismantlePolicyHead  →  logits (n,)  →  Categorical 采样动作
    ↑
局部分子场 h_local + 度数 deg

ValueHead  →  V(s) 标量
```

### 3.1 TGNNEncoder（3 层 GATConv）

**文件**：`models/encoder.py`

**输入**：PyG `Data(x, edge_index)`，其中 `x` 默认为节点度数（`utils/graph_metrics.py:203-206`）。

**架构**：
- 3 层 `GATConv`，隐藏维度 64；
- 前两层使用 4 头注意力，输出拼接后维度为 $64 \times 4 = 256$；
- 最后一层使用 1 头，输出维度 64；
- 每层后接 BatchNorm + ReLU + Dropout(0.1)。

**关键代码**：

```python
for i in range(num_layers):
    in_ch = in_channels if i == 0 else hidden_dim * num_heads
    out_heads = 1 if i == num_layers - 1 else num_heads
    concat = i != num_layers - 1
    self.convs.append(GATConv(in_channels=in_ch, out_channels=hidden_dim,
                               heads=out_heads, concat=concat, dropout=dropout,
                               add_self_loops=False))
```

**潜在问题**：
- **输入特征过于简单**：默认仅用节点度数作为特征（`nx_to_pyg_data` 中 `node_features=None` 时回退到 degree）。对于**随机正则图（RA）**，所有节点度数相同，GAT 在第一层就无法区分节点，导致嵌入同质化。
- 没有使用边特征或位置编码。

### 3.2 AttentionCoupling（耦合矩阵 $J$）

**文件**：`models/coupling.py`

**计算流程**：
1. $Q = Z W_Q$，$K = Z W_K$
2. scores $= QK^\top / \sqrt{d_k}$
3. 对角线 mask 为 $-\infty$（禁止自耦合）
4. Top-K 稀疏化（n > 50 时只保留每行前 50 个最大 score）
5. 按行 softmax 得到 $J$

**关键代码**：

```python
Q = self.W_Q(Z)  # (n, hidden_dim)
K = self.W_K(Z)  # (n, hidden_dim)
scores = torch.matmul(Q, K.t()) / math.sqrt(self.d_k)
mask = torch.eye(n, device=Z.device, dtype=torch.bool)
scores = scores.masked_fill(mask, float('-inf'))

# Top-K 稀疏化
if self.top_k is not None and self.top_k < n - 1:
    vals, indices = torch.topk(scores, k=min(self.top_k, n - 1), dim=-1)
    sparse_mask = torch.full_like(scores, float('-inf'))
    sparse_mask.scatter_(-1, indices, 0.0)
    scores = scores + sparse_mask

J = torch.softmax(scores, dim=-1)
```

**潜在问题**：
- **Softmax 的规模敏感性**：$J$ 是对 score 做**按行 softmax** 得到的。当 $n$ 从 40（训练）扩展到 500（测试）时，softmax 的分布特性发生显著变化。训练时模型学习的是在 40 个节点上的注意力模式，迁移到 500 节点时，归一化后的 $J_{ij}$ 数值范围与训练分布不一致。
- $J$ 被约束为行概率分布（非负、行和为 1），这与物理上的耦合强度 $J_{ij}$（可正可负）有本质区别。作者将其解释为“注意力权重”，但在能量公式中 $J_{ij}$ 是作为耦合强度出现的，软最大化约束可能限制了其表达能力。

### 3.3 DismantlePolicyHead（拆解策略头）

**文件**：`models/policy.py:14-119`

**输入特征拼接**：

```python
features = [Z, h_local.unsqueeze(-1), deg.unsqueeze(-1)]
# dim = hidden_dim + 1 + 1 = 66
```

**网络**：2 层 MLP（66 → 128 → 1），输出节点 logits。

**动作掩码**：已移除节点 logits 设为 $-10^9$。

**采样**：Categorical 分布采样节点。

**潜在问题**：
- **特征利用不充分**：策略头仅使用了节点嵌入 $Z$、局部分子场 $h$ 和度数 $deg$，没有显式利用全局图特征（如 LCC 大小、当前步数、图密度等）。
- 对于 RA 图，deg 为常数，h 的区分度有限，策略主要依赖 Z。而 Z 又因输入特征同质化（全为 degree）而缺乏区分度。

### 3.4 ValueHead（值函数头）

**文件**：`models/value.py`

**设计**：
- 全局 readout：`Z.mean(dim=0)`（单图）或 `global_mean_pool(Z, batch)`（多图）；
- 拼接图级特征（边数、LCC、$\lambda_2$、$\mathcal{H}$），维度为 $64 + 4 = 68$；
- 2 层 MLP → 标量 $V(s)$。

**关键缺陷**：在训练与推理代码中，`graph_features` 参数**始终为 None**（`train_ppo.py:289` 中 `v = self.value_head(Z, obs_data)`，未传入 graph_features）。因此 ValueHead 实际使用的图级特征全部为**零向量**，值函数完全没有利用 LCC、能量等全局状态信息，仅依赖节点嵌入的平均池化。

```python
# train_ppo.py:289
v = self.value_head(Z, obs_data)  # graph_features 默认为 None

# value.py:65-70
if graph_features is None:
    graph_features = torch.zeros(1, self.num_graph_features, ...)
```

这导致值函数对状态的估计能力严重不足，进而影响 GAE 优势计算的准确性。

---

## 4. MDP 环境设计

### 4.1 DismantleEnv

**文件**：`envs/topology_env.py:33-270`

**状态空间**：当前图 $G_t$（以 PyG Data 表示）。

**动作空间**：选择存活节点 $v \in V_t^{\text{alive}}$ 移除（级联删除其关联边）。

**状态转移**：
```python
self.G.remove_node(node_idx)
self.removed_nodes.add(node_idx)
```

**奖励函数**（`topology_env.py:210-213`）：

```python
r_sigma = (sigma_old - sigma_new)          # LCC 下降量
r_H = self.alpha * (self.H - H_old)        # 能量变化量，alpha=0.1
reward = r_sigma + r_H - self.step_cost    # step_cost=0.01
```

- 成功终止（$\sigma \le 0.1$）：额外 +10；
- 失败终止（步数耗尽）：无额外奖励。

**关键问题诊断**：

1. **奖励未直接优化拆解代价**：奖励中的 $r_\sigma$ 只反映 LCC 的**绝对下降量**，而非拆解效率（如 rem_ratio）。例如，在 n=500 时，移除一个 leaf 节点可能只让 LCC 下降 1/500=0.002，奖励极小；而移除 hub 节点可能让 LCC 下降 0.1，奖励很大。但模型并未被显式教导“用最少节点达到阈值”，而是被教导“每步让 LCC 下降最多”。这两者在局部可能冲突——有时移除一个非 hub 节点能为后续创造更好的拆解条件（如 CI 算法中的高阶邻居效应），但 PPO 无法从单步奖励中感知这种长期收益。

2. **能量项权重过低**：$\alpha = 0.1$，且能量变化 $\Delta H$ 的绝对值通常很小（因为 $J$ 是 softmax 输出，$s_{ij} \in \{-1,+1\}$），导致 $r_H$ 对总奖励的贡献微乎其微。物理引导作用几乎被淹没在 $r_\sigma$ 中。

3. **成功奖励稀疏**：仅在达到 LCC 阈值时获得 +10，中间过程无阶段性反馈。对于 n=500 的图，需要约 50~200 步才能获得一次成功信号，信用分配困难。

### 4.2 观测构造

`nx_to_pyg_data`（`utils/graph_metrics.py:185-217`）将 NetworkX 图转为 PyG Data：

```python
if node_features is None:
    degrees = np.zeros((n_total, 1), dtype=np.float32)
    for i in range(n_total):
        degrees[i, 0] = float(G.degree(i)) if G.has_node(i) else 0.0
    node_features = degrees
```

**问题**：节点特征只有 degree，无其他拓扑特征（如 betweenness、coreness、聚类系数）。这使得 Encoder 在 RA 图上完全失效（所有节点 degree 相同）。

---

## 5. PPO 训练流程

### 5.1 数据收集

**文件**：`train_ppo.py:251-389`

每轮迭代收集 `num_episodes=20` 条轨迹。对于拆解任务：

```python
for _ in range(num_episodes):
    G = self.generate_graph(self.n_nodes)  # n=20~40
    env = DismantleEnv()
    obs = env.reset(G)
    while not done:
        Z = encoder(obs_data)
        J = coupling(Z)
        mask = env.get_alive_mask(device=self.device)
        h_local = env.get_local_fields(J)
        degs = env.get_degrees().to(self.device)
        action, log_prob = self.policy_head.sample(Z, h_local, degs, mask)
        v = self.value_head(Z, obs_data)
        obs_next, reward, done, info = env.step(action.item())
```

**注意**：每步调用 `env.get_local_fields(J)` 时，J 是当前图状态重新计算的耦合矩阵，而非固定不变的物理参数。这与标准自旋玻璃模型中 $J$ 为固定外部参数不同——这里的 $J$ 是策略网络的输出，随图结构动态变化。

### 5.2 GAE 与 PPO 更新

**GAE 计算**（`train_ppo.py:33-66`）：

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

**奖励归一化**：使用 RunningMeanStd 对奖励做 Z-score 归一化。

**PPO 损失**（`train_ppo.py:478-484`）：

```python
ratio = torch.exp(log_prob - old_log_probs[idx])
surr1 = ratio * adv
surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
policy_loss = -torch.min(surr1, surr2)
value_loss = F.mse_loss(v.squeeze(), ret)
loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

**关键参数**：
- `eps_clip = 0.2`
- `value_loss_coef = 0.5`
- `entropy_coef = 0.01`
- `K_epochs = 2`

**课程学习**：每 100 iteration 增加 10 个节点，从 20 → 30 → 40，此后维持 40。300 iterations 中仅完成两次增量。

### 5.3 对齐损失（Align Loss）

每 5 轮迭代执行一次：

```python
loss_align = F.mse_loss(Z1.mean(dim=0), Z2.mean(dim=0))
```

**问题**：对齐损失要求两张不同图的节点嵌入均值相同，这在物理上缺乏明确动机。两张不同结构的图（BA vs WS）的节点嵌入分布本就不应相同，强制对齐可能损害 Encoder 的表达能力。

---

## 6. 推理流程

**文件**：`compare_dismantling.py:54-89`、`benchmark_synth.py:77-97`

推理时，模型以贪心模式（`sample` 而非 argmax）逐节点移除：

```python
while not done:
    obs_data = obs.to(device)
    with torch.no_grad():
        Z = encoder(obs_data)
        J = coupling(Z)
        mask = env.get_alive_mask(device=device)
        h_local = env.get_local_fields(J)
        degs = env.get_degrees().to(device)
        action, _ = policy_head.sample(Z, h_local, degs, mask)
        node = action.item()
    obs_next, reward, done, info = env.step(node)
```

**注意**：虽然用了 `sample()`（随机采样），但在评估拆解效果时，通常应该使用 `forward()` + argmax 做确定性推理。`sample()` 引入了随机性，可能导致同一图多次评估结果不同，且 rem_num 不稳定。

---

## 7. 效果不佳的根因分析

综合代码实现与测试结果，PPO_SpinGlass 在 BA/WS/RA 网络上表现不及 CI、EI_s1、FINDER 等方法的根因可归结为以下六点：

### 7.1 训练-测试分布严重不匹配（最严重）

| 维度 | 训练分布 | 测试分布 | 差距 |
|------|----------|----------|------|
| 图规模 | 20~40 节点 | 50~500 节点 | **10~25 倍** |
| 图类型 | BA、WS（随机生成） | BA、WS、RA | RA 完全 OOD |
| 密度 | 小图稀疏/稠密可变 | 固定参数生成 | 结构差异大 |

- **Encoder 的 OOD 问题**：GATConv 中的 BatchNorm 在 n=40 上学习的均值/方差，与 n=500 上的节点嵌入分布完全不同。AttentionCoupling 的 softmax 归一化也对 n 极其敏感。
- **ValueHead 失效**：值函数仅在 20~40 节点上训练，无法估计 500 节点状态的价值，GAE 优势计算失真。

**建议**：必须训练到 n=100 以上，或使用图大小不变性设计（如 GraphNorm 替代 BatchNorm、相对位置编码）。

### 7.2 节点特征过于贫乏

当前节点特征仅为 `degree`（`utils/graph_metrics.py:203-206`）。

- **BA 图**：degree 有一定区分度（幂律分布），GAT 能区分 hub 和 leaf；
- **WS 图**：degree 近似常数（k=4），区分度低；
- **RA 图**：degree 严格常数（d=4），**所有节点特征完全相同**，Encoder 输出几乎一致。

这直接解释了为什么 PPO 在 RA 图上表现最差（rem_ratio=0.42，而 CI 系列约 0.37）。

**建议**：增加节点特征，如 betweenness、closeness、coreness、eigenvector centrality、聚类系数，或使用恒等矩阵 + 可学习嵌入替代 degree。

### 7.3 ValueHead 未使用图级特征

如 3.4 节所述，`graph_features` 参数在训练和推理中始终为 None，图级特征（LCC、能量、代数连通度等）被零向量替代。值函数 $V(s)$ 仅依赖节点嵌入的均值池化，缺乏对全局拓扑状态的感知。

**后果**：
- 值函数无法区分“LCC 已经很小”和“LCC 仍然很大”的状态；
- GAE 优势估计不准确，策略梯度方向偏差大；
- PPO 的 clip 机制进一步放大了值函数误差的影响。

**建议**：在 `train_ppo.py` 和 `eval.py` 中显式计算并传入图级特征：

```python
graph_features = torch.tensor([G.number_of_edges(), env.sigma, lambda2, env.H])
v = value_head(Z, obs_data, graph_features)
```

### 7.4 奖励函数设计缺陷

当前奖励（`topology_env.py:210-213`）：

```python
reward = (sigma_old - sigma_new) + 0.1 * (H - H_old) - 0.01
```

**问题 1：未显式优化拆解代价**
目标是最小化 `rem_num`（达到 LCC 阈值所需节点数），但奖励只关心单步 LCC 下降量。这导致策略倾向于“每步拆除最大的 hub”，而忽视了有时需要优先拆除“桥梁”节点以分裂网络。CI 算法的核心就是考虑高阶邻居效应（$L$-hop），而 PPO 仅从单步奖励中无法学到这种长程依赖。

**问题 2：能量项权重过低**
$\alpha=0.1$ 且 $\Delta H$ 绝对值通常 $< 1$，贡献 $< 0.1$；相比之下 $r_\sigma$ 可达 $0.1 \sim 0.5$。自旋玻璃能量景观对策略的引导作用几乎为零。

**问题 3：成功奖励过于稀疏**
仅在达到阈值时 +10，无中间里程碑。对于 500 节点图，约需 100+ 步才能获得一次成功信号，信用分配困难。

**建议**：
- 引入稠密奖励：$r_t = -(\sigma_{t+1} - \sigma_t) / \sigma_t - \beta \cdot (1 / n)$，或直接使用 `rem_ratio` 作为 episode 级回报；
- 增加阶段奖励：如 LCC 下降到 50%、20%、10% 时分别给予奖励；
- 提升能量权重或设计能量约束（如要求策略沿能量梯度方向行动）。

### 7.5 推理策略的随机性

评估代码使用了 `policy_head.sample()`（`compare_dismantling.py:79`、`benchmark_synth.py:90`），即**随机采样**而非贪心 argmax。

```python
action, _ = policy_head.sample(Z, h_local, degs, mask)
```

这导致：
- 同一图多次运行结果不同，rem_num 不稳定；
- 评估时无法获得策略的“最佳表现”，与 baseline 方法的确定性输出不公平对比。

**建议**：评估时应改用确定性策略：

```python
logits = policy_head.forward(Z, h_local, degs, mask)
action = torch.argmax(logits)
```

### 7.6 耦合矩阵 J 的物理语义偏差

标准自旋玻璃中 $J_{ij}$ 是固定的外部耦合参数（可正可负）。而本实现中 $J$ 是注意力权重的 softmax 输出（非负、行和为 1），这导致：

- $J_{ij}$ 无法表示“反铁磁耦合”（负耦合）；
- 行和为 1 的约束使得 $n$ 变化时，每个 $J_{ij}$ 的期望值为 $1/(n-1)$，随规模增大而减小，与训练时的分布不同；
- 能量景观的“势阱”深度受限于 softmax 的数值范围，难以形成强烈的能量梯度引导策略。

**建议**：
- 移除 softmax，改用 tanh 或线性输出 + LayerNorm；
- 将 $J$ 设计为稀疏可学习的物理参数，而非注意力权重。

---

## 8. 总结与改进路线图

| 优先级 | 问题 | 改进措施 |
|:------:|------|----------|
| **P0** | 训练规模太小（最大 n=40） | 延长训练至 n=100+，或引入 Zero-Shot 规模泛化技术（GraphNorm、大小无关的 Pooling） |
| **P0** | 节点特征仅 degree | 增加 centrality 特征（betweenness、coreness、eigenvector）或使用可学习的位置编码 |
| **P0** | ValueHead 图级特征未使用 | 传入实际的 LCC、能量、边数等图级特征 |
| **P1** | 奖励未优化拆解代价 | 设计稠密奖励，引入 rem_ratio 作为 episode 回报，增加里程碑奖励 |
| **P1** | 推理使用随机采样 | 评估时改用 `argmax` 确定性策略 |
| **P1** | 未在 RA 图上训练 | 增加 RA/ER 等训练图类型，提升泛化性 |
| **P2** | 对齐损失无明确物理意义 | 移除或替换为更有意义的正则化（如能量平滑性约束） |
| **P2** | J 的 softmax 约束 | 尝试移除 softmax，使用无约束输出 |

---

> **文档版本**：基于项目源码 v1.0 解析生成
> **生成时间**：2026-06-02
