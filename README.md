# DRL-MTUCS: 多任务应急感知无人机群感系统

> 基于层级多智能体深度强化学习的多任务无人机群感算法复现

## 📄 论文信息

**论文**: *Multi-Task-Oriented Emergency-Aware UAV Crowdsensing: A Hierarchical Multi-Agent Deep Reinforcement Learning Approach*

**作者**: Chen Fang, Chi Harold Liu, Hao Wang, Guangpeng Qi, Zhongyi Liu, Dapeng Wu

**期刊**: IEEE Journal on Selected Areas in Communications (JSAC), Vol. 44, 2026

---

## 🏗️ 算法核心思想

### 问题场景

在城市交通监控场景中，无人机(UAV)需要同时处理两类任务：
- **监控任务 (Surveillance)**：在固定热点（公交站牌、路口计算单元）定期采集数据，AoI阈值较宽松（如35个时隙）
- **应急任务 (Emergency)**：突发交通事故等，需要立即响应，AoI阈值严格（如20个时隙）

核心挑战：如何调度有限的无人机，在两类任务之间取得平衡，最大化"有效任务处理指标"。

### DRL-MTUCS 架构（三层结构）

```
┌─────────────────────────────────────────────────────┐
│              高层分配器 (High-Level Allocator)         │
│  输入: 全局状态（紧急PoI位置、UAV位置/能量）           │
│  输出: 将新紧急PoI分配给哪个UAV                       │
└──────────────────────┬──────────────────────────────┘
                       │ 分配目标
┌──────────────────────▼──────────────────────────────┐
│         动态加权队列 (Dynamically Weighted Queue)     │
│  每个UAV维护一个紧急PoI队列                           │
│  时序预测器估计到达各PoI的期望时隙 → 确定优先级        │
└──────────────────────┬──────────────────────────────┘
                       │ 最高优先级PoI → 当前目标
┌──────────────────────▼──────────────────────────────┐
│          底层执行器 (Low-Level UAV Execution)         │
│  输入: 本地观测(AoI热力图) + 目标PoI特征              │
│  输出: 连续动作 (dx, dy, speed)                       │
│  奖励: 环境奖励 + 自平衡内在奖励                      │
└─────────────────────────────────────────────────────┘
```

### 三大核心创新

#### 1. 有效任务处理指标 (Valid Task Handling Index)
$$I = \frac{\min_{m \in M} I_m}{\eta}$$

- $I_m$：任务m的有效处理率（AoI在阈值内的时间占比）
- $\eta$：能量消耗率
- **关键**：取各任务中最低的处理率，体现"木桶效应"

#### 2. 动态加权队列
- 每个UAV维护长度为$l_{que}$的紧急PoI队列
- **时序预测器** $\psi$：估计从当前位置到达目标PoI的期望时隙数
- 队列按优先级排序，最高优先级PoI成为UAV当前目标
- 解决了紧急任务稀疏奖励、位置不可预测的问题

#### 3. 自平衡内在奖励 (Eq. 15)
$$r_{intr} = -(1-\omega) \cdot \hat{AoI}_g \cdot \hat{d}_g + \omega \cdot \hat{d}_{\bar{g}}$$

- **目标 $g$**：最高优先级的紧急PoI → 引导UAV前往
- **反目标 $\bar{g}$**：最近的其他UAV位置 → 避免重复工作
- **$\omega$ 超参数**：控制紧急/监控任务的权衡
  - $\omega \to 0$：专注紧急任务
  - $\omega \to 1$：平衡监控覆盖

---

## 📁 项目结构

```
uav-crowdsensing/
├── main.py                        # 主入口：训练/评估/对比/可视化
├── requirements.txt               # 依赖库
├── README.md                      # 本文档
├── env/
│   ├── __init__.py
│   └── uav_env.py                 # 无人机群感环境（系统模型）
├── agents/
│   ├── __init__.py
│   ├── drl_mtucs.py               # DRL-MTUCS主Agent（Algorithm 1）
│   ├── networks.py                # 神经网络架构（策略/价值/CNN）
│   ├── ppo_trainer.py             # PPO训练器（Eq.17-20, GAE）
│   ├── weighted_queue.py          # 动态加权队列（Section IV-A）
│   ├── intrinsic_reward.py        # 自平衡内在奖励（Eq.15）
│   ├── temporal_predictor.py      # 时序预测器（Eq.14, 16）
│   └── baselines.py               # 基线算法（Random/Greedy/mTSP）
└── results/                       # 输出目录（自动生成）
    ├── checkpoints/               # 模型检查点
    ├── training_curves.png        # 训练曲线
    ├── comparison.png             # 基线对比图
    ├── trajectories.png           # 轨迹可视化
    └── ablation_omega.png         # ω消融实验图
```

---

## 🚀 快速开始

### 安装依赖

```bash
cd uav-crowdsensing
pip install -r requirements.txt
```

### 依赖库说明

| 库 | 版本 | 用途 |
|---|---|---|
| `torch` | ≥2.0.0 | 神经网络、PPO训练 |
| `numpy` | ≥1.24.0 | 数值计算、环境状态 |
| `gymnasium` | ≥0.29.0 | 环境接口标准 |
| `matplotlib` | ≥3.7.0 | 可视化 |
| `scipy` | ≥1.10.0 | 科学计算辅助 |

### 训练

```bash
# 默认训练（4 UAV, 300监控PoI, 500 episodes）
python main.py --mode train

# 自定义参数
python main.py --mode train \
    --num_uavs 4 \
    --num_surv_pois 300 \
    --emer_interval 6 \
    --surv_aoi 35 \
    --emer_aoi 20 \
    --omega 0.7 \
    --episodes 1000
```

### 评估

```bash
python main.py --mode eval
```

### 基线对比

```bash
python main.py --mode compare
```

对比方法：
- **Random**：随机动作
- **Greedy**：贪心策略（最近/最高AoI的PoI）
- **mTSP**：多旅行商问题启发式算法

### 轨迹可视化

```bash
python main.py --mode visualize
```

### 消融实验

```bash
python main.py --mode ablation
```

测试不同 $\omega$ 值对紧急/监控任务权衡的影响。

---

## 📊 实验参数（论文 Table II & III）

| 参数 | 符号 | 默认值 | 说明 |
|---|---|---|---|
| UAV数量 | $U$ | 4 | |
| 工作区域 | - | 6km × 6km | |
| 时隙时长 | $\tau$ | 20秒 | |
| 最大时隙数 | $T$ | 120 | |
| 最大速度 | $v_{max}$ | 15 m/s | |
| 最大能量 | $E_{max}$ | 500,000 J | |
| 监控PoI数量 | - | 300 | |
| 监控AoI阈值 | $AoI_{surv}^{th}$ | 35时隙 | |
| 应急AoI阈值 | $AoI_{emer}^{th}$ | 20时隙 | |
| 紧急生成间隔 | $\Delta$ | 6时隙 | |
| 队列长度 | $l_{que}$ | 3 | 论文Table III最优 |
| 内在奖励权重 | $\omega$ | 0.7 | SF数据集最优 |
| 隐藏层维度 | - | 128 | 3层MLP |
| 学习率 | - | 5e-4 | Adam |
| PPO clip | $\epsilon$ | 0.2 | |
| GAE $\lambda$ | - | 0.95 | |

---

## 🧠 关键实现细节

### PPO训练损失函数

**策略损失** (Eq. 17):
$$L(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**价值损失** (Eq. 18):
$$L(\phi) = \mathbb{E}\left[\max\left(L_1(\phi), L_2(\phi)\right)\right]$$

其中 $L_2$ 使用裁剪的价值函数。

**总损失** (Eq. 19):
$$L_{high} = L(\theta) + c_{critic}L(\phi) + c_{entropy}H(\pi)$$

### 高层奖励 (Eq. 12)
$$r_{t',high} = c_{high} \sum_{t=t'}^{t'+AoI_{emer}^{th}} r_{t,low}^{u'=a_{t,high}}$$

即：从分配时刻到应急AoI阈值耗尽期间的累计底层奖励。

### 底层奖励 (Eq. 13)
$$r_{t,low} = r_{t,emer} + r_{t,surv} - r_{t,\epsilon}$$

- $r_{t,emer}$：成功处理+1，超时-1
- $r_{t,surv}$：AoI变化归一化
- $r_{t,\epsilon}$：低能量惩罚

---

## 📈 预期结果

根据论文，DRL-MTUCS在以下指标上优于所有基线：

| 指标 | DRL-MTUCS | 最优基线 | 提升 |
|---|---|---|---|
| $I$ (有效任务处理指标) | ~1.34 | ~0.90 (mTSP) | +49% |
| $I_{emer}$ (应急处理率) | ~0.95 | ~0.85 (mTSP) | +12% |
| $I_{surv}$ (监控处理率) | ~0.90 | ~0.95 (HAPPO) | 略低但平衡更优 |

---

## 🔬 论文与实现对应关系

| 论文章节 | 代码文件 | 关键类/函数 |
|---|---|---|
| III 系统模型 | `env/uav_env.py` | `UAVCrowdsensingEnv` |
| III-A 监控任务 | `env/uav_env.py` | `_handle_surv_pois()` |
| III-B 应急任务 | `env/uav_env.py` | `_handle_emer_pois()` |
| III-C AoI模型 | `env/uav_env.py` | `aoi_history`更新 |
| III-D 指标定义 | `env/uav_env.py` | `_get_info()` |
| IV-A 动态加权队列 | `agents/weighted_queue.py` | `DynamicallyWeightedQueue` |
| IV-A 时序预测器 | `agents/temporal_predictor.py` | `TemporalPredictor` |
| IV-B 自平衡奖励 | `agents/intrinsic_reward.py` | `IntrinsicRewardComputer` |
| IV-C 训练流程 | `agents/ppo_trainer.py` | `PPOTrainer` |
| IV-D 算法1 | `agents/drl_mtucs.py` | `DRLMTUCS.act/train` |
| 神经网络 | `agents/networks.py` | 所有Network类 |

---

## 📝 注意事项

1. **本实现为论文核心算法的忠实复现**，简化了部分通信模型（OFDMA、LoS/NLoS路径损耗）的物理层细节，聚焦于强化学习框架本身
2. 数据集使用随机生成的PoI位置（模拟300个监控热点），如需真实数据集可替换
3. 训练时间取决于硬件：CPU约2-4小时/500 episodes，GPU约30分钟
4. $\omega$ 的选择对性能影响显著，建议根据实际场景需求调整
