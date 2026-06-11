# ConnectX 强化学习大作业

强化学习课程大作业「选题一：Connect X」。目标：在 Kaggle ConnectX 上提交智能体拿分，
并探索/对比多种 RL 算法（通过本地对战评估，不依赖 Kaggle）。

## 算法与结果总览

四种算法 + 一个共享的战术安全层（tactical safety：先抢自己一步必胜点，再堵对手一步必胜点）。


| 算法                    | 棋盘                 | 对 negamax 胜率（最终/峰值）           | 备注                                         |
| --------------------- | ------------------ | ----------------------------- | ------------------------------------------ |
| **AlphaZero**（主力）     | 6x7 connect-4      | **0.90 / 0.93**（overnight 版本） | 残差策略-价值网络 + PUCT MCTS + self-play + gating |
| MaskablePPO self-play | 6x7 connect-4      | 0.73 / 0.95                   | sb3-contrib，对手池联盟自我对弈                      |
| DQN                   | 6x7 connect-4      | 0.60 / –                      | 卷积 Q 网络 + target network + replay          |
| 表格 Q-learning         | 4x5 connect-3（小棋盘） | 1.00 / –                      | 标准棋盘状态空间太大，作为基础 baseline                   |


本地 arena 互打（100 局，轮流先手）：**AlphaZero > DQN > PPO**（AlphaZero 对两者胜率 100%）。
详细数据见 `results/`（学习曲线 png、`negamax_final_comparison.csv`、`win_rate_matrix.txt` 等）。

最终提交选择：`submission/submission.py`（= AlphaZero 版，见 `submission/submission_kind.txt`）。

## 目录结构

```
connectx/
  envs/connectx_env.py        # 棋盘逻辑 + Gymnasium 环境（与 Kaggle 语义一致）
  agents/
    lookahead.py              # 战术安全层（一步必胜/必堵）
    q_learning.py             # 表格 Q-learning（negamax 风格 TD 更新）
    dqn.py                    # DQN（self-play + action mask）
    ppo_selfplay.py           # PPO 用的对手池 + 单智能体环境包装
    alphazero/                # 网络 / MCTS / self-play / replay buffer / 推理 agent
  training/                   # 各算法训练入口 + finalize_results.py（汇总出图）
  evaluation/                 # arena 互打评估 + 对 Kaggle negamax 评估
  submission/                 # 生成单文件 Kaggle submission（纯 NumPy 推理）
scripts/
  train_sequential.sh         # 完整训练流水线（断点续跑安全）
  train_overnight.sh          # 过夜加强训练（PPO 续训 + 更强 AlphaZero）
tests/                        # 单元测试（pytest）
runs/                         # 训练产物 checkpoint（不进 git）
results/                      # 评估结果与图表
submission/                   # 已生成的各算法 submission_*.py
```

## 环境安装

```bash
pip install -e ".[all]"      # 全部依赖（训练服务器用）
pip install -e ".[dev]"      # 只跑测试的最小依赖
```

## 快速开始

```bash
# 单元测试
python -m pytest tests

# 训练（单独跑某个算法）
python -m connectx.training.train_q_learning --run-dir runs/q_learning
python -m connectx.training.train_dqn --run-dir runs/dqn --device cuda
python -m connectx.training.train_ppo --run-dir runs/ppo --total-timesteps 500000 \
    --checkpoint-freq 50000 --add-checkpoints-to-pool
python -m connectx.training.train_alphazero --run-dir runs/alphazero --device cuda \
    --generations 30 --selfplay-games 40 --mcts-simulations 100

# 或者一键全流程（含汇总出图 + 生成 submission）
bash scripts/train_sequential.sh

# 汇总结果 / 重新生成图表与 arena 矩阵
python -m connectx.training.finalize_results --results-dir results

# 生成 Kaggle submission（全部四个算法各一份 + 默认 submission.py）
python -m connectx.submission.make_all_submissions --output-dir submission --validate
```

## 已完成 ✅

- [x] ConnectX 环境（与 Kaggle 兼容）+ 单元测试
- [x] 四种算法实现与训练：表格 Q-learning、DQN、MaskablePPO self-play、AlphaZero
- [x] 战术安全层（消除一步漏招的低级失误）
- [x] 评估体系：算法间 arena 互打（轮流先手）+ 对 Kaggle negamax 基准
- [x] 全部训练已跑完，学习曲线 / 对比图 / 胜率矩阵在 `results/`
- [x] 单文件 Kaggle submission 生成与本地验证（4 个算法各一份）

## 还需完成 ⬜

- [ ] **上传 Kaggle**：把 `submission/submission.py` 提交到 Kaggle ConnectX 比赛，记录线上分数与排名，目前（alpha-zero:700分)
- [ ] **报告/展示**：算法对比分析写成报告——为什么 AlphaZero 最强；
- [ ] **奖励设计实验**（作业建议项）：目前只有终局 ±1 奖励，可尝试 reward shaping（我正在做）
  ```
  （如按连子数 / 威胁数给中间奖励）并对比效果
  ```
- [ ] **超参数调优**（作业建议项）：学习率、MCTS 模拟次数、c_puct 等的消融实验
- [ ] 继续训练：PPO 曲线 150 万步时仍在上升；AlphaZero 可加大 simulations/generations 再提一档（正在做）