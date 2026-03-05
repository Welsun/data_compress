# 面向 AI 训练浮点数据集的压缩方案与框架（仅浮点）

## 1. 目标与约束

### 1.1 目标
- 在可接受精度损失前提下，实现**整体压缩比 >= 10x**。
- 降低训练数据存储与传输成本，并维持训练吞吐。
- 支持时序浮点与通用浮点两类主场景。
- 提供可回滚、可审计、可观测的工程化压缩流程。

### 1.2 约束
- 可以接受有损压缩，但必须受误差上限约束。
- 压缩方案以训练任务指标为准，不以单点压缩比为唯一目标。
- 分布式训练下，索引与元数据必须支持并发读取。

---

## 2. 架构总览（仅浮点）

沿用三层架构，但所有策略都围绕浮点数据：

1. **样本层（Sample Layer）**：对单样本浮点数组/张量进行预测、量化、编码
2. **分片层（Shard Layer）**：按 block 聚合压缩，支持按块自适应策略
3. **索引层（Index Layer）**：记录定位信息与质量元数据，支持回滚

---

## 3. 样本层：浮点压缩策略

## 3.1 时序型浮点（传感器、日志序列、行情序列）
- 预测编码：`Delta` / `Delta-of-Delta`
- 窗口分块：如 1k~16k 点/块，便于随机采样和并行解压
- 受控有损：
  - 绝对误差：`|x - x'| <= eps_abs`
  - 相对误差：`|x - x'| / max(|x|, 1e-12) <= eps_rel`
- 推荐链路：`预测残差 -> 整数映射(VarInt/定宽) -> zstd`

## 3.2 通用型浮点（Embedding、特征矩阵、激活快照）
- 按张量块压缩：按行/列/chunk 切块
- 量化或重编码：
  - `FP32 -> FP16/BF16`
  - `INT8/INT16` 定点量化（per-tensor/per-channel/per-block）
- 位打包：`bitpack`
- 推荐链路：`量化/重编码 -> bitpack -> zstd`

## 3.3 业界先进算法可选项（优先级）
1. **SZ / SZ3**：误差控制强，适合时序与科学浮点
2. **ZFP**：固定比特率/误差模式，适合块状浮点
3. **FP16/BF16 + bitpack**：工程简单，吞吐友好
4. **INT8/INT16 + scale**：高压缩比潜力，需严格做精度回归

## 3.4 字段级策略约束
- 每个字段必须声明：`loss_mode=bounded_lossy`
- 每个字段必须记录：`eps_abs`、`eps_rel`、`quant_bits`、`codec_family`
- 关键特征字段可配置“保守模式”（更小误差或无损）

---

## 4. 分片层：按块自适应压缩

### 4.1 分片与 block
- shard 建议：`256MB ~ 1GB`
- block 建议：`1MB ~ 4MB`

### 4.2 按块自动选策略
对每个 block 做快速 profiling（方差、稀疏度、动态范围）：
- 平滑时序块：优先 `SZ/ZFP`
- 高动态块：优先 `FP16/BF16 + zstd`
- 低敏感块：尝试 `INT8 + bitpack + zstd`

### 4.3 训练友好能力
- 预取 + 异步解压 + ring buffer
- worker 并行拉取下一 shard
- 本地 NVMe 解压缓存

---

## 5. 索引层：定位 + 质量追踪

### 5.1 定位索引
- Dataset Manifest：版本、shard 列表、统计信息
- Shard Index：block 偏移与边界
- Sample Index：`sample_id -> shard/block/offset/length`

### 5.2 质量元数据（必须）
- `codec_id`
- `codec_params_digest`
- `error_stat`（MAE/RMSE/MaxAE/P99AE）
- `task_guardrail`（如 `delta_top1 <= 0.2%`）

### 5.3 一致性与回滚
- 不可变 manifest 版本（v1/v2/v3）
- 原子发布（先数据后指针）
- 训练异常时快速回滚到上一版本

---

## 6. 10x 压缩比落地路径

## 6.1 目标门槛
- 压缩比：`>= 10x`
- 质量：主任务指标下降不超过阈值
- 吞吐：训练吞吐下降不超过 5%

## 6.2 推荐组合
- 路径 A（时序主路径）：`SZ/ZFP + zstd`
- 路径 B（通用主路径）：`FP16/BF16 或 INT8 + bitpack + zstd`
- 路径 C（混合）：字段级策略（高敏字段保守、低敏字段激进）

## 6.3 最小 A/B 实验矩阵
- Baseline：`delta + zstd`
- A：`sz(eps_rel=1e-3) + zstd`
- B：`zfp(fixed_rate) + zstd`
- C：`fp16 + bitpack + zstd`
- D：`int8(per-channel) + bitpack + zstd`

每个方案记录：
- 压缩比
- 解压吞吐（GB/s）
- 训练吞吐（samples/s）
- 任务指标差值

---

## 7. 端到端流程（浮点专用）

1. Ingest：读取原始浮点数据并校验
2. Normalize：统一 dtype、shape、缺失值策略
3. Profile：统计每字段分布与敏感度
4. Pack：按 block 自适应压缩
5. Index：生成定位与质量索引
6. Validate：离线误差检查 + 在线训练 smoke test
7. Publish：版本化发布
8. Monitor：持续监控压缩比、吞吐与训练指标

---

## 8. MVP 迭代建议

### 阶段 1（2~3 周）
- 上线时序浮点路径：`Delta/SZ + zstd`
- 跑通索引、版本化与回滚
- 拿到首个 10x 子集结果

### 阶段 2（3~4 周）
- 上线通用浮点路径：`FP16/BF16/INT8 + bitpack + zstd`
- 上线按块自适应策略
- 完成主要任务 A/B 回归

### 阶段 3（持续）
- 自动调参（按字段分布自动选 codec/误差阈值）
- 完善告警（误差、吞吐、训练指标联动）

---

## 9. 下一步建议

1. 抽取 200GB 代表性浮点子集。
2. 并行试跑 `SZ/ZFP/FP16/INT8` 四条路线。
3. 选前两名进入真实训练 A/B。
4. 达到“10x + 指标约束 + 吞吐约束”后再全量推广。
