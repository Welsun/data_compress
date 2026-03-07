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

## 6. 代码实现状态（MVP）

当前仓库已实现一个 Python MVP（`src/data_compress`）：

- 样本层：
  - `delta_zlib`（时序）
  - `fp16_zlib`（通用）
  - `int8_zlib`（低敏感）
  - `delta_zstd` / `fp16_zstd` / `int8_zstd`（需安装可选依赖 `zstandard`）
  - `delta_sz` / `fp16_sz` / `int8_sz`（需安装可选依赖 `python-sz3`）
- 分片层：按 block profiling 自动选择 codec
- 索引层：manifest + shard/sample index + quality metadata
- 验证：基于 MAE / 相对误差门限的快速校验

> 说明：当前 MVP 默认使用纯 Python + `zlib`；同时支持可选 `zstd` 与 `SZ` 依赖以便快速验证多后端流程，后续可进一步替换为生产级 codec。

若要启用 zstd 变体 codec，请先安装：

```bash
pip install zstandard
```

若要启用 SZ 变体 codec，请先安装：

```bash
pip install python-sz3
```

若要启用 zstd 变体 codec，请先安装：

```bash
pip install zstandard
```

---

## 7. 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest
```

最小示例：

```python
import numpy as np
from data_compress import CompressionConfig, CompressionPipeline, FieldStrategy

samples = [np.sin(np.linspace(0, 20, 2048, dtype=np.float32) + i * 0.01) for i in range(4)]

config = CompressionConfig(
    strategies={
        "sensor": FieldStrategy(field_name="sensor", codec_family="delta_zlib", eps_abs=1e-3, eps_rel=1e-3)
    }
)

pipeline = CompressionPipeline(config)
result = pipeline.pack_field("sensor", samples)
print(result.manifest)
print("validate:", pipeline.validate(result, max_mae=0.01, max_rel=0.1))
```



命令行启动脚本：

```bash
python start_csv_compress.py train.csv
# 或使用模块方式
python -m data_compress.cli train.csv
```

可通过参数传入 CSV 路径并调整列范围：

```bash
python start_csv_compress.py /path/to/train.csv --start-col 2 --end-col 31
```


如需先统计压缩前浮点数据的实际内存占用，可使用：

```bash
python data/csv_to_bin.py /path/to/train.csv --start-col 2 --end-col 31 --dtype float32
```

脚本会提取指定列的浮点值，按连续二进制写出 `.bin`，并打印总字节数。
若首行是列名（非浮点），脚本会自动识别并跳过；也可通过 `--skip-header` 强制跳过首行。

CSV 输入（第 2~31 列共 30 维）示例：

```python
from data_compress import CompressionConfig, CompressionPipeline, FieldStrategy

config = CompressionConfig(
    strategies={
        "csv_features_2_31": FieldStrategy(field_name="csv_features_2_31", codec_family="fp16_zlib")
    }
)
pipeline = CompressionPipeline(config)

# 默认读取第 2~31 列（1-based, inclusive）
result = pipeline.pack_csv("train.csv")
print(result.manifest)
```

