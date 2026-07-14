# OPW 复现协议

更新日期：2026-07-13

本文记录仓库中唯一支持的 CAPA OPW 评测协议，以及可重复执行的命令。OPW 必须由代码计算并写入 `summary.json`，不得手工填写均值。

## 1. 固定定义

- 实现入口：`capa/utils/metric.py::compute_opw()`
- 协议：`capa_strict`，代码不再提供其他 mode
- RGB 可见性权重：论文给定的 `beta=50`
- 评测区域：稠密 GT 中有限且大于 0 的像素
- 光流：GMFlow Sintel checkpoint
- 有效对应：默认启用 forward-backward consistency
- 报告值：相邻帧对 OPW 的平均值乘 100
- Metropolis 正式复现：启用 forward-backward consistency，不缩放光流输入

论文没有完整说明“有效 backward-flow correspondence”的工程判定，因此是否启用 forward-backward consistency 必须写入结果元数据，不能在实验之间静默切换。

## 2. GMFlow 自动发现

正常目录结构下不需要传 `--gmflow-repo` 或 `--gmflow-ckpt`。代码按以下顺序寻找 GMFlow：

1. 环境变量 `GMFLOW_REPO` / `GMFLOW_CKPT`（仅供非标准位置使用）
2. `~/gmflow`
3. 项目内 `third_party/gmflow` 或 `gmflow`

标准权重文件放在 `<gmflow>/pretrained/models/gmflow_sintel-0c07dcb3.pth`。

## 3. 推荐流程

### 3.1 默认在线计算

普通分辨率实验直接运行 `run.py`。OPW 默认开启，成功后 `summary.json` 同时包含 `avg_metrics.opw`、逐场景 OPW 和 `opw_evaluation` 元数据。

```bash
CUDA_VISIBLE_DEVICES=7 python run.py \
  --config config/vggt_baseline.yaml \
  --input <dataset_dir> \
  --output <prediction_dir> \
  --save-pt
```

### 3.2 高分辨率离线计算

VGGT 与 GMFlow 同时驻留可能导致显存不足。此时先显式跳过在线 OPW：

```bash
CUDA_VISIBLE_DEVICES=7 python run.py \
  --config config/vggt_baseline.yaml \
  --input <dataset_dir> \
  --output <prediction_dir> \
  --save-pt \
  --no-opw
```

预测完成后执行完整审计。脚本默认生成 `<prediction_dir>/opw.json`、`opw.tsv`，并自动备份和更新相邻的 `summary.json`：

```bash
CUDA_VISIBLE_DEVICES=7 python -u scripts/audit_opw_metric.py \
  --input-dir <dataset_dir> \
  --pred-dir <prediction_dir> \
  --flow-batch-size 2
```

仅做单场景排错时可加 `--max-scenes 1`。probe 会写独立文件，且不会自动合并 summary。完整审计若不希望更新 summary，显式加 `--no-update-summary`。

FB consistency 默认开启。`--no-fb-consistency` 仅用于明确标注的消融实验，不用于正式表格复现。

## 4. Summary 写入约束

完整审计在写回前必须满足：

1. 输入、预测、OPW JSON 与 summary 的场景集合完全一致。
2. 所有逐场景 OPW 均为有限数。
3. JSON 声明的均值等于逐场景均值。
4. 输出为标准 JSON，不允许 `NaN` 或 `Infinity`。
5. 原 summary 自动备份，更新使用原子替换。

历史审计文件中的 `opw_mode: capa_strict` 会在读取时兼容转换为 `protocol: capa_strict`；新文件只写 `protocol`。

## 5. 已确认的 Metropolis 8-line v3 结果

- 数据：`dataset/metropolis/metropolis_8line_noisy_v3`
- 预测：`output/noise_probe/metropolis_8line_v3/vggt`
- AbsRel：`0.100054`，即 `10.0054%`
- OPW：`150.40766694810657`
- 覆盖：`36/36` 场景
- 论文 VGGT 参考：AbsRel `10.0%`，OPW `149.4`

本轮接口整理不改变 `compute_opw()` 的数值公式。

## 6. 回归检查

```bash
python -m py_compile \
  run.py capa/utils/metric.py \
  scripts/audit_opw_metric.py scripts/merge_opw_summary.py

python -m pytest tests -q
```
