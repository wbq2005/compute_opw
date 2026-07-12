# OPW 复现协议

更新日期：2026-07-12

本文档记录当前仓库中 OPW 的唯一推荐评测路径。目标是让预测、OPW
计算和结果汇总彼此独立，并保证不完整结果不会被写成一次成功实验。

## 1. 评测定义

- 实现入口：`capa/utils/metric.py::compute_opw()`
- 协议模式：`capa_strict`
- RGB 可见性权重：`beta=50`
- 评测区域：稠密 GT 中有限且大于 0 的像素
- 光流：GMFlow Sintel checkpoint
- 最终报告值：逐相邻帧对 OPW 的平均，再乘 100
- Metropolis 最终复现设置：启用 forward-backward consistency，不缩放光流输入

论文没有完整给出“有效 backward-flow correspondence”的工程判定细节。
因此 `fb_consistency` 必须作为协议元数据记录，不能在不同实验间静默切换。

## 2. 推荐流程

### 2.1 生成预测

基线阶段不加载 GMFlow：

```bash
CUDA_VISIBLE_DEVICES=7 python run.py \
  --config config/vggt_baseline.yaml \
  --input <dataset_dir> \
  --output <prediction_dir> \
  --save-pt \
  --no-opw
```

此时 `summary.json` 的 `avg_metrics` 不包含 `opw`，并记录：

```json
"opw_evaluation": {"status": "not_requested", "online": false}
```

### 2.2 离线计算并写回 OPW

高分辨率视频采用分块光流，避免 VGGT 与 GMFlow 同时占用显存：

```bash
export GMFLOW_REPO=/home/tankh/gmflow
export GMFLOW_CKPT=/home/tankh/gmflow/pretrained/models/gmflow_sintel-0c07dcb3.pth
export PYTHONPATH="$GMFLOW_REPO:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=7 python -u scripts/audit_opw_metric.py \
  --gmflow-ckpt "$GMFLOW_CKPT" \
  --gmflow-repo "$GMFLOW_REPO" \
  --input-dir <dataset_dir> \
  --pred-dir <prediction_dir> \
  --flow-batch-size 2 \
  --fb-consistency \
  --out-json <prediction_dir>/opw_capa_strict_fb.json \
  --out-tsv <prediction_dir>/opw_capa_strict_fb.tsv \
  --update-summary <prediction_dir>/summary.json
```

审计与合并必须满足：

1. 输入、预测、OPW JSON 和 summary 的场景集合完全相同。
2. 所有逐场景 OPW 都是有限数。
3. JSON 中声明的均值等于逐场景均值。
4. summary 使用标准 JSON，不允许 `NaN` 或 `Infinity`。
5. 写回前自动备份原 summary，写回采用原子替换。

## 3. 已确认的 Metropolis 8-line v3 结果

- 数据：`dataset/metropolis/metropolis_8line_noisy_v3`
- 预测：`output/noise_probe/metropolis_8line_v3/vggt`
- 正式汇总：`output/noise_probe/metropolis_8line_v3/vggt/summary.json`
- OPW 审计：`output/noise_probe/metropolis_8line_v3/vggt/opw_capa_strict_fb.json`
- 逐场景 OPW：`output/noise_probe/metropolis_8line_v3/vggt/opw_capa_strict_fb.tsv`
- AbsRel：`0.100054`，即 `10.0054%`
- OPW：`150.40766694810657`
- 覆盖：`36/36` 场景
- 论文 VGGT 参考：AbsRel `10.0%`，OPW `149.4`

代码整理前后的 v3 数值完全一致。整理只改变失败语义、结果校验和 JSON
写入方式，不改变 `compute_opw()` 的数值公式。

## 4. 回归检查

```bash
python -m py_compile \
  run.py capa/utils/metric.py \
  scripts/audit_opw_metric.py scripts/merge_opw_summary.py

python -m pytest tests -q
```

2026-07-12 第二轮严谨性审阅后的结果为 `49 passed`。
