# Server-side VGGT baseline and OPW evaluation guide

本文档用于之后在服务器上复现 CAPA Table 1 中的 VGGT zero-shot baseline，并计算 CAPA-compatible OPW。  
当前阶段只准备命令和检查流程；不要在本地运行真实 VGGT、GMFlow 或大数据集实验。

## 1. 检查服务器环境

先进入项目目录：

```bash
cd /home/tankh/capa_reproduce
```

检查 GPU：

```bash
nvidia-smi
```

确认能看到可用 GPU、显存占用和驱动版本。若多人共用服务器，先选空闲 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
```

检查 conda 环境：

```bash
conda env list
conda activate capa
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

检查数据集路径。7-Scenes 是第一阶段推荐验证目标：

```bash
ls dataset/7scenes/7scenes_sfm | head
ls dataset/7scenes/7scenes_random100_noisy | head
ls dataset/7scenes/7scenes_lt3m_noisy | head
```

如果之后要跑 ScanNet 或 Metropolis，也先确认目录存在且容量合理：

```bash
du -sh dataset/scannet/
du -sh dataset/metropolis/
```

检查 GMFlow repo。GMFlow 源码应放在项目外部或 `third_party/gmflow`，不要放进 `capa/` 包内部：

```bash
export GMFLOW_REPO=/path/to/gmflow
test -f "$GMFLOW_REPO/gmflow/gmflow.py" && echo "GMFlow repo OK"
export PYTHONPATH="$GMFLOW_REPO:$PYTHONPATH"
```

检查 GMFlow checkpoint：

```bash
export GMFLOW_CKPT=/path/to/gmflow_sintel-0c07dcb3.pth
test -f "$GMFLOW_CKPT" && echo "GMFlow checkpoint OK"
```

## 2. 先跑 dry-run，不执行实验

默认脚本只包含 `vggt` 方法和 7-Scenes 三个输入。先用 dry-run 看命令是否符合预期：

```bash
DRY_RUN=1 \
CONDA_ENV=capa \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=output/tab1_zeroshot_vggt_baselines \
bash scripts/run_zeroshot_vggt_baselines.sh
```

如果同时想检查 OPW 命令拼接：

```bash
DRY_RUN=1 \
COMPUTE_OPW=1 \
CONDA_ENV=capa \
CUDA_VISIBLE_DEVICES=0 \
GMFLOW_REPO=/path/to/gmflow \
GMFLOW_CKPT=/path/to/gmflow_sintel-0c07dcb3.pth \
OUTPUT_ROOT=output/tab1_zeroshot_vggt_baselines \
bash scripts/run_zeroshot_vggt_baselines.sh
```

`DRY_RUN=1` 只打印命令，不创建输出目录，不运行模型，也不计算 OPW。

## 3. 跑一个单样本 debug

正式跑完整 7-Scenes 前，先选一个 `.pt` 文件做端到端冒烟测试：

```bash
SAMPLE=$(find dataset/7scenes/7scenes_sfm -name "*.pt" | head -n 1)
echo "$SAMPLE"
```

运行 VGGT baseline，确认可以产生 `*_pred.pt`：

```bash
conda run --no-capture-output -n capa python run.py \
  --config config/vggt_baseline.yaml \
  --input "$SAMPLE" \
  --output output/debug_vggt_one \
  --save-pt
```

然后只对这个 debug 输出计算 OPW。这里用 `--max-scenes 1` 和 `--verbose` 看 pair-level 诊断：

```bash
conda run --no-capture-output -n capa python scripts/audit_opw_metric.py \
  --gmflow-ckpt "$GMFLOW_CKPT" \
  --gmflow-repo "$GMFLOW_REPO" \
  --input-dir dataset/7scenes/7scenes_sfm \
  --pred-dir output/debug_vggt_one \
  --max-scenes 1 \
  --verbose \
  --out-json output/debug_vggt_one/opw.json \
  --out-tsv output/debug_vggt_one/opw.tsv
```

如果这里不能成功，不要继续跑完整实验。

## 4. 跑 7-Scenes VGGT baseline

默认脚本会处理这三个 7-Scenes 输入：

- `7scenes_sfm`
- `7scenes_random100_noisy`
- `7scenes_lt3m_noisy`

先只生成预测，不计算 OPW：

```bash
CONDA_ENV=capa \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT=output/tab1_zeroshot_vggt_baselines \
bash scripts/run_zeroshot_vggt_baselines.sh
```

脚本不会删除已有输出目录；如果目录已存在，会继续往对应 `run.log` 写日志，并由 `run.py` 负责覆盖或生成预测文件。

## 5. 计算 OPW

默认 OPW 是 formula-strict CAPA OPW：

- backward flow `F_{t+1=>t}`
- `beta=50`
- no depth normalization
- `fb_consistency=true`
- reported value multiplied by `100`

正式复现默认启用 forward-backward consistency；只有明确的消融实验才传 `--no-fb-consistency`。

如果希望 baseline 结束后自动计算 OPW，打开 `COMPUTE_OPW=1`：

```bash
COMPUTE_OPW=1 \
CONDA_ENV=capa \
CUDA_VISIBLE_DEVICES=0 \
GMFLOW_REPO=/path/to/gmflow \
GMFLOW_CKPT=/path/to/gmflow_sintel-0c07dcb3.pth \
OUTPUT_ROOT=output/tab1_zeroshot_vggt_baselines \
bash scripts/run_zeroshot_vggt_baselines.sh
```

每个 dataset/method 输出目录下会生成：

- `run.log`
- `opw.log`
- `opw.json`
- `opw.tsv`

也可以对已有预测单独计算 OPW：

```bash
conda run --no-capture-output -n capa python scripts/audit_opw_metric.py \
  --gmflow-ckpt "$GMFLOW_CKPT" \
  --gmflow-repo "$GMFLOW_REPO" \
  --input-dir dataset/7scenes/7scenes_sfm \
  --pred-dir output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt \
  --out-json output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/opw.json \
  --out-tsv output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/opw.tsv
```

## 6. 检查日志和输出

检查 run log：

```bash
less output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/run.log
```

确认：

- config 路径正确；
- input/output 路径正确；
- 每个 `.pt` 都生成了对应 `*_pred.pt`；
- 没有 CUDA OOM 或 checkpoint 下载失败。

检查 OPW log：

```bash
less output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/opw.log
```

检查机器可读结果：

```bash
cat output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/opw.json
head output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt/opw.tsv
```

`opw.json` 应包含：

- `input_dir`
- `pred_dir`
- `gmflow_ckpt`
- `beta`
- `fb_consistency`，默认应为 `true`
- `protocol`，应为 `capa_strict`
- `per_scene_opw`
- `mean_opw`
- `num_valid_scenes`
- `num_total_scenes`

## 7. 常见错误

### No *_pred.pt files found

含义：`--pred-dir` 里没有预测文件。常见原因：

- `run.py` 没跑成功；
- `--output` 和 `--pred-dir` 不一致；
- 输出目录层级写错。

处理：

```bash
find output/tab1_zeroshot_vggt_baselines -name "*_pred.pt" | head
```

### input-dir / pred-dir mismatch

含义：`audit_opw_metric.py` 会用 `scene_pred.pt` 去找 `input-dir/scene.pt`。如果 stem 对不上，就会报 missing input sample。

处理：

```bash
ls dataset/7scenes/7scenes_sfm | head
ls output/tab1_zeroshot_vggt_baselines/7scenes_sfm/vggt | head
```

确保预测文件名来自同一批输入样本。

### missing depth_pred_nhw

含义：`*_pred.pt` 内没有默认 key `depth_pred_nhw`。

处理：

```bash
python - <<'PY'
import torch
p = "output/.../example_pred.pt"
x = torch.load(p, map_location="cpu", weights_only=False)
print(x.keys())
PY
```

如果你的预测 key 不同，使用：

```bash
python scripts/audit_opw_metric.py ... --depth-key your_key
```

### CUDA OOM

处理建议：

- 换空闲 GPU：`export CUDA_VISIBLE_DEVICES=...`
- 先单样本 debug；
- 降低并发，不要多个大任务同时跑；
- 检查 `nvidia-smi` 是否有残留进程。

### GMFlow import failed

处理：

```bash
export GMFLOW_REPO=/path/to/gmflow
export PYTHONPATH="$GMFLOW_REPO:$PYTHONPATH"
test -f "$GMFLOW_REPO/gmflow/gmflow.py"
test -f "$GMFLOW_CKPT"
```

然后重新运行 OPW audit。不要把 GMFlow 源码复制进 `capa/` 包里。

### OPW is numerically unreasonable

先确认 `opw.json` 中：

- `protocol` 是 `capa_strict`；
- `beta` 是 `50.0`；
- `fb_consistency` 是 `true`；只有明确的消融实验才应为 `false`；
- RGB 输入是 `[T,3,H,W]`；
- depth 是 metric depth，未做 median/mean/MAD/min-max normalization；
- 返回值已经乘以 100。

典型异常：

- 差 100 倍：可能忘了 reported OPW 要乘以 100，或重复乘以 100。
- OPW 几乎为 0：可能 flow/warp 方向错了、权重全为 0、或只统计了极少有效像素。
- OPW 极大：可能 RGB `[0,255]` 未缩放、深度单位不是米、预测和输入帧顺序不匹配。

## 8. 第一阶段 validation target

第一阶段只建议对齐 VGGT 在 7-Scenes 的三列，目标是数量级接近 CAPA 表格，而不是追求单次完全一致：

- SfM: AbsRel / OPW around `4.4 / 3.1`
- Random100: around `4.0 / 3.2`
- `<3m`: around `4.0 / 2.9`

这些表格值是 reported metrics，即已乘以 100。若 AbsRel 或 OPW 差异很大，优先检查：

1. 输入 `.pt` 是否与预测目录匹配；
2. `depth_pred_nhw` 是否是 metric depth；
3. OPW 是否使用 backward flow `F_{t+1=>t}`；
4. RGB visibility weight 是否使用 `[0,1]` 范围和 `beta=50`；
5. `opw.json` 中的 `num_valid_scenes` 是否等于 `num_total_scenes`。
