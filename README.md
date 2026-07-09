# **CAPA:**
# **D**epth **C**ompletion **a**s **P**arameter-Efficient Test-Time **A**daptation

[![Website](assets/docs/badges/badge-website.svg)](https://research.nvidia.com/labs/dvl/projects/capa/)
[![Paper](assets/docs/badges/badge-pdf.svg)](https://arxiv.org/abs/2602.14751)

**Bingxin Ke<sup>1,2</sup>, Qunjie Zhou<sup>1</sup>, Jiahui Huang<sup>1</sup>, Xuanchi Ren<sup>1</sup>, Tianchang Shen<sup>1</sup>, Konrad Schindler<sup>2</sup>, Laura Leal-Taixé<sup>1</sup>, Shengyu Huang<sup>1</sup>**

<sup>1</sup>NVIDIA <sup>2</sup>ETH Zürich


<img src="assets/docs/imgs/example_optimization.gif" alt="Optimization Process" width="600">

## Setup

Each base model requires its own Python environment due to dependency conflicts.

```bash
bash scripts/setup_env.sh all       # setup all three base models
# Or setup for individual base models
# bash scripts/setup_env.sh vggt      # setup VGGT env
# bash scripts/setup_env.sh moge      # setup MoGe-2 env
# bash scripts/setup_env.sh unidepth  # setup UniDepth-v2 env
```

Each command creates a virtualenv under `venv/`, installs the appropriate torch build, model-specific dependencies, and the capa package. After setup, activate the corresponding environment:

```bash
source venv/vggt_env/bin/activate      # for VGGT
source venv/moge_env/bin/activate      # for MoGe-2
source venv/unidepth_env/bin/activate  # for UniDepth-v2
```

## Sample data

Sample data can be downloaded [here](https://share.phys.ethz.ch/~pf/bingkedata/capa/sample_data/).

To download all sample data:
```bash
mkdir -p input && wget -P input/ https://share.phys.ethz.ch/~pf/bingkedata/capa/sample_data/ 
```

## Usage

```bash
# Single image
python run.py --config config/vggt_vpt.yaml --input input/sample_data/ibims1_max-depth-5m_noise_10pct/corridor_02.pt --save-vis --verbose

# Single scene (video)
python run.py --config config/vggt_vpt.yaml --input input/sample_data/scannet_sift_noise_10pct/scene0777.pt --save-vis --verbose

# Process a directory
python run.py --config config/vggt_vpt.yaml --input input/sample_data --save-vis --verbose
```
<small>Note: VPT is slightlyl more stable to random state and more reproduciable than LoRA.</small>


## License

### CAPA Source Code 

Copyright © 2026 NVIDIA Corporation. All rights reserved.

The CAPA source code — all files in this repository **excluding** the `third_party/` directory — is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Use, reproduction, distribution, and creation of derivative works are permitted for **non-commercial purposes only**, subject to the attribution requirements of that license. Any redistribution must retain this copyright notice and a reference to the license.

Commercial use of the CAPA source code requires a separate written license from NVIDIA Corporation.

### Third-Party Components

This repository includes modified versions of third-party software located in the `third_party/` directory. Each component is governed exclusively by its own license; the CC BY-NC 4.0 license above does **not** apply to those files.

| Component | Directory | License | License File |
|---|---|---|---|
| VGGT | `third_party/VGGT_VPT/` | VGGT License v1 | [`third_party/VGGT_VPT/LICENSE.txt`](third_party/VGGT_VPT/LICENSE.txt) |
| MoGe-2| `third_party/MoGe_VPT/` | MIT License (MoGe code) / for Apache License 2.0 (DINOv2 code in `moge/model/dinov2/`) | [`third_party/MoGe_VPT/LICENSE`](third_party/MoGe_VPT/LICENSE) |
| UniDepth v2 | `third_party/UniDepth_VPT/` | CC BY-NC 4.0 | [`third_party/UniDepth_VPT/LICENSE`](third_party/UniDepth_VPT/LICENSE) |

The files within each subdirectory have been modified from their original versions. Modifications are limited to the DINOv2 encoder to support Visual Prompt Tuning (VPT); all modified blocks are delimited by the comments `# >>>>>>>>>>> Modified for VPT >>>>>>>>>>>` and `# <<<<<<<<<<< Modified for VPT <<<<<<<<<<<`. Redundant scripts are removed. Notwithstanding such modifications, each component remains subject to its original license as listed above.

<details>
    <summary>Modified files</summary>
    
- VGGT (from [version](https://github.com/facebookresearch/vggt/tree/8492456ce358ee9a4fe3274e36d73106b640fb5c)):
    - `third_party/VGGT_VPT/vggt/layers/vision_transformer.py`

- MoGe2 (from [version](https://github.com/microsoft/MoGe/commit/0286b495230a074aadf1c76cc5c679e943e5d1c6)):
    - `third_party/MoGe_VPT/moge/model/dinov2/models/vision_transformer.py`
    - `third_party/MoGe_VPT/moge/model/v2.py`

- UniDepth v2 (from [version](https://github.com/lpiccinelli-eth/UniDepth/tree/8d8cfe4c7ee15297099983607febf0d4f32eb3d6)):
    - `third_party/UniDepth_VPT/unidepth/models/backbones/dinov2.py`
    - `third_party/UniDepth_VPT/unidepth/models/unidepthv2/unidepthv2.py`

</details>

### Disclaimer

The software is provided "as is", without warranty of any kind, express or implied. To the fullest extent permitted by applicable law, NVIDIA Corporation disclaims all warranties, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall NVIDIA Corporation be liable for any claim, damages, or other liability arising from the use of this software.


## Bibtex
```bibtex
@misc{ke2026capa,
    Author = {Bingxin Ke and Qunjie Zhou and Jiahui Huang and Xuanchi Ren and Tianchang Shen and Konrad Schindler and Laura Leal-Taixé and Shengyu Huang},
    Title = {Depth Completion as Parameter-Efficient Test-Time Adaptation},
    Year = {2026},
    Eprint = {arXiv:2602.14751},
}
```