# arb-stem-mss

基于 DiCoSe 的文本查询音乐源分离项目，当前实现第一阶段 **DET**：输入混音与冻结 CLAP 文本向量，使用 DiCoSe 的真实一维 U-Net 主干，输出一个由查询指定的目标声源。数据采用 MGE-LDM 的 mixture/source/submix 三元组组织方式；GT source 与 submix 只用于监督，不进入模型输入。

```text
mixture [B,1,T] + text embedding [B,512]
                    ↓
             DiCoSe 1D U-Net
                    ↓
             target [B,1,T]
       remainder = mixture - target
```

正式配置为 22050 Hz、单声道、262144 samples；共享输出头不绑定四种乐器。支持的声源范围仍取决于数据、查询语义和训练，不能将“可输入任意文本”解释为已验证所有未见声源都能分离。

完整复现流程见 [项目 skill](skills/arb-stem-det/SKILL.md)，数据布局见 [data-layout.md](skills/arb-stem-det/references/data-layout.md)，设计依据见 [design.md](docs/design.md)，本次实际运行范围见 [reproduction-status.md](docs/reproduction-status.md)。

## 目录与路径

服务器目标布局如下，项目内路径不要求安装到这一固定位置：

```text
/data/work_backup/zixuanzhang/audio/mss-combine/
├── datasets/
│   ├── MUSDB18HQ/
│   └── MoisesDB/
├── DiCoSe/
├── MGE-LDM/
└── arb-stem-mss/
```

Slakh2100 可继续放在 `/data/work_backup/zixuanzhang/audio/MGE-LDM/datasets/slakh2100`，在 `configs/data.yaml` 配置其绝对路径。当前开发环境仅模拟该目录结构，没有挂载服务器数据。

## 从数据到训练

以下命令均在 `arb-stem-mss` 根目录运行。先安装本项目依赖，再核对 `configs/data.yaml` 中各数据集根目录、启用状态和划分设置。

项目要求 Python 3.10 或更新版本；本次 CPU 工程验证的精确环境见 `requirements-tested-cpu.txt`。GPU 训练请安装与服务器驱动匹配的 CUDA 版 PyTorch，并保存实际环境信息。当前训练入口为单进程、单设备训练。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/doctor.py --check-datasets \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine

python prepare_data.py --config configs/data.yaml \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine

python cache_text_embeddings.py \
  --manifests artifacts/manifests/train.jsonl \
              artifacts/manifests/validation.jsonl \
              artifacts/manifests/test.jsonl \
  --clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
  --output artifacts/text_embeddings.pt

python train_determination.py --config configs/det_text.yaml

python evaluate.py --checkpoint runs/det_text/best.pt \
  --manifest artifacts/manifests/test.jsonl \
  --embedding-cache artifacts/text_embeddings.pt \
  --spectral-metrics \
  --output runs/det_text/test_metrics.json
```

CLAP 是可选的预处理依赖，可用 `pip install -e '.[clap]'` 安装；实际兼容环境及 checkpoint 身份要求见 [详细流程](skills/arb-stem-det/references/reproduction.md)。首次使用请先执行 `pip install -e '.[test]'`、`python -m pytest -q`，再启动正式模型。`configs/smoke.yaml` 缩小的是同一个 DiCoSe U-Net，只验证工程链路；随机测试向量不会成为正式语义编码器。`scripts/doctor.py` 不加 `--check-datasets` 时只检查环境与 vendor 来源，不能替代完整扫描。

无需真实数据或 CLAP 权重的一键工程复现：

```bash
python scripts/smoke_reproduction.py
```

脚本在 `artifacts/smoke` 构造明确标记的合成夹具，执行准备→mock 缓存→训练→完整 epoch 恢复→全曲测试→双查询 WAV 推理，并核对恢复训练与连续训练。该目录须不存在或为空；也可用 `--work-dir artifacts/smoke_new`。详细命令记录与结果写入工作目录，不作为真实语义或音质实验。

上游 DiCoSe 的实际入口叫 `train_det.py`。本项目按需求提供 `train_determination.py`，`train_det.py` 为兼容别名。`configs/det_text.yaml` 使用 L1＋MR-STFT；`configs/det_mse.yaml` 是相同文本任务下的 MSE 对照，不等同于原论文固定类别实验的逐指标复现。

有原 DiCoSe DET checkpoint 时，可给训练命令加 `--init-upstream /path/to/model.ckpt`，只迁移名称及形状兼容的 U-Net 权重，并保存 `upstream_initialization.json`。新的文本 adapter 和不兼容条件投影仍需训练；初始化不是恢复训练，不能与 `--resume` 同时使用。

## 按文本提取

缓存已包含相同提示时：

```bash
python separate.py --checkpoint runs/det_text/best.pt \
  --input /path/to/mixture.wav \
  --text 'The sound of piano' \
  --text 'The sound of acoustic guitar' \
  --embedding-cache artifacts/text_embeddings.pt \
  --output-dir runs/det_text/separated
```

新提示可使用相同 CLAP checkpoint 在线编码，详见 skill。长音频按块重叠推理，输出单声道 target/remainder WAV 和推理元数据。多个查询独立执行，可能对应重叠声源；不要求它们的目标输出共同求和等于混音。

## 当前边界

- 实现文本条件 DET、三数据集适配、训练/恢复、全曲测试和 WAV 推理；实际验证范围以 `docs/reproduction-status.md` 为准。
- 未实现文本 DIF、三轨联合 latent diffusion 或 consistency distillation；为后续 DIF 保留混音、查询、目标估计及中间特征接口。
- 缺席查询与组合目标需要可靠标签及对应训练样本。MUSDB18 的 `other` 不能当作吉他、钢琴等细类真值。
- `remainder = mixture - target` 的加和正确是定义，不能作为分离质量证据；静音目标不混入 SI-SDR 平均值。
- 代码来源、固定 commit 与许可证见 `UPSTREAM.json` 和 `THIRD_PARTY_NOTICES.md`。

## 导入自己的 GitHub 仓库

把完整 `arb-stem-mss` 目录复制到服务器目标位置后，可单独初始化仓库；不要提交数据、缓存、权重或训练输出。

```bash
git init -b main
git add .
git commit -m 'Initialize text-conditioned DiCoSe DET project'
git remote add origin git@github.com:Lonely-house/arb-stem-mss.git
git push -u origin main
```

上面的远端 URL 是目标命名示例，需先在 GitHub 创建同名空仓库，或替换为已创建的仓库地址。本项目不会将示例 URL 当作已存在的远端。

如果交付位于 DiCoSe 的独立分支下 `arb-stem-mss/` 子目录，可直接复制该目录；若希望保留该子目录的 Git 历史，在包含它的 DiCoSe checkout 根目录执行 `git subtree split --prefix=arb-stem-mss -b arb-stem-mss-export`，再将导出的分支推到已创建的独立空仓库。无需把兄弟 DiCoSe/MGE-LDM 或 datasets 一并提交。
