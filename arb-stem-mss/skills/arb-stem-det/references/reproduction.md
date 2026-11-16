# 训练、恢复与测试流程

所有命令在项目根目录执行。正式运行应固定项目 commit、数据版本、split、文本编码器 checkpoint、依赖版本与配置；输出目录采用独立实验名，避免用新配置覆盖旧权重。

## 环境与 CLAP

先创建项目环境并安装主依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e '.[test]'
python -m pytest -q
python scripts/doctor.py
pip freeze > work-environment.txt
```

将依赖快照放入实验记录，不把整个虚拟环境提交到 Git。当前项目没有强制 WandB 登录。

真实 CLAP 文本缓存需要官方 LAION CLAP 的兼容环境与音乐 checkpoint。官方音乐模型示例使用 `HTSAT-base`、non-fusion 与 RoBERTa text encoder，不能把默认 tiny 架构用于不同 checkpoint。参考 [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP#pretrained-models)。可在独立环境安装 `laion-clap`，执行缓存脚本后将生成的 `.pt` 文件供训练环境读取。

```bash
pip install -e '.[clap]'
python cache_text_embeddings.py \
  --manifests artifacts/manifests/train.jsonl \
              artifacts/manifests/validation.jsonl \
              artifacts/manifests/test.jsonl \
  --clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
  --output artifacts/text_embeddings.pt
```

CLAP 的模型/tokenizer 可能需要网络或预先准备的依赖缓存。真实权重尚未加载时，不能以脚本成功导入或 `--help` 通过替代实际编码验证。长提示会受 tokenizer 长度限制；使用训练覆盖的简洁英文模板作为第一阶段主设置，中文和复杂自然语言另行评测。

## 数据预检

```bash
python scripts/doctor.py --check-datasets \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine
python prepare_data.py --config configs/data.yaml \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine
```

先查看输出的 audit 与 split 文件，确认各 split 非空、曲目无重叠、音轨与标签可解析。数据配置中只启用真实存在的数据集。组合目标的文本只由有证据的类别标签生成，未经标注的属性不能自动补成 GT。

## 小样本排错

先执行一键合成复现，不依赖用户数据与真实 CLAP 权重：

```bash
python scripts/smoke_reproduction.py
```

默认 `artifacts/smoke` 必须不存在或为空。更换工作目录可使用 `--work-dir artifacts/smoke_repeat`，不要通过删除已有正式实验来排错。脚本记录完整子命令到 `commands.jsonl`，结果到 `result.json`；同时保留 manifest、缓存、checkpoint、evaluation JSON 及双查询 WAV，便于定位失败阶段。恢复一致性检查是相同环境中的数值验证，不承诺跨不同硬件逐位一致。

`configs/smoke.yaml` 仍使用原 DiCoSe U-Net 类，但缩小宽度、层级和音频片段。其默认数据路径为一键脚本生成的合成产物，手动运行须加 mock 门禁。如果将配置改为小规模真实 manifest 和真实缓存，使用：

```bash
python train_determination.py --config configs/smoke.yaml \
  --device cpu --max-steps 2 --output-dir runs/smoke
```

若运行仓库合成测试提供的 mock 缓存，应显式加 `--allow-mock-embeddings`。mock 标记需要随 checkpoint 保存，评测或推理时继续受到门禁；不能换一个文件名就把 mock 权重当真实文本模型。

检查数据 batch 的 shape 与文本向量维度。检查模型参数确有梯度、loss 有限、输出长度与输入一致。对同一混音更换不同非零查询，验证条件确实进入计算图；这只证明条件路径连通，不证明语义理解。

## 正式训练

```bash
python train_determination.py --config configs/det_text.yaml
```

训练主配置包含：

| 配置项 | 作用 |
| --- | --- |
| `model` | 文本向量维度、adapter 和原 DiCoSe U-Net 参数 |
| `audio` | 训练采样率与 segment samples |
| `data` | train/validation manifest 和 embedding cache |
| `train` | seed、epoch、batch、workers、优化器、设备、AMP、输出目录 |
| `loss` | waveform L1/MSE、MR-STFT 参数与可选 absence 权重 |

正式 U-Net 较大；先根据显存缩小 batch。变更 segment、batch、梯度策略或模型结构时建立独立配置。不要仅改默认值而不保存解析配置。

训练输出：

```text
runs/det_text/
├── config.json
├── runtime.json
├── metrics.jsonl
├── last.pt
└── best.pt
```

`best.pt` 按验证集选择，`last.pt` 用于进度保存。`runtime.json` 与 checkpoint 的 runtime 记录 Python、PyTorch、NumPy、CUDA、设备和源码 Git commit 等信息。检查日志中的训练 loss、验证 loss、有效样本数和静音/负例计数。全零输出可能让负例能量更低，但需要同时检查弱目标和有效目标的保留情况。

MSE 对照：

```bash
python train_determination.py --config configs/det_mse.yaml
```

保证两组使用相同 manifest、文本缓存、切分和预算。它比较损失设计；不能据此宣称复现了原论文完全相同的固定四类任务与数值。

## 恢复训练

```bash
python train_determination.py --config configs/det_text.yaml \
  --resume runs/det_text/last.pt
```

当前只保证完整 epoch 边界恢复。`--max-steps` 在 epoch 中间截断会产生 `partial_epoch=true`，拒绝作为恢复起点是预期行为。恢复会校验模型、编码器和数据身份，避免不知情地切换训练任务。

要迁移旧四维类别条件权重时，先制定显式映射并报告缺失/重初始化层；本项目不把旧权重做一次 `strict=False` 加载就称为完成文本迁移。重新训练的 adapter 与条件层需要真实语义监督。

```bash
python train_determination.py --config configs/det_text.yaml \
  --init-upstream /path/to/DiCoSe/Deterministic_model_unet/model.ckpt \
  --output-dir runs/det_text_initialized
```

初始化记录保存在 `upstream_initialization.json`，包括来源权重 SHA256、已加载参数及被跳过的层；它与 `--resume` 互斥。公开权重未在本次开发环境实际加载，首次迁移时检查该报告后再做小样本训练。

## 全曲评测

```bash
python evaluate.py --checkpoint runs/det_text/best.pt \
  --manifest artifacts/manifests/test.jsonl \
  --embedding-cache artifacts/text_embeddings.pt \
  --overlap 0.5 \
  --spectral-metrics \
  --output runs/det_text/test_metrics.json
```

全曲保存在 CPU，送入设备的是分块。各块重叠加权并归一化，处理尾段 padding，最终与 GT 对齐。显存不足时先检查 batch/segment；CPU 内存仍需能承载单曲音轨，极长音频需要另行流式读取方案。

全曲评测输出有效目标 SI-SDR、尺度敏感 SDR、L1/MSE；零目标/缺席目标能量另列。开启 `--spectral-metrics` 时，另以非重叠训练 segment 长度块计算 MR-STFT、按有效样本数加权，输出 `chunked_mrstft_loss` 并按歌曲汇总。该指标不是整曲一次 STFT，短尾按损失实现处理，并会增加 CPU FFT 开销；不加该参数时跳过。MR-STFT 同时作为训练/验证 loss 保存。不同录音的目标相位、时间偏移与重采样误差会显著影响波形指标，应先检查数据对齐。当前 SDR 非 BSS-Eval，不能与原论文使用不同定义的数字直接比较。

以歌曲为统计单位汇总，并保留逐查询记录。后续论文分析可在歌曲级做配对 bootstrap；同曲多个切片不视为独立样本。只评测 `--limit` 子集时在报告中写明条数与选择规则。

## 推理与产物

```bash
python separate.py --checkpoint runs/det_text/best.pt \
  --input /path/to/music.wav \
  --text 'The sound of piano' \
  --embedding-cache artifacts/text_embeddings.pt \
  --output-dir runs/det_text/separated
```

缓存没有新提示时，使用与训练相同的 `--clap-checkpoint` 代替 `--embedding-cache`；编码器身份不符应报错。输出为恢复原输入采样率和帧数的 mono target/remainder 浮点 WAV；输入立体声会被转 mono，不可把结果称为立体声分离。

检查 WAV 读取后的采样率、帧数、有限值和模型/prompt 元数据。浮点 WAV 可保留超出 [-1,1] 的幅度；若需要试听版限幅，应额外保存，不能独立裁剪 target/remainder 后继续声称保持原波形加和。

## 开放 stem 的实验登记

至少分别登记已见细类、同义提示、组合目标、目标监督留出类、完全声学留出类、真正缺席类、弱目标和同类实例查询。若某类仍存在于训练 mixture/submix，只能称目标监督未见；CLAP 预训练的外部类别暴露也需披露。

首先比较文本 DET 与其 MSE/L1＋MR-STFT 对照。之后才考虑冻结 DET 加 query-agnostic DIF、text-DIF 和 CD；本阶段没有这些训练结果。
