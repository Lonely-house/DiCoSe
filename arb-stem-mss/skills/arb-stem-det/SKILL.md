---
name: arb-stem-det
description: 在 arb-stem-mss 中准备多轨数据、缓存真实文本条件、训练恢复和评测 DiCoSe 文本单目标 DET；适用于可复现任意 stem 查询实验，不包含尚未实现的 DIF 或 CD 训练。
---

# 文本条件 DiCoSe DET 复现

本 skill 用于从原始 MUSDB18HQ、MoisesDB、Slakh2100 音频建立可追踪的数据三元组，训练文本条件单目标分离器，并完成全曲评测与 WAV 推理。先完成 DET 能力验证，再规划 DIF。不得将合成接口检查、随机 embedding 或未训练模型输出报告成真实音乐分离效果。

## 1. 确认项目与复现范围

在执行前读取 `README.md`、`UPSTREAM.json`、`configs/data.yaml`、将使用的模型配置以及 `docs/reproduction-status.md`。保留工作区已有修改。以下路径以项目根目录为工作目录；文中服务器目录是部署目标，不表示开发环境已经连接用户服务器。

```text
<mss-combine>/
  datasets/
  DiCoSe/
  MGE-LDM/
  arb-stem-mss/          ← 当前工作目录
```

用户所称 `train_determination.py` 在本项目作为主入口；上游 DiCoSe 实际文件为 `train_det.py`，本项目同名文件只是主入口别名。不要把本项目的文本适配实验描述成未改动的原版四类别训练。

本轮必须保持以下设计：

1. 主干是固定版本 DiCoSe 的真实 `UNet1d`，正式音频规格为 mono、22050 Hz、262144 samples。
2. 模型只接收 mixture 与文本 embedding。GT source、GT submix 和目标音频的 CLAP embedding 不能进入推理输入。
3. `source = sum(selected stems)`，`mix = sum(all listed stems)`，`submix = mix - source`。
4. 网络直接输出一个 source；其余声音以减法计算。不要重复添加等价的 residual loss。
5. 多条相同类别轨道以该类别文本查询时应合并，组合查询的 target 必须包含文本对应的全部目标轨道。

设计细节与上游差异见 [design.md](../../docs/design.md)。

## 2. 环境安装与来源核验

使用独立 Python 环境，安装项目依赖。CLAP 预处理可能需要单独的兼容环境；缓存生成后，DET 训练无需在线运行文本编码器。

项目要求 Python 3.10 或更新版本。`requirements-tested-cpu.txt` 记录本次 CPU 数值验证环境；GPU 训练采用匹配服务器驱动的 CUDA 版 PyTorch，记录实际版本。当前入口支持单进程单设备。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/doctor.py
pip install -e '.[test]'
python -m pytest -q
```

核对 `UPSTREAM.json` 中 DiCoSe commit `a1dc0a41ad2b1829674a60ea74e74edfd1509083` 和 vendored 文件校验信息。MGE-LDM 的参考 commit 为 `e180d9a16e31a78afc6b40ca40cde60eae96382a`。保留 MIT 许可证与 `THIRD_PARTY_NOTICES.md`。

项目要求 Python>=3.10、PyTorch>=2.3；具体经过实测的 CPU 依赖见 `requirements-tested-cpu.txt`。`doctor.py` 不加参数时检查环境、CUDA 可用性及 vendor SHA，不证明数据已经挂载。确认模型 forward/backward、真实 U-Net 条件路径、三元组、裁剪/重采样、恢复策略和长音频输出长度通过后再扩大实验。测试夹具只能写入临时目录或明确的 smoke 输出目录。

## 3. 准备原始数据

读取 [data-layout.md](references/data-layout.md)，确认实际目录是原始波形及元数据，而非 MGE 的预编码 latent。先启用已经下载的数据集；Slakh 尚未到位时关闭对应项，不要创建空目录冒充数据。

编辑 `configs/data.yaml`，核对路径、划分和目标组合选项，再执行：

```bash
python scripts/doctor.py --check-datasets \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine
python prepare_data.py --config configs/data.yaml \
  --workspace-root /data/work_backup/zixuanzhang/audio/mss-combine
```

检查 `artifacts/manifests/train.jsonl`、`validation.jsonl`、`test.jsonl`、`audit.json`、`splits.json` 和 `prompts.json`，至少确认：

- 每个记录的 `song_id`、dataset、split、stem 文件路径、文本、目标 ID 都可追踪。
- 官方测试集未被重新混入训练集；同一首歌的所有查询只处于一个 split。
- 单类别、多类别、缺席查询的数量与配置一致；真实存在但当前切片静音的目标不伪装成不存在类别。
- 每个目标 ID 指向当前歌曲的 stem；同类文件全部合并；不将未知 `other` 内部类别标成缺席。
- 原始路径移动后重新准备 manifest；不要只改训练命令而继续读取旧服务器绝对路径。

第一版采用 stems 求和的 mixture，避免 mastered mixture 与 stems 非严格可加引入歧义。只在所有音轨上共享缩放，不分别峰值归一化 target 和 mix。

## 4. 缓存真实文本条件

使用同一冻结 CLAP text encoder/checkpoint 编码训练、验证和测试提示。缓存只编码文字，不能读取测试真值声源音频作为条件。模板提示涵盖类别与组合，但类别名不等于已解决任意复杂描述、中文或同类实例问题。

```bash
pip install -e '.[clap]'
python cache_text_embeddings.py \
  --manifests artifacts/manifests/train.jsonl \
              artifacts/manifests/validation.jsonl \
              artifacts/manifests/test.jsonl \
  --clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
  --output artifacts/text_embeddings.pt
```

检查缓存的 `encoder_id`、`query_dim`、`vectors` 和 `is_mock`，以及同名 `.pt.json` 元数据。正式训练必须是 `is_mock=false`，向量维度与模型配置一致，训练和推理必须使用同一编码器身份与规范化方式。缓存文件已存在时会拒绝覆盖，修改提示或编码器应先选择新输出路径。保存 checkpoint SHA256、包版本与提示文本以便复现。

显式 mock 缓存只用于 CPU smoke。不得给正式命令默认加 `--allow-mock-embeddings`，不得把 mock checkpoint 用于真实音质或语义泛化结论。安装与新提示编码细节见 [reproduction.md](references/reproduction.md)。

## 5. 先做小规模链路检查

无需真实数据的一键检查使用同一个 DiCoSe U-Net 的缩小配置，并明确生成合成 WAV 与 mock 向量：

```bash
python scripts/smoke_reproduction.py
```

默认工作目录为 `artifacts/smoke`，必须不存在或为空；需要重复实验时使用新的 `--work-dir artifacts/smoke_repeat`。脚本完整执行原始格式夹具准备、缓存、训练、epoch 恢复、恢复/连续训练一致性检查、全曲测试与双查询 WAV 推理，也检查 mock 门禁和非法恢复拒绝。结果保存在该目录的 `result.json`、`commands.jsonl`、`evaluation.json` 和 `separated/`。

正式数据的少量 batch 检查仍应使用真实文本缓存。单独使用 `configs/smoke.yaml` 时，先将配置指向已准备好的小规模 manifest 与缓存；该配置默认指向上述脚本生成的 `artifacts/smoke`，不是自动生成数据的入口。指定新输出目录可手动检查截断训练：

```bash
python train_determination.py --config configs/smoke.yaml \
  --device cpu --max-steps 2 --output-dir runs/smoke_manual \
  --allow-mock-embeddings
```

上述手动命令使用合成 mock fixture，因而明确声明 `--allow-mock-embeddings`；未声明时拒绝执行是正确行为。换成真实文本缓存后不要添加该参数。`--max-steps` 可能在 epoch 中间停止，这类 checkpoint 标记为 `partial_epoch=true`，可以用于推理检查，不能作为完整 epoch 恢复训练的起点。

检查 loss 为有限值，query adapter 与 U-Net 都收到梯度，没有将 GT 传入 forward，且产生 `last.pt`、`metrics.jsonl` 和解析后的配置。smoke 通过后仍不能报告音质收益。

## 6. 正式 DET 训练与 MSE 对照

按照可用显存调整 `train.batch_size` 和 workers；不要假设正式 256-channel U-Net 能在 CPU 或任意显卡上运行。音频长度、模型结构与主要超参数变更需保存在实验配置中。

```bash
python train_determination.py --config configs/det_text.yaml
```

默认文本方案使用 L1＋MR-STFT。MSE 对照使用相同数据、split、查询缓存和训练预算：

```bash
python train_determination.py --config configs/det_mse.yaml
```

MSE 对照保留原 DiCoSe 回归目标思想，但仍是文本任务；若需要复现原论文固定四类指标，必须另外运行原仓库 `train_det.py` 及其完整数据/评测设置，不能用本项目文本结果替代。

训练产生 `last.pt`、验证最佳 `best.pt`、`metrics.jsonl`、`config.json` 和 `runtime.json`。检查 checkpoint 保留模型配置、优化器、训练进度、RNG、文本编码器身份、manifest 身份和运行时软件/设备信息。根据验证集选模型，测试集仅用于最终评测；不要通过测试指标反复选择超参数。

如果使用上游固定类别 DET checkpoint 初始化，显式添加 `--init-upstream /path/to/model.ckpt`。程序仅迁移名称/形状兼容的 U-Net 参数，记录 checkpoint SHA256、已加载与跳过的 key 到 `upstream_initialization.json`。文本 adapter、条件维度改变的层需要重新训练；该操作不是 `--resume`，两者互斥。

## 7. 按完整 epoch 恢复

```bash
python train_determination.py --config configs/det_text.yaml \
  --resume runs/det_text/last.pt
```

当前恢复语义只覆盖完整 epoch 边界。恢复前核对训练/验证 manifest、文本编码器身份、模型配置与优化器设置；修改数据划分或文本模型后应建立新实验。不要删除 checkpoint 的身份检查来强行恢复。中途步数截断的 checkpoint 必须重新从上一个完整 epoch 开始训练。

若需修改训练预算，应区分可调整的结束 epoch、设备与输出目录和影响优化轨迹的超参数；以程序实际校验为准。无法确认时先读 checkpoint 元数据，不要猜测恢复成功。

## 8. 全曲测试

```bash
python evaluate.py --checkpoint runs/det_text/best.pt \
  --manifest artifacts/manifests/test.jsonl \
  --embedding-cache artifacts/text_embeddings.pt \
  --spectral-metrics \
  --output runs/det_text/test_metrics.json
```

评测对全曲分块并重叠归一化；GPU 只处理当前块，不能把整首长音频直接塞入显存。固定 overlap 与设备，保留逐记录和按歌曲聚合结果。用 `--limit` 时只能称为子集诊断。

脚本分别输出有效目标 SI-SDR、尺度敏感 SDR、L1/MSE，以及静音或负查询输出能量。可选 `--spectral-metrics` 另算 `chunked_mrstft_loss`：按训练 segment 长度进行非重叠频谱评估，按有效样本数加权后做歌曲级汇总；它不是整曲一次 STFT，增加 CPU FFT 成本。MR-STFT 也在训练/验证 loss 中记录。当前 SDR 定义为波形误差的 scale-dependent SDR，不是 `museval` 的 BSS-Eval SDR。零目标不参与 SI-SDR/SDR 均值。按歌曲汇总避免查询较多的歌曲隐式占更大权重。

“剩余等于 mix 减 target”使 pair 加和恒成立；该数值不得列作高质量分离证据。开放类别结论还需要类别留出、组合留出、弱目标与负查询实验，见 design 文档。

## 9. 新音频与多查询推理

缓存已包含同一文本时：

```bash
python separate.py --checkpoint runs/det_text/best.pt \
  --input /path/to/mixture.wav \
  --text 'The sound of piano' \
  --text 'The sound of drums' \
  --embedding-cache artifacts/text_embeddings.pt \
  --output-dir runs/det_text/inference
```

新提示使用训练时的相同 CLAP checkpoint：

```bash
python separate.py --checkpoint runs/det_text/best.pt \
  --input /path/to/mixture.wav \
  --text 'The sound of electric guitar' \
  --clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
  --output-dir runs/det_text/inference_new_query
```

输入转换为模型采样率的 mono，分块前向后恢复原输入采样率与帧数；输出仍为 mono。WAV 使用浮点格式，避免独立剪裁破坏目标/补集的关系。保留文本、checkpoint、编码器、采样率、overlap、输出文件名等推理元数据。

查询 `guitar` 与 `acoustic guitar` 可以重叠；多个查询不是互斥分解。想区分同类两件乐器时，需要数据支持的实例属性或额外条件，不能仅靠一个类别名。

## 10. 验收、记录与交付

执行后把真实运行范围写入 `docs/reproduction-status.md` 与 `VALIDATION.json`：代码版本、Python/依赖/设备、数据审计摘要、配置、缓存身份、checkpoint 与结果路径、已通过项和尚未运行项。只记录实际执行的验证；未运行完整数据集、真实 CLAP 或 GPU 训练时明确保留为待验证。

项目可作为独立文件夹加入 GitHub 工作区。提交前检查 `.gitignore`，排除 datasets、缓存、权重、音频、训练日志、虚拟环境和私有服务器信息；保留用于复现的代码、配置、许可及文档。已有远端则按用户提供的地址推送；没有远端时先完成本地项目，再由明确的目标仓库承接，不虚构仓库 URL。

失败时先读 [troubleshooting.md](references/troubleshooting.md)，修正数据/身份/形状问题后从相关阶段继续，不通过关闭验证器来隐藏错误。DIF/CD 属于下一阶段，当前 skill 不提供不存在的训练入口。
