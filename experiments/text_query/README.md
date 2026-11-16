# Text-query hybrid separation：可运行的研究接口原型

方法可行性、两仓库的代码证据、新颖性边界与实验矩阵见 [RESEARCH.zh-CN.md](RESEARCH.zh-CN.md)，已执行验证见 [validation.json](validation.json)。

本目录独立于原项目的 Lightning/W&B 依赖，只需要 `torch>=2.1` 和 `numpy`。它验证文本向量条件、两阶段优化、判别式锚点、生成式残差修复以及可变查询数量的接口。**没有附带训练权重，也没有验证真实音乐分离质量。** 小型卷积模型的时域感受野很短，正式实验应换成 DiCoSe U-Net 或其他具备长程上下文的查询分离骨干。

它借鉴 DiCoSe 的“先判别式分离、再生成式修复”，借鉴 MGE-LDM 的 mixture／目标／其余声源角色语义。这里是 **waveform 域的单目标 EDM 残差扩散**，不是 MGE-LDM 的三轨联合 latent diffusion：尚未实现其共享三轨去噪、observed-track mask、分轨时间步或潜空间编解码器。也没有实现 DiCoSe 的 consistency distillation 或直接复用其权重。

## 模型和变量

所有音频张量均为 `(B,C,T)`，查询向量为 `(B,D)`：

1. 共享查询分离器 `d=D(x,q)`，输出单个目标估计，`r0=x-d` 为其余声音。
2. 冻结 `D`，训练干净修正量 `u=s-d`；输入噪声变量 `u_sigma=u+sigma*epsilon`。
3. 修复网络条件为 `[u_sigma*c_in, x, d, r0]` 和同一查询向量。EDM 使用 `c_skip=sigma_data²/(sigma²+sigma_data²)`、`c_out=sigma*sigma_data/sqrt(sigma²+sigma_data²)`、`c_in=1/sqrt(sigma²+sigma_data²)`，损失权重 `(sigma²+sigma_data²)/(sigma*sigma_data)²`。
4. 训练 `log(sigma)~N(-2,1)`，默认 `sigma_data=0.1`；噪声单位和标准化后 waveform 幅度一致。推理采用 Karras 网格（默认 `sigma_max=0.5, sigma_min=0.002, rho=7`）和 Euler probability-flow 更新，从高噪声修正量开始，最终输出 `y=d+u_hat`。

这和原 DiCoSe 的训练变量、条件特征提取与采样初值并不相同，应作为新的残差参数化消融项。有限 `sigma_max` 的纯高斯初始分布是近似；需通过真实验证集调参，不能把默认值当论文复现设置。一步 Euler 只是数值基线，不是蒸馏后的一步一致性模型。

负查询采用真实不存在的目标，监督全零目标，另加无目标能量惩罚。它是待检验的减少幻觉策略，不保证未见类别能正确拒识。没有无条件训练，因此不提供 CFG 参数。

`remainder=x-y` 使每个查询对满足波形加和，**这只是记账关系，不能证明提取正确**。多条查询可重叠，例如“吉他”和“电吉他”，不能约束这些查询的目标共同加和等于混音。

## 数据与真实文本向量

不在本目录伪造文本编码器。请用同一个冻结的真实 CLAP text encoder/checkpoint，在外部将文本编码为单个向量并保存为 `.npy`；推理也必须使用该编码器。模型会归一化非零向量，但不会验证向量确实来自所声明的编码器。`--embedding-encoder` 将编码器及 checkpoint 的身份写入权重，并在第二阶段训练／推理时做字符串一致性校验。

训练 manifest 是 JSONL，路径相对 manifest 所在目录。每行一个混音／目标／文本查询：

```json
{"mix":"audio/song01_mix.npy","target":"audio/song01_electric_guitar.npy","prompt":"The sound of electric guitar","prompt_embedding":"queries/electric_guitar.npy","target_present":true,"sample_rate":22050}
{"mix":"audio/song01_mix.npy","target":null,"prompt":"The sound of saxophone","prompt_embedding":"queries/saxophone.npy","target_present":false,"sample_rate":22050}
```

- 音频 `.npy` 为有限实数 PCM amplitude，shape `(C,T)`；单声道可用 `(T,)`。文本向量为 `(D,)`，通常 CLAP 是 512 维，但应以实际 checkpoint 为准。
- 混音和目标必须对齐、同长度、同采样率。**不要分别对 mix 和 target 做峰值归一化**；如需缩放，二者共享同一个系数。这里不自动重采样，也不读取 WAV。
- `target_present=false` 时可省略目标文件，loader 构造零目标；若提供文件则必须全零。负查询的标签必须由数据集可靠确定，不能把未标注声源当成不存在。
- 一个目标文件可以是单件乐器，也可以是多个原始 stem 的和，配合相符的文本。训练时按共享起点裁剪；短片段零填充。
- 请按原歌曲／曲目／表演者划分数据，再创建查询和裁剪；同一混音的正负查询不能跨训练／验证／测试集。训练脚本本身不创建验证集或计算音质指标。

## 使用

需要从文字生成真实向量时，可在兼容的 CLAP 环境安装可选依赖 `laion-clap` 并使用本目录的导出命令。下面音乐 checkpoint 对应官方 `HTSAT-base`、non-fusion API；初始化还需要 CLAP 的 RoBERTa tokenizer/model 依赖：

```bash
python -m experiments.text_query.encode_queries \
  --clap-checkpoint /path/to/music_audioset_epoch_15_esc_90.14.pt \
  --prompt 'The sound of electric guitar' --prompt 'The sound of saxophone' \
  --output work/clap_queries
```

该命令写出 L2 归一化的 `query_000.npy` 等文件及 `queries.json`。后者记录提示、维度、包版本和 checkpoint SHA256；将其中 `embedding_encoder` 的完整值作为训练与推理的 `--embedding-encoder`。请保留元数据，manifest 中向量路径需相对 manifest 正确定位。CLAP API 已对照[官方说明](https://github.com/LAION-AI/CLAP#pretrained-models)核验；本次环境没有 CLAP 权重，尚未实际执行其模型加载与编码。导出脚本不会把随机向量作为文本替代。

在 DiCoSe 仓库根目录运行，先执行纯合成的连通性检查：

```bash
python -m experiments.text_query.smoke
```

smoke 中的随机查询向量仅是张量测试夹具，不代表文本语义。检查包括 det 和 EDM 反向传播、冻结判别式参数、5 个独立查询、有限采样结果、负样本加载与 checkpoint 往返。

真实训练先训练判别式模型，再冻结它训练修复器。下面的 `data/train.jsonl` 和音频／真实 CLAP 向量需自行准备；`--channels` 必须匹配音频：

```bash
python -m experiments.text_query.train train \
  --stage det --manifest data/train.jsonl --channels 2 --query-dim 512 \
  --embedding-encoder laion-clap/music_audioset_epoch_15_esc_90.14 \
  --sample-rate 22050 --epochs 10 --output work/det.pt

python -m experiments.text_query.train train \
  --stage refiner --manifest data/train.jsonl --det-checkpoint work/det.pt \
  --embedding-encoder laion-clap/music_audioset_epoch_15_esc_90.14 \
  --sample-rate 22050 --epochs 10 --output work/refiner.pt

python -m experiments.text_query.train infer \
  --checkpoint work/refiner.pt --mix data/song01_mix.npy \
  --queries data/queries/electric_guitar.npy data/queries/saxophone.npy \
  --embedding-encoder laion-clap/music_audioset_epoch_15_esc_90.14 \
  --sample-rate 22050 --steps 8 --seed 0 --output work/query_outputs
```

默认 CPU，可加 `--device cuda`。上面 10 epoch 只演示命令，不是有效收敛配方。模型不做长音频分块，建议先以短切片验证。输出每条查询的目标和 remainder `.npy`，以及推理设置 `inference.json`；不写 WAV。对仅有 det 权重的基线，推理使用 `--no-refine`。所有加载的训练／推理权重均用 `weights_only=True`。

`--sigma-data` 可在 refiner 阶段显式覆盖 checkpoint 中的值，应按训练集残差尺度校准；不指定时沿用已加载配置。声道、查询维度和网络大小由 det checkpoint 决定，refiner 阶段不重新建网。训练权重记录实际模型配置与主要训练设置，但当前 CLI 不提供优化器状态恢复或自动音质验证集评测。

## Python 接口

```python
from experiments.text_query.model import HybridSeparator, ModelConfig

model = HybridSeparator(ModelConfig(channels=2, query_dim=512))
seed = model.deterministic(mixture, query)                  # B,C,T / B,D
model.freeze_separator()
losses = model.diffusion_loss(mixture, target, query, target_present)
target_hat, remainder = model.separate(mixture, query, steps=8)
targets, remainders = model.separate_queries(mixture, queries, steps=8)
# queries: K,B,D -> outputs: K,B,C,T。K 不进入模型参数，不是固定四头。
```

“可输入任意数量查询”只是接口属性。开放词表、未见乐器、复杂属性描述、同类不同实例、负查询拒识都需要各自的真实数据、独立测试与指标支持。当前原型没有这些实验结果。
