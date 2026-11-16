# 文本查询的判别式＋生成式 音乐源分离

基于 DiCoSe 与 MGE-LDM 的可行性分析：从固定类别条件走向开放查询，优先研究对输入的忠实度、缺席声源抑制与少步推理。

核验日期：2026-11-16 · 文献截至该日期 · 代码版本固定到下述 commit

**结论先行**

两个目标在架构上都可实现，但“任意 stem”应定义成**共享模型按文本抽取目标、输出数量由查询决定**。这不保证所有未见乐器、所有同类演奏实例都可辨识。MGE-LDM 已有文本开放类别抽取；应把研究贡献放在**判别式证据能否约束生成式修复，在开放查询下减少幻觉并保持波形细节**，而非只宣称首次加入文本或超过四轨。

## 1. 把“任意 stem”拆成可验证的问题

| 能力 | 合理承诺 | 成立所需条件 |
| --- | --- | --- |
| 可变查询数量 | 输入 K 个提示，调用同一模型 K 次或批量处理，无须新增类别输出头。 | 接口支持；计算通常随 K 增长。不是自动发现全部声源数量。 |
| 开放类别抽取 | 查询 piano、organ、acoustic guitar 等具体或组合声源。 | 文本语义条件＋覆盖更多类别的多轨监督；不能从 MUSDB18 的 other 标签获得细类真值。 |
| 分离训练未见类别 | 在预先留出的类别上评估泛化。 | 严格定义留出范围，披露语义编码器的预训练暴露；无通用成功保证。 |
| 同类实例抽取 | 区分两把吉他、独奏与伴奏等。 | 单个 guitar 标签通常不唯一；可能需要属性、时间、空间或参考音频。 |

“guitar”与“acoustic guitar”允许对应重叠声源。只有明确互斥且覆盖完整的查询集合，才适合全局求和约束；一般的文本查询列表不能强制所有输出之和等于混音。

## 2. 两个 fork 的真实能力与改动点

DiCoSe：`a1dc0a41ad2b1829674a60ea74e74edfd1509083`
MGE-LDM：`e180d9a16e31a78afc6b40ca40cde60eae96382a`

### DiCoSe：固定的主要是条件词表与监督覆盖

| 已核验事实 | 代码位置 | 实现含义 |
| --- | --- | --- |
| U-Net 随机选择一个目标，返回 one-hot 与所有 stems。 | [main/data.py:334](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/main/data.py#L334) | 本身就是单目标查询，不必先改成任意数量输出头。 |
| 四维条件；Slakh 目标为 bass/drums/guitar/piano。 | [configs/deterministic_model/unet_train.yaml:19](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/configs/deterministic_model/unet_train.yaml#L19) | 将全局条件换成真实文本向量，并扩大目标监督。 |
| 冻结判别式网络，抽取与类别条件相关的多层特征。 | [main/module_base.py:125](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/main/module_base.py#L125) | 判别式与 refiner 应接收同一个文本查询。 |
| EDM 预条件化去噪；采样从判别式输出加噪开始。 | [audio_diffusion_pytorch_/diffusion.py:259](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/audio_diffusion_pytorch_/diffusion.py#L259)；[audio_diffusion_pytorch_/diffusion.py:340](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/audio_diffusion_pytorch_/diffusion.py#L340) | 不能把该路径描述为从零生成目标；要保留训练与采样的一致性。 |
| U-Net 支持全局 features 和 token embedding。 | [audio_diffusion_pytorch_/modules.py:983](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/audio_diffusion_pytorch_/modules.py#L983) | 优先用 pooled CLAP 向量替换 one-hot；token cross-attention 是额外设计选择。 |
| BS-RoFormer 具有固定 mask heads、整数 stem embedding；当前 embedding 参数未成为文本条件。 | [main/msst/models/bs_roformer/bs_roformer.py:645](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/main/msst/models/bs_roformer/bs_roformer.py#L645)；[main/msst/models/bs_roformer/bs_roformer.py:2010](https://github.com/Lonely-house/DiCoSe/blob/a1dc0a41ad2b1829674a60ea74e74edfd1509083/main/msst/models/bs_roformer/bs_roformer.py#L2010) | 需要共享查询 head 与真实文本注入；仅多传一个 embedding 参数不够。 |

最小原仓库接入：新增查询样本适配器；令 `features=q_text`，将 `context_features` 改为编码器的实际维度；分别修改 deterministic、diffusion、CD 的输入与日志评测。旧 one-hot 投影层需替换，不能完整 strict-load 原权重。CLAP 可冻结，文本判别式、refiner 与对应 CD student 需要训练。BS-RoFormer 可迁移时频主干，但固定 mask heads 不能直接代表开放类别。

### MGE-LDM：值得迁移的是三种角色的关系

| 已核验事实 | 代码位置 | 迁移判断 |
| --- | --- | --- |
| 拼接 mixture/submix/source 三轨 latent。 | [multi_track_stable_audio/training/pipeline_diffusion.py:156](https://github.com/Lonely-house/MGE-LDM/blob/e180d9a16e31a78afc6b40ca40cde60eae96382a/multi_track_stable_audio/training/pipeline_diffusion.py#L156) | 三轨表示角色，不是三种乐器。目标之外所有声源可聚合为 submix。 |
| 当前 source 条件随机混合 CLAP audio/text 向量。 | [multi_track_stable_audio/models/diffusion.py:125](https://github.com/Lonely-house/MGE-LDM/blob/e180d9a16e31a78afc6b40ca40cde60eae96382a/multi_track_stable_audio/models/diffusion.py#L125) | 可以消融音频与文本语义对齐；测试时禁止使用真值声源的 audio embedding。 |
| 给观测轨道时间 0，不加噪、不计算该观测轨道的损失。 | [multi_track_stable_audio/training/pipeline_diffusion.py:184](https://github.com/Lonely-house/MGE-LDM/blob/e180d9a16e31a78afc6b40ca40cde60eae96382a/multi_track_stable_audio/training/pipeline_diffusion.py#L184) | 可将混音始终作为观测证据。精确模式是 (t,t,t)、(0,t,t)、(t,0,t)、(t,t,0)。 |
| 固定 mixture 进行 source extraction。 | [multi_track_stable_audio/inference/task_wrapper.py:369](https://github.com/Lonely-house/MGE-LDM/blob/e180d9a16e31a78afc6b40ca40cde60eae96382a/multi_track_stable_audio/inference/task_wrapper.py#L369) | 已有文本条件抽取；列表提示是 batch 条件，不等于联合发现全部 stem。 |
| loader 跳过 source 标签含 other 的组合。 | [data/dataset_multi_latent.py:247](https://github.com/Lonely-house/MGE-LDM/blob/e180d9a16e31a78afc6b40ca40cde60eae96382a/data/dataset_multi_latent.py#L247) | 不能把吸收异构数据的能力等同于所有聚合类都已充分监督。 |

当前公开配置为 16 kHz 单声道、64-channel latent 的三角色 DiT。它与 DiCoSe 的采样率、声道、EDM 目标不同；可迁移设计，不宜直接拼接权重、时间参数或噪声日程。公开 checkpoint 也与原论文训练设置不同，复现实验需分别记录。[checkpoint 说明](https://github.com/Lonely-house/MGE-LDM#-model-checkpoints)。

## 3. 推荐的两层实现路线

### A. 先建立文本条件的波形修复基线

```mermaid
flowchart LR
  A[混音 x 与文本 q] --> B[共享判别式 D]
  B --> C[目标估计 d]
  C --> R[条件生成式修复]
  A --> R
  R --> Y[目标输出]
  Y --> U[其余声音等于混音减去目标]
```

```text
d = D(x, q)
u_d = x − d
δ = y − stop_gradient(d)
δ_σ = δ + σ ε
δ_hat = EDM(δ_σ, σ ; x, d, u_d, q)
ŷ = d + δ_hat
```

使用同一个冻结文本编码器的真实 embedding，同时调制两个模型。先训 D，再固定 D 的权重与 eval 状态训练 refiner。该残差版是**新设计的研究基线**，不是 DiCoSe 原样复现，也未实现完整 MGE-LDM。原 DiCoSe 的特征条件、估计加噪与此处学习误差分布的方式需保持区分。

残差参数化本身不构成强新意：在给定 d 的理想函数族中，预测 y 与预测 y−d 可通过平移互换。需要在相同条件、主干、噪声预算下比较原始 warm-start、残差预条件化与普通二阶段回归。`σ_data` 应按训练残差的幅度统计校准，不能机械复制原音频超参数。锚点也只是条件与初始化，不保证后续生成忠于输入。

训练目标可由 EDM 去噪损失、目标波形/多分辨率频谱监督，以及不存在目标的输出能量约束组成。对不存在目标，正确误差目标是 `δ=−d`，而不是 0；否则模型会保留错误粗分离。负例需与“目标类别存在、当前片段静音”分别标注。文本存在性 gate 或不确定性 gate 作为后续消融，不先用启发阈值掩盖模型错误；同时检验低音量真目标，防止模型靠过度静音取得低误报。

条件中的 `x−d` 不增加独立信息，作用最多是角色归纳偏置，需要与仅用 `x,d` 比较。另需检查 D 在训练集过准造成的残差分布偏移；必要时按歌曲交叉预测产生 refiner 的训练估计。不能只对训练集上的微小残差拟合，再期望修复测试集上的大错误。

**物理求和的边界：**定义 `u=x−ŷ` 后，`ŷ+u=x` 是恒等式；它不能证明没有幻觉或泄漏。若只预测一个目标，补集 MSE 与目标 MSE 也完全相同，不能算额外独立监督。

### B. 在 A 确有收益后检验 MGE 的联合建模

让模型同时修复 target 与 submix，并将 mixture 作为不加噪的观测轨道；引入逐角色时间与观测掩码。增加没有显式代数耦合的双输出对照，检验角色交互是否有益；在解码后的波形或复数谱施加加和约束，同时评测每条目标的误差。负例与置信度条件单独消融。

若选择 latent 版本，分别计算 `E(x), E(y), E(x−y)`；通常 `E(x)−E(y) ≠ E(x−y)`。先测 clean stem 经过 codec 的重构上限、相位/延时、立体声与采样率损失，再决定是否值得搬入完整 MGE codec。可先迁移角色建模而保留 DiCoSe 波形路径。

最后才做文本条件 consistency distillation：教师、学生与冻结判别式接收同一查询，固定前端后比较 teacher 与 1/2/4 步 student。少步收益是待测结果；不能因为 DiCoSe 在固定类别上有效，就推断开放类别必然保留。

## 4. 实验：怎样证明组合有用

待验证假设：在相同文本前端、歌曲划分与预算下，生成式细化改善有效目标的保真度，且不增加不存在目标的误报；再验证少步蒸馏能保留该收益。

| 组别 | 实验 | 要排除的解释 |
| --- | --- | --- |
| E0 | 复现固定类 D 与 DiCoSe refinement；锁定原 commit 与 checkpoint。 | 基础管线或指标不可比。 |
| E1 | 文本 D；另加 AudioSep/CLAPSep 强文本基线。 | 收益只来自换文本前端。 |
| E2 | 同预算 D＋D 回归修复；D＋无文本生成器；D＋文本生成器。 | 收益只来自更多参数、更多算力或混音条件。 |
| E3 | 原 warm-start 与残差 EDM；逐项加入 source/rest 联合建模、负例、置信度条件。 | 不能把多个改动的收益混称为三角色创新。 |
| E4 | MGE-LDM、可用的 SAM Audio/FlowSep 系列；teacher 与 1/2/4 步 student。 | 开放类别生成式先例与推理成本被忽略。 |

### 数据与查询

- 第一轮用 Slakh 的完整可用乐器类别；读取底层单轨与标签，不能沿用只含四种聚合文件的预处理就宣称开放类别。
- 先按歌曲/演奏切分 train/validation/test，再切片与构造查询。Slakh 同一 MIDI 或近重复演奏的跨集合情况也需检查。
- 下一轮用 MoisesDB 等真实多轨检验域迁移；MUSDB18 适合标准四轨回归，other 不能提供独立钢琴/吉他真值。
- 每条记录包含 mixture、目标、其余、文本向量、类别存在标记、时间活跃标记及来源。随机 target、语义组和 hard-negative 查询分别分层采样，比例在验证集调节。
- 区分类别名、同义改写、属性描述、乐器组、相似类别与真正缺席类别。中文使用需另测多语能力；若用主要面向英文的编码器，应记录中文到规范英文提示的映射。
- 若类别只从 target 监督留出、仍出现在训练混音，称“目标监督未见”；若称“声学未见”，须从全部训练音轨排除。两者都应披露 CLAP 预训练的外部暴露。

### 指标与判定

| 测什么 | 建议指标与控制 |
| --- | --- |
| 有效目标保真 | SI-SDR/SDR、MR-STFT/Log-Mel，按歌曲及类别汇总，给歌曲级置信区间；有尺度敏感波形误差作补充。 |
| 不存在目标 | 输出 RMS、相对混音的能量比、验证集阈值下误报率。安静混音用绝对能量单列；不要把零目标混入 SI-SDR 均值。 |
| 泄漏与同类混淆 | 对完整 GT 多轨、近似乐器与重叠查询分别分析；盲听目标保留、干扰和新增内容。仅提高 CLAP 分数不足以证明正确分离。 |
| 多查询 | K 增长时的延迟、重复抽取及互斥 partition 下的重构误差。定义补集产生的零误差不能作为模型质量指标。 |
| 成本与公平性 | NFE、候选数、峰值显存、RTF、编码器/codec/重排全部耗时，同采样率/声道；SAM Audio 的辅助模型及多候选选择需计入。 |

推进条件：相对强文本 D，改善达到歌曲级统计支持，且负查询不超过预先确定的误报/能量容忍界限，再投资三角色模型与蒸馏。若同预算 D＋D 已有同样收益，或生成修复增加幻觉，应先重审组合价值，而不是扩大模型。具体阈值应根据实际数据与用途预先登记，本报告不虚构收益数字或 GPU 训练时长。

## 5. 本次实现与验证边界

新增代码位于 DiCoSe checkout 的 `experiments/text_query/`。这是仅需 PyTorch/NumPy 的小型波形域研究原型：共享文本向量条件的判别式网络、EDM 残差修复、真实外部向量缓存数据契约、两阶段训练、缺席负例及可变查询数推理。

- CPU smoke 通过：两个模型反向传播、冻结前端、5 个独立查询、有限数值采样、负查询加载与 checkpoint 往返。
- 独立复核通过：det 训练→refiner 训练→双查询 CLI 推理；前端参数保持不变、refiner 参数更新。
- 解析 oracle 检验通过：1／2／7 次网络调用准确恢复指定目标，未在 σ=0 调用去噪器。
- 残差尺度配置复核通过：refiner 可显式设 σ_data=0.25 并保存，0／负值被拒绝；训练设置写入 checkpoint。
- 可选 CLAP 导出脚本的官方 API、命令帮助和编译已检查；真实权重加载与语义编码尚未运行。

运行环境：Python 3.12、PyTorch 2.15.0+cpu、NumPy 2.5.4。详见同目录 validation.json；所有通过结果仅证明数值与工程接口。

合成张量中的 embedding 只用于数值与接口验证，不代表任何自然语言语义。正式训练应先用真实冻结文本编码器生成向量；推理必须使用同一编码器、版本与规范化方式。原型没有发布可用音质权重，也没有完成真实数据训练、MGE 三轨联合 latent、原 DiCoSe U-Net/BS-RoFormer 训练适配或 consistency student。

当前环境仅有 CPU，无数据集与预训练权重。已能检验工程链路，但两个目标的实际音乐分离质量、开放类别泛化与推理效率仍未获得实证。本研究目录包含说明、验证记录与新增原型代码；可通过分支 diff 审阅全部变更。

## 6. 相关工作与新颖性边界

| 来源 | 已有能力 | 本研究应回答的剩余问题 |
| --- | --- | --- |
| [DiCoSe，v2 2026](https://arxiv.org/abs/2412.06965) | 判别式后接扩散修复与一致性蒸馏。 | 文本开放查询下是否仍有保真与效率收益。 |
| [MGE-LDM，2025](https://arxiv.org/abs/2505.23305) | 三角色联合 latent、文本抽取、异构多轨。 | 判别式锚点如何减少生成错误与缺席幻觉。 |
| [AudioSep](https://arxiv.org/abs/2308.05037) | 自然语言条件的开放域抽取。 | 为何需要额外生成修复而非强回归模型。 |
| [CLAPSep](https://arxiv.org/abs/2402.17455) | CLAP 迁移、正负提示与泛化。 | 在强文本前端上，生成器增益是否仍成立。 |
| [Banquet，ISMIR 2024](https://arxiv.org/abs/2406.18747) | 查询共享 decoder，音乐分离超过四类。 | 文本细粒度条件与开放类别长尾表现。 |
| [FlowSep，ICASSP 2025](https://arxiv.org/abs/2409.07614) | 语言查询的 latent rectified flow 分离。 | 生成与输入保真之间的权衡。 |
| [SAM Audio，2025](https://arxiv.org/abs/2512.18099) | 多模态提示、flow matching target/residual。 | 同预算的音乐开放查询与轻量细化。 |
| [FlowSep2，2026 预印本](https://arxiv.org/abs/2608.22111) | mixture/text 条件 DiT 与 Self-Flow。 | 与当前生成式方法对比，而非只对四轨系统。 |

这些先例使“文本＋生成分离”“共享头超过四类”“target/rest 联合建模”都不能单独作为首次贡献。可探索的论文主线是：**在开放类别音乐抽取中，判别式证据约束的生成式细化，能否同时改善输入忠实度、缺席抑制与少步推理成本？**这是一项可以证伪的研究假设，新颖性和效果需由更全面文献检索及上述对照实验确认。
