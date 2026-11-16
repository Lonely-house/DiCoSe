# 文本条件 DET 的设计

本项目按用户上传方案的第一阶段实现：继承 MGE-LDM 的三元组数据组织，改造 DiCoSe 的判别式 U-Net，让文本决定要提取的目标。第一版不把三条 GT 音轨当作 DET 输入，也不实现联合 latent diffusion。

## 输入、输出和条件

```text
mixture:          [B, 1, T]
CLAP text vector: [B, 512]
predicted source: [B, 1, T]
predicted submix: mixture - predicted source
```

正式配置为 22050 Hz、T=262144。真实主干来自固定版本 DiCoSe `audio_diffusion_pytorch_/modules.py` 中的 `UNet1d`。文本向量经 adapter 后进入原 U-Net 的全局 `context_features` 条件路径，参与多个残差层的调制。它不是只接收一个实际上未使用的 `embedding` 参数。

wrapper 为任意输入长度补齐到主干所需倍数，再裁回原长度。网络始终共享一个目标输出头，类别数量不决定 mask head 数量。多提示推理重复调用该共享模型；它不自动发现一首歌的全部声源数量。

## 数据三元组

对一首歌的完整 stem 集合 S 和目标子集 A：

```text
mix = sum(S)
source = sum(A)
submix = mix - source
```

相同 label 的多条轨道默认合并。只有可靠属性/实例标注支持时才区分同类实例。组合目标需要相应组合文本；负查询要求可靠确认目标不存在，不能因为缺标签就自动认定缺席。

所有音轨共享采样率、裁剪边界与缩放。第一版混音使用 stems 求和，避免 mastered mixture 的额外处理破坏严格加和关系。GT source/submix 仅供训练/评测，模型不读取它们的文本外语义或音频 embedding。

## 损失与评测

主设置使用 waveform L1＋MR-STFT；MSE 作为接近原 DiCoSe 回归目标的对照。预测 submix 是减法定义时，额外 residual L1 与 source L1 数值等价，因此不重复计入；pair 加和误差也不是独立质量证据。

有效目标报告 SI-SDR、尺度敏感 SDR 和波形误差；`evaluate.py --spectral-metrics` 可选输出按训练长度非重叠分块、按有效样本数加权的 `chunked_mrstft_loss`，不等同于整曲一次 STFT。尺度敏感 SDR 与 BSS-Eval/museval 定义不同，不能直接对比数值。静音/缺席目标另报输出能量，不混入 SI-SDR 均值；负例改善必须与弱目标召回一起评估。

## 原仓库与上传方案的差异

| 项目 | 原实现/文献 | 本阶段 |
| --- | --- | --- |
| DiCoSe 入口 | `train_det.py` | 主入口 `train_determination.py`，提供同名别名 |
| U-Net 查询 | 四维 one-hot | 冻结 CLAP text 向量和语义 adapter |
| 目标头 | U-Net 本身已经是单目标 | 保留共享单目标输出 |
| 主损失 | 原 DET 的 MSE | L1＋MR-STFT，另提供 MSE 对照 |
| MGE 数据角色 | mixture/submix/source | 保留三元组数据组织 |
| MGE 模型目标 | 三轨联合 latent 去噪/velocity | DET 仅做波形单目标回归 |
| 后续 DIF/CD | DiCoSe 原有类别条件路径 | 尚未实现文本适配与蒸馏 |

上传方案描述了 MGE-LDM 论文中的音频条件训练、文本条件推理。当前参考源码 `e180d9a` 对 source CLAP audio/text 向量随机凸组合，且实际组成 `[B,3,512]` 全局条件；这与论文设定要分别记录。本项目先使用 text→text 一致的条件设置，避免把目标音频条件作为测试时的额外信息。

## 复现与研究边界

“任意 stem”在第一版指不需要为每个类别增加输出头、可由文本查询改变目标。以下能力分别需要独立证据：已见细类、未作为 target 监督的类别、完全没出现在分离训练音频的类别、组合查询、细粒度属性以及同类实例。CLAP 预训练可能暴露过相应概念，需要披露。

原固定四类权重的条件投影维度不同，不能完整 strict-load 后认为语义能力已经迁移。该项目的 MSE 配置也不等价于原论文的全部数据、训练与评测设置。正式训练前应先完成原始数据检查和真实文本缓存，再给出音质/泛化结论。

## 后续 DIF 接口

DET 预留混音、目标估计、查询 embedding 与中间特征，供后续构造 `g(noisy_target, sigma, mixture, det_target, query, det_features)`。实现顺序建议为：

1. 独立证明文本 DET 的查询提取能力与负例/弱目标表现。
2. 冻结 DET，对比等预算判别式修复、query-agnostic DIF 和 text-DIF。
3. 单独检验 target/submix 联合修复与角色条件收益。
4. 教师有稳定收益后，再做 1/2/4 步 consistency distillation。

以上为后续实验设计，不代表当前代码已经实现相应模型。
