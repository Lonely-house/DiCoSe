# 常见问题与恢复办法

| 问题 | 先检查 | 处理 |
| --- | --- | --- |
| 数据目录不存在 | `configs/data.yaml`、workspace root、解压后的层级和大小写 | 配置真实根目录；关闭尚未到位的数据集，不建空目录蒙混 |
| 扫描不到 MoisesDB | 是否为 song UUID/data.json/stem/*.wav 原始布局 | 对照扫描器支持的 schema；预编码 latent 不能替代原波形 |
| Slakh 标签缺失 | metadata.yaml 的 stems key、instrument 标签、音轨扩展名 | 使用原始 metadata；不按 S00 文件名猜类别 |
| split 为空或有歌曲重叠 | 过滤条件、歌曲 ID、显式 split 文件 | 修正后重新 prepare，并建立新实验；不得取消泄漏检查 |
| artist split 无法执行 | metadata 是否包含可靠 artist | 补齐可信字段，或明确采用 song split 并披露限制 |
| 某提示没有缓存向量 | manifest prompt 是否原样一致，是否新增组合/同义词 | 重新对完整提示集合生成真实缓存，不用随机向量补洞 |
| CLAP checkpoint 加载失败 | HTSAT-base/non-fusion 与权重架构、包版本、依赖缓存 | 使用匹配官方环境；先单独完成真实文本编码再训练 |
| encoder identity 不一致 | checkpoint SHA256、CLAP 版本、规范化、is_mock | 使用同一真实编码器；新编码器需新实验，不改字符串绕过 |
| mock 权重被拒绝 | 是否只在 smoke 中生成随机向量 | 合成测试显式 allow-mock；正式训练生成真实 CLAP 缓存 |
| U-Net shape 不匹配 | mono、query_dim、网络 stride、配置参数 | 用模型 wrapper 的 padding/cropping；不截断 GT 到不同位置 |
| loss 变 NaN/Inf | 音频有限值、重采样、学习率、STFT 分母、AMP | 先用 FP32 小 batch 定位，修复源数据或参数后再继续 |
| loss 很低但目标缺失 | 是否全静音输出、negative 比例、有效目标活跃情况 | 分开看有效/弱目标指标，不能只优化负例能量 |
| 两条同类轨道训练冲突 | 相同文本是否随机对应不同同类文件 | 将同标签轨道合并；没有实例标签时不训练任意单轨实例 |
| 全曲推理尾段突变 | overlap、块边界权重、尾段 padding、有效长度 | 使用归一化重叠合成并检查短音频/非整块长度 |
| GPU OOM | 正式模型宽度、segment、batch、workers | 先减 batch；改结构/时长需保存新配置与实验标识 |
| resume 被拒绝 | partial_epoch、manifest/encoder/model 身份 | 回到完整 epoch checkpoint；不同任务按新实验训练 |
| 无法复现论文 SDR | 指标定义、声道/采样率、数据 split、任务条件是否一致 | 不将 scale-dependent SDR 与 BSS-Eval 混比；另跑原仓库完整基线 |
| target＋remainder 恰好等于 mix | remainder 是否由减法定义 | 这是恒等式；看 GT 误差、泄漏、负查询及听测 |

修改数据路径、标签或划分后，重新生成 manifest 与文本缓存，并记录差异。不要只覆盖旧文件而继续恢复旧实验。真实音乐尚未跑通时保留失败记录与具体检查范围，不写“已完成开放类别复现”。
