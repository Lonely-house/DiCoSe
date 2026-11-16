# 数据布局与三元组

数据准备器统一从原始音轨建立 JSONL 查询记录。不同数据集先各自扫描和切分，再生成同类聚合、组合及可选负查询。正式运行前以 `configs/data.yaml` 与准备器实际报告为准。

## 服务器路径

```text
/data/work_backup/zixuanzhang/audio/mss-combine/
├── datasets/
│   ├── MUSDB18HQ/
│   └── MoisesDB/
├── DiCoSe/
├── MGE-LDM/
└── arb-stem-mss/
```

`--workspace-root` 指向 `mss-combine`，数据配置的相对根目录据此定位。Slakh 可用绝对路径 `/data/work_backup/zixuanzhang/audio/MGE-LDM/datasets/slakh2100`；若实际再包含一层解压目录，应配置到包含 train/validation/test 的层级。

开发环境模拟目录不含用户真实数据。不能将 fixture 复制到正式 `datasets` 后继续使用同一名称报告实验。

## MUSDB18HQ

```text
MUSDB18HQ/
├── train/
│   └── <song>/
│       ├── bass.wav
│       ├── drums.wav
│       ├── vocals.wav
│       ├── other.wav
│       └── mixture.wav
└── test/
    └── <song>/...
```

第一版使用 bass/drums/vocals/other 的求和作为 mixture，不把 `mixture.wav` 再当成一个声源相加。保留官方 test；validation 从 train 中按稳定、可记录的歌曲级规则产生，或使用显式 split 文件。

`other` 是聚合声源，不能用它作为独立 piano/guitar 真值。由 other 的存在也不能推出 saxophone 等具体乐器缺席，因此该数据集不应自动产生未经证实的细类负查询。

## MoisesDB

参考 MGE-LDM 已使用的原始目录格式：

```text
MoisesDB/
└── <song-uuid>/
    ├── data.json
    ├── vocals/
    │   ├── <track-uuid>.wav
    │   └── <another-track-uuid>.wav
    ├── guitar/
    │   └── <track-uuid>.wav
    └── drums/
        └── <track-uuid>.wav
```

数据标签以扫描器已支持的 metadata/目录结构为准。默认查询 guitar 时聚合该标签下所有轨道，避免一首歌中不同 guitar 文件配合同一文本产生矛盾监督。元数据没有明确属性时，不生成“失真独奏吉他”等未经证实的描述。

MoisesDB 没有在此项目中凭空指定唯一的“官方” train/validation/test。使用稳定 seed 的歌曲级划分，或传入实验明确记录的 split 文件。需要艺术家级隔离时，必须有可靠 artist 字段并启用对应策略；只做歌曲分组不能声称消除了所有表演者泄漏。

## Slakh2100

```text
slakh2100/
├── train/
│   └── Track00000/
│       ├── metadata.yaml
│       └── stems/
│           ├── S00.wav
│           └── S01.wav
├── validation/
└── test/
```

扫描器支持其实现明确列出的 WAV/FLAC 扩展名。乐器标签从 `metadata.yaml` 读取，不能通过 S00/S01 文件名猜测乐器。保留官方 split，并检查实际版本的路径和 metadata key。静音或不活跃 stem 的处理记录在准备报告；不要把原项目四类聚合预处理误当作完整乐器覆盖。

同一 MIDI 或近重复演奏可能需要额外跨集合检查；标准歌曲 ID 隔离不自动等于声学内容完全去重。

## 查询 manifest

每行对应一个混音与一个查询，所有 stem 路径可解析：

```json
{"song_id":"moisesdb:example-song","dataset":"moisesdb","split":"train","stems":[{"id":"guitar-1","label":"guitar","path":"/absolute/path/guitar/1.wav"},{"id":"guitar-2","label":"guitar","path":"/absolute/path/guitar/2.wav"},{"id":"drums-1","label":"drums","path":"/absolute/path/drums/1.wav"}],"target_ids":["guitar-1","guitar-2"],"prompt":"The sound of guitar","target_present":true}
```

该例 target 为两条 guitar 之和，其余为 drums。组合查询选多个标签对应的全部 ID；负查询使用空 `target_ids` 与 `target_present=false`。源标签不完备时应关闭自动负例。

prepare 阶段预展开有限数量的组合，便于真实 CLAP 缓存覆盖全部提示。训练时随机改变对齐裁剪位置，不在缓存外隐式生成新提示。迁移数据路径后重新生成 manifest，并使用新 manifest 的身份重新建立实验。

默认配置给每首歌保留全部单标签查询，另取最多两个组合查询、组合最多三个标签；缺席负例默认关闭。MUSDB 的验证比例为 train 内稳定 song hash 的 0.1；Moises 默认 0.8/0.1/0.1 的 song hash 划分，二者均不是新增的官方验证名单。需要固定名单时用 `split_file` 指向 canonical song ID 到 split 的 JSON 映射，映射必须覆盖所有扫描歌曲。`require_nonempty_splits=true` 会在正式准备时拒绝空集合；极小测试 fixture 可单独关闭，并保留 audit 中的 empty_splits 记录。

准备阶段写出 `train.jsonl`、`validation.jsonl`、`test.jsonl`、`splits.json`、`prompts.json` 和 `audit.json`。扫描器检查路径与规则，不做跨数据集音频内容指纹去重；这项限制也记录在 audit 中。

## 波形加载约束

- 使用 `soundfile` 读取原始波形并转 mono，按共同时间边界裁剪及重采样。同一首歌的 native sample rate 必须相同；混用采样率会报错，应先离线对齐。禁止对不同 stem 随意移动时间轴。
- mixture 来自相同 stem 集合的求和；source 是目标子集之和；submix 为差值。
- 不同音轨时长仅按末尾静音处理，较短轨右侧补零；不能修复未知时间偏移。短片段统一 padding，`num_samples` 记录有效长度，损失与指标避免把 padding 当真实静音。
- 不独立归一化 mix/source；如果使用 gain，所有三元组成员共享同一变换。
- 缺文件、重复 stem ID、未识别标签、空 split、缓存缺提示、采样长度问题均需检查报告。不要自动填充未知 stem 为零再宣称数据有效。

`QueryDataset` 返回 mixture/source/submix `[1,T]`、embedding `[D]`、prompt、present、song_id 与 num_samples。GT source/submix 属于监督；模型调用必须只取 mixture 与 embedding。
