# 复现状态

本次已完成 DET 项目实现和合成数据工程验收：完整测试 31 passed，0 failures，0 errors；默认路径的一键 smoke 也已通过。机器可读记录见 [VALIDATION.json](../VALIDATION.json)，精确 CPU 环境见 `requirements-tested-cpu.txt`。

## 数据与环境

- 服务器目标：`/data/work_backup/zixuanzhang/audio/mss-combine/arb-stem-mss`。
- 当前开发环境为模拟服务器目录，不可直接访问用户的 MUSDB18HQ、MoisesDB 或 Slakh2100。
- DiCoSe 参考 commit：`a1dc0a41ad2b1829674a60ea74e74edfd1509083`。
- MGE-LDM 参考 commit：`e180d9a16e31a78afc6b40ca40cde60eae96382a`。
- 实测 Python 3.12.14、PyTorch 2.15.0+cpu、NumPy 2.5.4，CPU。
- 运行发生在新建本地仓库尚无 HEAD 的阶段，runtime 中源码 commit 为 null；交付版本由 GitHub 分支固定，VALIDATION.json 记录被验证代码、配置与测试的 SHA256。
- 训练入口为单进程、单设备；CUDA 和 DDP 未验证，当前未实现 DDP。

## 验证记录

| 检查 | 状态 | 证据 |
| --- | --- | --- |
| vendored 文件与许可/来源核对 | 通过 | 三份文件逐字节对照原 checkout，SHA256 与固定版本匹配 |
| 缩小的真实 DiCoSe U-Net forward/backward | 通过 | 条件梯度、任意长度 pad/crop、feature 接口与来源校验 |
| 三数据集 fixture 扫描与 triplet 正确性 | 通过 | MUSDB/Moises/Slakh 原始布局的合成 WAV；同标签全轨与组合目标、非法目标拒绝 |
| 重采样/对齐裁剪/短音频 padding | 通过 | 共享时间窗、有效长度 masking、静音/短片段有限损失 |
| mock 与真实语义缓存门禁 | 通过 | mock 默认拒绝、encoder/hash 漂移拒绝、train/validation split 角色与泄漏检查 |
| DET 训练→完整 epoch 恢复→全曲测试 | 通过 | 一键脚本；1 epoch 后恢复至 2 epochs 与连续 2 epochs 逐参数一致；partial 恢复被拒绝；含可选 chunked MR-STFT |
| 多查询 WAV 推理与长度/采样率恢复 | 通过 | 双查询、原采样率与帧数、mono FLOAT WAV、有限输出及 runtime 记录 |
| 用户真实数据扫描与标签审计 | 未运行 | 当前未挂载真实数据 |
| 真实 CLAP checkpoint 加载及文本编码 | 未运行 | 需实际权重与兼容依赖 |
| 正式 U-Net 的完整数据训练 | 未运行 | 需真实数据与 GPU 预算 |
| 开放类别音质/未见类别泛化 | 未运行 | 需独立实验 |
| 文本 DIF/CD | 未实现 | 仅保留接口与后续设计 |

合成音频和 mock embedding 仅用于数值、数据与接口检查。它们不验证真实文本语义、音乐分离质量或任何已发表指标。

验收命令：`python -m pytest -q --junitxml=artifacts/validation/pytest.xml` 与 `python scripts/smoke_reproduction.py`。一键运行产物位于 `artifacts/smoke/`，被 Git 忽略；重跑请使用新的 `--work-dir`。环境预检确认 vendor SHA 正确，同时明确显示当前开发环境中的真实数据根目录不存在。

公开原 DET checkpoint 初始化仅提供兼容加载接口和报告，本次未加载真实公开权重。
