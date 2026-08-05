# Physical Conflict 模型实验 v1.3

## 最终结论

最终选择的模型是 **Qwen3-VL-4B-Instruct + LoRA checkpoint-104**。checkpoint 只使用官方训练集内部划分的 dev-validation 进行选择。模型锁定后，在未参与训练的官方 validation 和 test 上都优于原始 4B baseline：

| 数据划分 | 题目数量 | 4B Baseline | LoRA step-104 | 提升 | 配对 McNemar p 值 |
|---|---:|---:|---:|---:|---:|
| 内部 dev-validation | 311 | 64.31% | 71.70% | +7.40 个百分点 | 0.002667 |
| 官方 validation | 329 | 61.70% | 70.21% | +8.51 个百分点 | 0.001093 |
| 官方 test | 334 | 58.38% | 66.17% | +7.78 个百分点 | 0.000222 |

所有评测均未出现无效输出。官方 test 仅用于一次最终锁定的 baseline 与 LoRA 对比，并且是在模型和实验方案全部确定之后进行的。

## 标准数据流程

- Task C 基于 Task B 原有的 1,000 个 samples 生成，没有重新采样。
- 官方划分：train 703 个、validation 148 个、test 149 个 samples。
- 模型开发只使用官方 train。
- 内部 dev-train：562 个 samples、1,241 道题。
- 内部 dev-validation：141 个 samples、311 道题。
- 数据划分随机种子：`20260805`。
- 两个内部划分之间的 sample 和 source group 重叠数量均为 0。
- 官方 validation 和 test 没有参与训练或 checkpoint 选择。
- 官方 validation 只用于一次确认对比；官方 test 只用于一次最终锁定对比。

## 基础模型选择与 LoRA 训练

首先在相同的内部 dev-validation 上比较 Qwen3-VL-4B 和 Qwen3-VL-8B。4B 的准确率为 64.31%，8B 为 57.56%，而且 8B 的运行时间慢 45.4%，因此选择 4B 作为基础模型。

正式 LoRA 训练在一张 NVIDIA L40 上完成，使用 BF16 和 SDPA。训练期间冻结视觉编码器和 aligner，只在 `q_proj`、`k_proj`、`v_proj`、`o_proj` 上训练 LoRA adapter。主要配置如下：

- LoRA rank：8
- LoRA alpha：32
- LoRA dropout：0.05
- Epoch：1
- 学习率：`1e-4`
- 学习率调度：cosine
- Weight decay：0.01
- 有效 batch size：8
- 视频采样：最多 32 帧、2 FPS
- 训练图像像素上限：393,216

训练过程中保存了 step-52、step-104 和 step-156 三个 checkpoint，并在同一内部 dev-validation 上评测：

| 内部 dev 候选模型 | 正确题数 | 准确率 | 决策 |
|---|---:|---:|---|
| 原始 4B Baseline | 200/311 | 64.31% | 对照模型 |
| LoRA step-52 | 213/311 | 68.49% | 不采用 |
| LoRA step-104 | 223/311 | 71.70% | **最终选择** |
| LoRA step-156 | 221/311 | 71.06% | 不采用 |
| 定向 refinement step-46 | 214/311 | 68.81% | 不采用 |
| 定向 refinement step-92 | 215/311 | 69.13% | 不采用 |

step-104 的验证准确率最高，因此最终选择该 checkpoint。虽然 step-156 训练时间更长、loss 更低，但真实答题准确率略有下降。

另外进行了一次范围受限的定向 refinement，重点增加 quarter 和 duration 题目的训练比例，并回放其他题型。两个 refinement checkpoint 的整体 dev 准确率都低于 step-104，因此在运行官方 validation/test 之前就已拒绝，没有替换最终模型。

## 官方 Test 详细结果

| 问题类型 | 4B Baseline | LoRA step-104 | 提升 |
|---|---:|---:|---:|
| 是否存在冲突 | 87.92% | 91.28% | +3.36 个百分点 |
| 冲突出现在哪些 quarter | 7.89% | 31.58% | +23.68 个百分点 |
| 第一段冲突持续时间 | 36.84% | 39.47% | +2.63 个百分点 |
| 第一段冲突开始时间 | 57.89% | 57.89% | 0.00 个百分点 |
| 冲突时间最长的 quarter | 42.42% | 45.45% | +3.03 个百分点 |
| 非重叠冲突总时长 | 28.95% | 55.26% | +26.32 个百分点 |

| 数据集 | 4B Baseline | LoRA step-104 | 提升 |
|---|---:|---:|---:|
| NTU-CCTV-Fights | 45.29% | 55.16% | +9.87 个百分点 |
| RWF-2000 | 89.19% | 89.19% | 0.00 个百分点 |
| Surveillance Camera Fight | 72.97% | 83.78% | +10.81 个百分点 |
| UBI-Fights | 91.89% | 91.89% | 0.00 个百分点 |

## 已知问题

目前最明显的问题是 quarter 多选题的过度预测。在官方 test 的 38 道 quarter 题中，LoRA 有 32 道选择了全部四个 quarter，但标准答案中只有 9 道包含全部四个 quarter。

虽然 quarter 完全匹配准确率已经从 7.89% 提升到 31.58%，但仍然较低。第一段冲突持续时间的准确率也只有 39.47%。这些问题适合作为下一轮训练目标，本轮不会继续根据官方 test 调参。

## 固定配置与复现信息

- lmms-eval commit：`469f433341c53e8c1a673c80ef0fd9b678b8ac6e`
- 基础模型 snapshot：`ebb281ec70b05090aa6165b016eac8ec08e71b17`
- Adapter SHA-256：`36f77ff3a122fd639b17241cd309e13cfad135e51c3fdd41c776a0b5c4d31159`
- 视频解码器：decord
- 最大帧数：32
- 视频采样：2 FPS
- 评测图像像素上限：786,432
- Attention：SDPA
- Batch size：1
- Adapter 支持测试：17 个通过、1 个跳过，其中包含 8 个 adapter 专项子测试

## 产出物管理

本仓库只保存评测代码和实验报告，不提交视频、数据集、adapter、checkpoint、逐题输出或原始日志。模型与评测产出使用上面的 SHA-256 标识，并保存在独立的实验归档中。

## 最终决策

当前 physical conflict 任务使用 **Qwen3-VL-4B-Instruct + LoRA checkpoint-104**。不采用定向 refinement checkpoint。后续如需继续优化，应重新开始一轮 train/dev 实验，并保持本轮官方 test 结果锁定，不再用它选择模型。
