# 2026-10-04 票据错误改进记录

日期按用户时区记为 2026-10-04；原始时间为 UTC，部分运行发生于 2026-10-03 UTC。

## 开发阶段

五种候选仅用已用于开发的前 20 张验证票据（72 个标注字段）设计和调试。每阶段重新运行 baseline；阶段内部同一数据、代码与模型配置。阶段之间代码不同，不能把全部阶段作为一次只改 prompt 的对照。

| 候选 | baseline 正确字段 | 候选正确字段 | 候选准确率 | baseline / 候选平均秒数 | 原始对照 |
|---|---:|---:|---:|---:|---|
| label-aware-v1 | 54/72 | 52/72 | 72.22% | 2.236 / 6.881 | [对照](pilot-long-rules/comparison.md) |
| receipt-compact-v1 | 54/72 | 49/72 | 68.06% | 2.211 / 2.138 | [对照](pilot-compact-rules/comparison.md) |
| amount-recheck-v1 | 54/72 | 54/72 | 75.00% | 2.173 / 2.519 | [对照](pilot-full-image-recheck/comparison.md) |
| percent-crop-v1 | 54/72 | 54/72 | 75.00% | 2.206 / 2.328 | [对照](pilot-crop-recheck/comparison.md) |
| label-context-v1 | 54/72 | 55/72 | 76.39% | 2.207 / 3.012 | [对照](pilot-context-recheck/comparison.md) |

最终候选在开发集修正了一个税额（54/72 → 55/72），整单准确率仍为 50%。详见 [唯一被评分字段的变化](pilot-context-recheck/field-changes.json)。

## 一次性 holdout 验证

冻结提取规则后，使用剩余 80 张验证票据，共 281 个标注字段；未按其错误再调整本轮方案。

| 方案 | 正确字段 | 字段准确率 | 整单准确率 | 失败文档 | 平均秒数 |
|---|---:|---:|---:|---:|---:|
| baseline | 227/281 | 80.78% | 65.00% | 0 | 2.338 |
| baseline + label-context-v1 | 227/281 | 80.78% | 65.00% | 0 | 2.321 |

只触发了一次额外复核，输出仍含异常百分比，未采纳；额外请求失败为 0。所有被评分字段保持一致，改善 / 退步均为 0。单次耗时的小差异不能说明加速。
**未观察到 holdout 上的准确率提升，额外复核默认关闭。** 保留这些尝试供复现和后续研究，不把开发集单例改善当作整体提升。

[holdout 对照](holdout-context-recheck/comparison.md) · [holdout baseline](holdout-context-recheck/baseline/run.json) · [holdout 候选](holdout-context-recheck/label-context-v1/run.json)

## 配置、验证与范围

固定模型版本 `66285546d2b821cf421d4f5eb2576359d3770cd3`，CORD v2 版本 `7f0115a4b758a71d6473b8d085751692da2fef98`。单卡 RTX 5090、NF4 权重、BF16 计算、greedy decoding、batch size 1，像素范围 200704–802816，生成上限 512 tokens。依赖及各阶段代码哈希见 run.json。

共核对 360 条本轮真实预测：JSONL 与完整报告逐条一致，重新计算的字段评分及汇总匹配。最终本地和云端 43 项离线检查通过。

holdout 代码 SHA256：`a2da2815ba44486b0f7a7f90cfb54cd9a358e336579e3a85a4ca7577e5a0988c`。开发对照后修正了合并结果的旧证据、解释与 confidence；没有改变模型输入、金额或评分。holdout 阶段两组使用完全相同的最终代码。

protocols.json 保存初次提取的完整 prompt；复核组的 recheck_attempt 记录第二次问题、字段、覆盖用的 receipt_profile（如有）、原始响应、耗时及失败，initial_prediction 保存初次响应。金额来自模型原图读取，没有将标准答案、建议金额或账目算式输入模型。

原 2026-10-03 测试记录、CORD 标签与评分代码未改。这是 holdout 验证，不能称为新的官方测试成绩；这 80 张现已使用，后续调参后须另收集独立样本。详细说明见 [错误改进实验](../../docs/ERROR_IMPROVEMENT.md)。
