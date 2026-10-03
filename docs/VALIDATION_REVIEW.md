# 验证集排错记录

2026-10-03，复核 CORD 验证集前 20 张的实际输出，并查看以下五张原图。所有改动在正式测试前完成；标准答案与评分规则没有改动。

## 验证过程

| 阶段 | 权重 / 计算精度 | 正确标注字段 | JSON 解析失败票据 |
|---|---|---:|---:|
| 初始配置 | NF4 / FP16 | 46/72（63.89%） | 2/20 |
| 对照检查 | 不量化 / FP16 | 45/72（62.50%） | 4/20 |
| 精度排查 | NF4 / BF16，保留旧提取要求 | 49/72（68.06%） | 0/20 |
| 最终验证 | NF4 / BF16，修正提取要求 | 54/72（75.00%） | 0/20 |

FP16 的部分原始输出是一长串重复符号，无法解析成 JSON。不量化也出现这一现象；改为 BF16 后，这 20 张的解析失败降为零。尚未检查中间张量的 NaN/Inf，因此这里记录观察结果与配置选择，不把数值溢出写成已经证明的原因。

旧提取要求让模型删除货币符号，但 CORD 的部分金额值含 `Rp`。修正要求为保留印刷金额旁的货币前缀，并明确 JSON null、印刷零值与税额标签。修正消除了所复核 `0015` 的前缀不一致问题，但部分其他票据变差；最终验证仍有 18 个字段错误，未把它描述为全面改善。

四个阶段使用相同 20 张图片和 72 个标注字段。计算精度与提示词的修改有分开的阶段，原始记录均保存；阶段间源文件哈希、提取问题文本不同，因此不能直接用 `compare_runs.py` 把它们作为只改变权重量化的测试对比。最终的两组测试使用相同提取问题与源代码。

原始记录：[初始 NF4](../benchmarks/2026-10-03/cord-validation-nf4-initial/run.json)、[初始 FP16](../benchmarks/2026-10-03/cord-validation-fp16-initial/run.json)、[BF16 排查](../benchmarks/2026-10-03/cord-validation-nf4-bf16-diagnostic/run.json)、[最终验证](../benchmarks/2026-10-03/cord-validation-nf4/run.json)。

## 五张原图复核

| 样本 | 图上可见内容与最终输出 | 复核判断 |
|---|---|---|
| [cord-validation-0001](../benchmarks/2026-10-03/validation-review-images/cord-validation-0001.png) | Total 为 `23.000`，Tunai 为 `50.000`，Kembali 为 `27.000`。最终模型把 total 写为 `50.000`；初始 NF4 则输出重复符号。 | 最终错误属于取错字段；解析成功不代表值正确。 |
| [cord-validation-0002](../benchmarks/2026-10-03/validation-review-images/cord-validation-0002.png) | PB1 旁的金额为 `1,818`，票据下方另有 `10% Tax Included`；Cash 为 `100,000`。最终模型税额返回 `10%`，cash 返回 `20,000`。 | 把税率当税额，把总额当现金；可见字段对应错误，不能笼统归因于 OCR。 |
| [cord-validation-0004](../benchmarks/2026-10-03/validation-review-images/cord-validation-0004.png) | Change 后印有 `:9,000`，模型返回 `9,000`，标准答案为 `:9,000`。 | 标注边界包含冒号。原评分保留这一错误，并记录边界问题；没有为了提高成绩删除标点。 |
| [cord-validation-0015](../benchmarks/2026-10-03/validation-review-images/cord-validation-0015.png) | Subtotal、Tax、Total、Cash、Change 都印有 `Rp` 前缀。初始模型按旧要求返回无前缀的金额；最终五个标注值均正确。 | 旧提示词与标注要求冲突，本例已修正。 |
| [cord-validation-0017](../benchmarks/2026-10-03/validation-review-images/cord-validation-0017.png) | 图上金额分别为 `20,000`、`20,000`、`100,000`、`80,000`；最终模型改成带点并多一个零的文本。标准答案 change 却写作 `80.000`。 | 模型存在分隔符和额外零错误；change 的标注也与可见分隔符不一致。保留官方标注及原分数，分别记录两类问题。 |

对应的逐字段复核见 [validation-review.csv](../benchmarks/2026-10-03/validation-review.csv)。未逐张查看的错误只保留自动差异类别，不给它们虚构成因。

## 下一步可以做的实验

使用剩余验证样本或另外收集的已标注票据，单独测试更高图像分辨率、只量化语言模块、金额与税率的 Schema 约束。每次说明改动和预期，再冻结配置使用独立测试集评测。本轮没有把测试集错误用于继续调参。

图片来自 CORD v2，保留其公开版本中的遮挡，并导出为 RGB PNG；遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。来源与转换说明见 [ATTRIBUTION.md](../benchmarks/2026-10-03/ATTRIBUTION.md)。
