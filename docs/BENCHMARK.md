# Qwen2.5-VL 票据金额提取实测

2026-10-03，在 AutoDL 单卡 RTX 5090 上完成真实推理。模型为 Qwen2.5-VL-3B-Instruct；没有训练或微调。任务仅包含 CORD v2 的六个金额字段，不是完整 CORD 官方评测。

## 测试结果

| 设置 | 正确字段 / 标注字段 | 字段准确率 | 整单准确率 | 失败票据 | 成功请求平均耗时 | Peak allocated memory |
|---|---:|---:|---:|---:|---:|---:|
| NF4 | 268/319 | 84.01% | 72.63% | 0 | 2.284 秒 | 2.458 GiB |
| BF16 | 272/319 | 85.27% | 72.63% | 0 | 1.595 秒 | 7.204 GiB |

本次配对测试中，NF4 与 BF16 的字段准确率差为 -1.25 个百分点；NF4 的峰值显存占用低 65.9%。NF4 每张平均耗时比 BF16 长 43.2%。数值只描述这一批票据、这张 GPU 和当前设置。

耗时统计针对成功请求，包括图像预处理、生成和解析，不包含模型加载；不额外删除首张推理。显存采用 PyTorch 的 peak allocated memory，取所有成功请求在设备 0 上记录的最大值。该指标包含模型和推理张量，不等于整机显存占用，也不包含 CUDA 上下文或全部缓存。每种设置只完整运行一次，未测试服务并发。

| 设置 | 每张耗时中位数 | 每张耗时 P95 | 平均同步 GPU 生成耗时 |
|---|---:|---:|---:|
| NF4 | 2.143 秒 | 2.593 秒 | 2.190 秒 |
| BF16 | 1.594 秒 | 1.869 秒 | 1.502 秒 |

## 分字段结果

| 字段 | 标注数 | NF4 正确数 | NF4 准确率 | BF16 正确数 | BF16 准确率 |
|---|---:|---:|---:|---:|---:|
| subtotal_amount | 61 | 53 | 86.89% | 57 | 93.44% |
| tax_amount | 39 | 29 | 74.36% | 29 | 74.36% |
| total_amount | 91 | 78 | 85.71% | 81 | 89.01% |
| cash_amount | 63 | 54 | 85.71% | 55 | 87.30% |
| change_amount | 53 | 45 | 84.91% | 43 | 81.13% |
| service_charge | 12 | 9 | 75.00% | 7 | 58.33% |

## 数据与实验流程

- 验证集导出 100 张，按原始顺序评测前 20 张。验证字段正确 54/72（75.00%），失败 0 张。
- 验证后固定 prompt、解析与评分规则、模型版本、像素和输出长度。测试集上没有再调参。
- 官方测试集检查 100 张，排除 5 张，评测全部 95 张可用票据；两种设置使用相同样本和评分代码。
- 仅对存在真实标注的字段评分。未标注字段不视为缺失，不据此计算幻觉率。
- 金额按印刷文本匹配，保留小数点及千位分隔符；使用 Unicode NFKC、空白规整和大小写归一化，不进行模糊匹配或任意标点删除。失败请求的标注字段全部计为错误。

| 排除的测试样本 | 原因 |
|---|---|
| cord-test-0013 | Ambiguous/nontext CORD label: sub_total.subtotal_price |
| cord-test-0039 | No labelled amounts in the selected task |
| cord-test-0058 | Ambiguous/nontext CORD label: total.cashprice |
| cord-test-0095 | Ambiguous/nontext CORD label: total.cashprice |
| cord-test-0099 | Ambiguous/nontext CORD label: sub_total.tax_price |

## 固定配置

- 模型：`Qwen/Qwen2.5-VL-3B-Instruct`；revision：`66285546d2b821cf421d4f5eb2576359d3770cd3`。
- 数据：`naver-clova-ix/cord-v2`；revision：`7f0115a4b758a71d6473b8d085751692da2fef98`。
- 图像像素范围：200704–802816；输出最多 512 tokens；greedy decoding，不采样；batch size 1；SDPA。
- NF4：4-bit 权重、double quantization、BF16 计算；BF16：16-bit 权重和计算。
- GPU：`NVIDIA GeForce RTX 5090`；Python：`3.12.3`；PyTorch：`2.8.0+cu128`；Transformers：`4.57.6`；bitsandbytes：`0.50.2`。
- 评分和推理源文件组合 SHA256：`0beeef398635f64bf34a26cea4dd783dfa38a7e6bf3d66404bfa9ee6a6442354`。
- 模型从 ModelScope 下载后逐文件校验，与上述官方 Hugging Face revision 的文件校验值一致；推理使用固定版本的本地缓存。

## 原始记录

- [NF4 测试记录](../benchmarks/2026-10-03/cord-test-nf4/run.json)、[逐张输出](../benchmarks/2026-10-03/cord-test-nf4/predictions.jsonl)、[错误表](../benchmarks/2026-10-03/cord-test-nf4/errors.csv)。
- [BF16 测试记录](../benchmarks/2026-10-03/cord-test-bf16/run.json)、[逐张输出](../benchmarks/2026-10-03/cord-test-bf16/predictions.jsonl)、[错误表](../benchmarks/2026-10-03/cord-test-bf16/errors.csv)。
- [验证集记录](../benchmarks/2026-10-03/cord-validation-nf4/run.json)、[验证错误复核](VALIDATION_REVIEW.md)。
- [测试集清单](../benchmarks/2026-10-03/test-manifest.json)、[验证集清单](../benchmarks/2026-10-03/validation-manifest.json)、[模型完整性校验](../benchmarks/2026-10-03/model-integrity.json)、[环境记录](../benchmarks/2026-10-03/environment.json)。

## 限制

公开的小样本数据可能出现在模型预训练中；不能排除污染。测试结果不证明对中文发票、合同、钢铁生产单据或新企业数据的效果。模型生成的 confidence 和 evidence 未校准、未逐条验证。本轮没有训练、微调、工业部署或业务效益测量。

CORD 的图像与标签遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，来源与转换说明见数据导出中的 ATTRIBUTION.md。代码遵循项目的 MIT 许可证。
