# 多模态文档问答与字段提取评测

[English](README.md) | 简体中文

这是一个基于多模态大模型的个人项目：输入文档图片和提取要求，生成结构化字段，再与人工标注对比，输出准确率、推理耗时、显存占用和错误记录。

项目支持兼容 OpenAI 的 API，以及使用 Qwen2.5-VL 的本地或云端 NVIDIA GPU 推理；提供 Gradio 交互界面、批量评测、原始输出留存和结果复核流程。

**已完成真实 GPU 评测。** 2026 年 10 月 3 日，在 AutoDL 单卡 RTX 5090 上运行 Qwen2.5-VL-3B-Instruct，比较 BF16 权重与 NF4 四位量化。仓库内附带的三张合成单据用于检查程序流程，其模拟结果与真实模型成绩分别标记。

## 实测结果

两组实验使用相同的 **95 张 CORD v2 测试票据、319 个有效标注字段**，计算精度均为 BF16，仅改变权重量化方式。

| 权重设置 | 正确字段 / 标注字段 | 字段准确率 | 整单准确率 | 推理或解析失败票据 | 平均每张耗时 | 最大已记录的已分配显存 |
|---|---:|---:|---:|---:|---:|---:|
| BF16，不量化 | 272/319 | **85.27%** | 72.63% | 0 | **1.595 秒** | 7.204 GiB |
| NF4，四位量化 | 268/319 | **84.01%** | 72.63% | 0 | 2.284 秒 | **2.458 GiB** |

在本次实验中，NF4 的最大已分配显存低 **65.9%**，平均每张耗时长 **43.2%**。量化节省了显存，但在这张 GPU 和当前配置下没有加快推理。

- **字段准确率**：所有有效标注字段中，提取结果匹配标准答案的比例。
- **整单准确率**：一张票据的全部有效标注字段均正确，才计为整单正确。
- **耗时**：包括图片预处理、模型生成和 JSON 解析，不包括模型加载。
- **显存**：PyTorch 记录的已分配显存，不代表整机全部显存占用。
- 官方测试集原有 100 张票据；4 张包含歧义或非文本金额标注，1 张没有本任务的有效金额标注。5 张的排除原因已保存在数据清单中。

任务包含六类金额字段：

| 字段 | 含义 |
|---|---|
| `subtotal_amount` | 小计 |
| `tax_amount` | 税额 |
| `service_charge` | 服务费 |
| `total_amount` | 总额 |
| `cash_amount` | 现金支付金额 |
| `change_amount` | 找零 |

成绩对应上述六个字段的提取任务，不是完整 CORD 官方评测成绩。

[查看完整实测报告及原始记录](docs/BENCHMARK.md) · [查看五张验证票据的人工复核](docs/VALIDATION_REVIEW.md)

[查看票据错误改进实验](docs/ERROR_IMPROVEMENT.md)：新增可切换的标签规则、输出复核提示与独立留出验证。上表仍为原始 baseline 的测试成绩。
复核方案修正了一个开发字段；80 张留出验证的准确率仍为 227/281（80.78%），没有观察到整体提升，因此默认不启用额外复核。

## 项目已实现的功能

- **数据准备**：导入官方 CORD 数据，记录数据版本、原始顺序、排除原因和文件校验值。
- **结构化提取**：按字段 Schema 生成 JSON，并保留模型原始输出，便于排查解析问题。
- **批量评测**：记录逐张预测、字段准确率、整单准确率、耗时和显存指标。
- **错误复核**：生成标准答案与预测值的差异表，提供人工复核列和原图样例。
- **结果复现**：记录模型版本、依赖环境、数据与源代码指纹；支持对已有预测重新评分。
- **配置对比**：比较样本和评测代码一致的完整真实运行，拒绝混用模拟结果、回放结果或不同样本。
- **交互与测试**：提供 Gradio 界面、离线回归测试，以及 Python 3.10 / 3.12 的 GitHub Actions 检查。

## 从哪里开始

| 目标 | 文档 |
|---|---|
| 了解本次实测与各字段表现 | [实测报告](docs/BENCHMARK.md) |
| 查看 FP16 无效输出、提示词修正和字段混淆案例 | [验证集排错记录](docs/VALIDATION_REVIEW.md) |
| 在 AutoDL 复现已验证的配置 | [AutoDL 运行与复现](docs/AUTODL_RUN.md) |
| 在其他 Linux NVIDIA 云主机上运行 | [云端运行说明](docs/CLOUD_RUN.md) |
| 了解数据划分、评分和实验对比规则 | [评测协议](docs/EVALUATION.md) |
| 在 Jupyter 中运行 | [云端评测 Notebook](notebooks/cloud_evaluation.ipynb) |
| 理解代码与准备项目讲解 | [学习清单](docs/LEARNING.md) |

## 快速开始：离线检查

需要 Python 3.10 或更高版本。离线测试和模拟评测只依赖 Python 标准库，无需模型权重或 API 密钥。

```bash
git clone https://github.com/Yizhi-Q/multimodal-document-qa-evaluator.git
cd multimodal-document-qa-evaluator
python -m unittest discover -s tests -v
python evaluate.py --dry-run
```

每次运行都会在 `artifacts/runs/` 下创建独立目录。模拟评测明确标记为 **PIPELINE CHECK ONLY**，只用于验证流程，不能作为模型准确率。

## 在云端 GPU 上运行 Qwen

需要 Linux、可正常工作的 NVIDIA 驱动及支持 CUDA 的 PyTorch。本项目已在 RTX 5090 上实测；使用其他 GPU 时需确认 BF16 支持，并测量实际显存与速度。

```bash
python -m pip install -r requirements-local.txt -r requirements-eval.txt
python -m unittest discover -s tests -v
bash scripts/run_cloud.sh validation
```

脚本下载官方 CORD v2 验证集，选择导出后的前 20 张有效票据进行试运行，加载模型并保存真实输出。先检查验证集错误、固定提示词与配置，再运行测试集：

```bash
bash scripts/run_cloud.sh test
```

测试集用于最终评测，避免反复根据其错误调整提示词。每个官方数据划分需要下载数百 MB，模型权重和环境依赖也需要存储空间。

若要复现本轮成绩，请按 [AutoDL 复现说明](docs/AUTODL_RUN.md) 使用记录中的 CUDA / PyTorch 环境、`requirements-benchmark.txt`、固定版本及模型校验流程。

本轮固定版本：

- 模型：`Qwen/Qwen2.5-VL-3B-Instruct`，版本 `66285546d2b821cf421d4f5eb2576359d3770cd3`。
- 数据：`naver-clova-ix/cord-v2`，版本 `7f0115a4b758a71d6473b8d085751692da2fef98`。

**运行脚本不会自动关闭付费云实例。** 保存并下载结果后，需要在云平台控制台停止实例，并核对存储费用。

## 项目结构与输出

```text
app.py                         Gradio 交互界面
evaluate.py                    真实推理、模拟评测与已有预测回放
compare_runs.py                检查实验可比性并生成对比报告
mllm_docqa/
  core.py                      模型后端、提取要求与 JSON 解析
  cord.py                      CORD 字段定义与标签映射
  dataset.py                   数据校验与内容指纹
  scoring.py                   按字段 Schema 评分
  evaluation.py                指标、逐张记录和错误表
scripts/
  prepare_cord.py               官方数据下载与标注导出
  cache_verified_model.py       校验本地权重并建立固定版本缓存
  run_cloud.sh                  云端验证集 / 测试集运行入口
benchmarks/2026-10-03/          本轮实测、环境、版本与复核记录
docs/                          评测协议、运行说明与错误分析
notebooks/                     云端评测 Notebook
tests/                         离线回归测试
```

每次评测保存以下文件：

| 文件 | 内容 |
|---|---|
| `run.json` | 配置、可用的模型版本、依赖版本、数据与源代码指纹及指标 |
| `predictions.jsonl` | 每张文档的输出；逐条写入，运行中断时保留已完成记录 |
| `errors.csv` | 标准答案、预测值、可观察的差异类别及人工复核列 |
| `summary.md` | 标记评测模式的可读报告 |

只对真实存在标注的字段评分；未标注字段不计入成绩。失败请求保留在分母中，其标注字段计为错误，并返回非零退出状态。已有输出目录不会被覆盖，未完成报告标记为 `complete=false`。

### 不调用模型，重新评分已有预测

```bash
python evaluate.py --dataset data/cord/validation/annotations.jsonl \
  --limit 20 --predictions artifacts/runs/YOUR_RUN/predictions.jsonl
```

原运行的 `run.json` 需与预测文件位于同一目录，且数据指纹匹配。回放保留原运行来自真实推理还是模拟数据的身份。

### 比较两次真实运行

```bash
python compare_runs.py artifacts/runs/FIRST/run.json artifacts/runs/SECOND/run.json
```

比较工具会拒绝模拟或回放结果、不同样本，以及不同版本的评测源代码。

## API 与交互演示

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

在本地 `.env` 中配置：

- `MLLM_BACKEND=openai`：使用兼容 API 的后端。
- `MLLM_API_KEY`、`MLLM_BASE_URL`、`MLLM_MODEL`：密钥、服务地址与模型名。
- `MLLM_API_MODE`：选择服务支持的 Responses 或 Chat Completions 接口。

不要提交真实密钥或 `.env` 文件。启动交互界面或批量评测：

```bash
python app.py
python evaluate.py --dataset data/cord/validation/annotations.jsonl --limit 20
```

设置 `MLLM_BACKEND=transformers` 可使用 Qwen 后端，需另行安装 GPU 推理依赖。`MLLM_MODEL_REVISION` 用于固定模型版本；图片像素范围、权重量化、计算精度和生成长度等配置见 `.env.example`。

默认计算精度为 `bfloat16`，程序会检查 GPU 是否支持。初始验证使用 `float16` 时出现了无效输出，不量化配置也出现同类现象；排查记录见 [验证集排错记录](docs/VALIDATION_REVIEW.md)。

官方 parquet 数据已缓存时，`scripts/prepare_cord.py` 支持 `--offline`，但仍需明确提供不可变的数据版本。

## 适用范围与限制

- 本项目用于学习、实验和作品展示，尚未完成生产部署。
- CORD 包含印尼票据；当前任务评测六类金额字段，对中文发票、合同或企业业务单据的效果需要另外验证。
- 官方标注不覆盖合成样例中的商家、单据编号和日期任务，项目没有为真实数据虚构这些标签。
- CORD 金额按印刷文本匹配，保留小数点和千位分隔符；显式数值 Schema 支持小数与合法逗号分组，不自动猜测地区格式。
- 模型生成的 `confidence` 和 `evidence` 未经过概率校准或独立验证。
- 本轮未训练或微调模型；公开数据可能出现在预训练中，无法排除数据污染。
- 每种正式测试配置只在一张 GPU 上完整运行一次，未评测并发服务或其他硬件。

## 数据来源与许可证

CORD 由 Seunghyun Park、Seung Shin、Bado Lee、Junyeop Lee、Jaeheung Surh、Minjoon Seo 和 Hwalsuk Lee 于 2019 年发布：[官方项目](https://github.com/clovaai/cord) · [CORD v2 数据集](https://huggingface.co/datasets/naver-clova-ix/cord-v2)。

数据图片和标注遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，保留独立的来源说明；五张复核图片的来源与转换记录见 [ATTRIBUTION.md](benchmarks/2026-10-03/ATTRIBUTION.md)。项目代码遵循 [MIT 许可证](LICENSE)。
