# 云端 GPU 运行说明

## 建议配置

本轮已在 AutoDL 单卡 RTX 5090 上完成真实评测，使用约 50 GB 数据盘与镜像自带的 CUDA PyTorch。安装与下载细节见 [AutoDL 实测复现说明](AUTODL_RUN.md)。本项目使用普通 Python，运行脚本不依赖服务商 API，也可迁移到其他 Linux NVIDIA 云主机。

模型：Qwen2.5-VL-3B-Instruct，BF16 计算，输入最多约 1024 visual tokens，输出上限为 512 tokens。NF4 与不量化 BF16 的实测 peak allocated memory 分别为 2.458 与 7.204 GiB；这些值不包括整机的全部显存开销，也不是其他硬件的容量保证。

本轮用户提供的控制台 GPU 报价为人民币 2.88 元/小时；实际账单总额未读取。创建或重启实例前核对当前报价和存储费用。下载权重也占用时间，应优先复用缓存。

## 上传项目

从 GitHub 获取项目及已归档的实测记录：

~~~bash
git clone https://github.com/Yizhi-Q/multimodal-document-qa-evaluator.git
cd multimodal-document-qa-evaluator
~~~

也可在云主机的持久工作目录上传并解压项目 ZIP，进入包含 README.md 和 evaluate.py 的目录。在 Jupyter 中可打开 notebooks/cloud_evaluation.ipynb，依次运行里面的步骤。以下命令全部在云主机执行。

~~~bash
python -m pip install -r requirements-local.txt -r requirements-eval.txt
python -m unittest discover -s tests -v
bash scripts/run_cloud.sh validation
~~~

第一轮会下载真实 CORD 数据和模型，运行验证集前 20 个可用样本。此阶段可以调整参数并检查错误。下载需要网络；无需向本地 Mac 下载模型，也无需付费模型 API 密钥。

## 看结果

每次运行都会在 artifacts/runs/ 下生成一个新目录：

- summary.md：整体和分字段结果。
- run.json：模型/数据版本、配置、环境、完整结果。
- predictions.jsonl：逐条输出，运行中持续保存。
- errors.csv：错误值以及人工分析栏。

打开五张失败票据，对照原图、标注和模型输出，判断是读错、字段混淆、格式差异还是标注问题。不要根据错误类别自动断言模型“幻觉”。

## 固定配置后跑测试集

~~~bash
bash scripts/run_cloud.sh test
~~~

默认请求测试集最多 100 个样本。如果转换时排除了无有效标签或歧义标签的单据，manifest.json 会明确记录；报告样本量以实际值为准。记录后不要为了提高分数继续反复调整测试集。

两轮不同模型/量化设置的报告可以比较：

~~~bash
python compare_runs.py artifacts/runs/FIRST/run.json artifacts/runs/SECOND/run.json
~~~

模型、量化、计算精度、像素上限等环境变量在 .env.example 中有说明。对比时尽量只改一个因素。本轮两组测试使用相同的 BF16 计算，仅改变是否启用 NF4 权重量化。

## 保存后结束租用

下载整个运行目录，并保存 data/cord 下的 manifest.json 和 ATTRIBUTION.md。用于复现实验的模型和数据 revision 在报告及清单中。

程序结束不会自动关闭云实例。确认结果已下载，再在平台控制台停止或终止实例，并核对仍然存在的卷/存储费用。RunPod 停止后的 volume disk 仍可能收费；临时 container disk 在停止时会清除。不要把唯一一份结果留在临时磁盘里。

## 目前验证到哪里

本地与云端的 25 项离线检查通过。真实模型下载与校验、100 张验证票据的导出、前 20 张的验证实验，以及 95 张测试票据的两组 GPU 推理已完成。实测结果与原始记录见 [BENCHMARK.md](BENCHMARK.md)，五张验证图片的人工复核见 [VALIDATION_REVIEW.md](VALIDATION_REVIEW.md)。
