# AutoDL RTX 5090 运行记录与复现

2026-10-03 的云端机器：单卡 NVIDIA GeForce RTX 5090，32607 MiB 显存，驱动 580.76.05；基础镜像自带 Python 3.12.3、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。实际 CUDA 矩阵运算已通过。评测结果以 `docs/BENCHMARK.md` 及对应原始运行记录为准。

## 环境

项目、模型缓存与数据都放在 `/root/autodl-tmp` 数据盘。使用独立环境复用镜像自带的 GPU 框架，避免重复下载和覆盖基础环境：

```bash
cd /root/autodl-tmp/YOUR_PROJECT
/root/miniconda3/bin/python -m venv --system-site-packages .venv
source .venv/bin/activate
export HF_HOME=/root/autodl-tmp/receipt-cache/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/receipt-cache/pip
python -m pip install -r requirements-local.txt -r requirements-eval.txt
python -m unittest discover -s tests -v
```

5090 需要支持 Blackwell 的 PyTorch 构建，CUDA 12.8 或更新版本。CUDA 版本必须与主机驱动匹配。若已有正常工作的 GPU 框架，先验证再安装；`requirements-local.txt` 中的通用最低版本不能单独保证 5090 兼容。

来源：[PyTorch Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)。

## 下载与评测

本轮模型通过魔搭官方工具下载。不要将代理地址、SSH 密码或开发者 Token 保存到仓库。模型约 7.5 GB，下载速度取决于实际线路；GPU 在下载期间也可能计费。

```bash
python -m pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct \
  --local_dir /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
export HF_HOME=/root/autodl-tmp/receipt-cache/huggingface
python scripts/cache_verified_model.py \
  --model-dir /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

缓存工具使用本轮记录的官方模型文件校验值，全部匹配后才建立固定 revision 的缓存。魔搭 `master` 后续若发生变化，校验会失败；不要将新的模型文件当成本轮版本。工具优先创建硬链接，跨磁盘时使用符号链接，此时需保留原模型目录。

来源：[ModelScope 官方下载命令](https://github.com/modelscope/modelscope/blob/master/docs/source/command.md)。

若无法直接访问 Hugging Face 数据集，可以使用 AutoDL 内置的学术资源加速或 `HF_ENDPOINT=https://hf-mirror.com`；选择一条可用线路并确认缓存。镜像只负责传输，固定 revision 与文件校验值用于核对数据身份。

```bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export MLLM_BACKEND=transformers
export MLLM_LOAD_IN_4BIT=true
export MLLM_COMPUTE_DTYPE=bfloat16
export MLLM_MODEL_REVISION=66285546d2b821cf421d4f5eb2576359d3770cd3
python scripts/prepare_cord.py --split validation \
  --revision 7f0115a4b758a71d6473b8d085751692da2fef98
python evaluate.py --dataset data/cord/validation/annotations.jsonl --limit 20 \
  --output artifacts/runs/cord-validation-nf4
```

数据已在 HF 缓存时，可添加 `--offline`，避免再次联网查询；必须明确提供 40 位的 revision。模型缓存就绪后可以设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`。

本轮在验证集上排查了 FP16 的无效输出，改为 BF16 计算，并修正货币前缀与税额的提取要求。BF16 需要 GPU 支持，程序会检查。Qwen 模型卡使用 `torch_dtype="auto"` 或 BF16 的示例；不要把所有 16 位格式视为相同的数值设置。

来源：[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。

在验证集检查错误并固定配置后，运行测试对比。以下两个运行只改权重量化方式，计算精度都为 BF16：

```bash
python scripts/prepare_cord.py --split test \
  --revision 7f0115a4b758a71d6473b8d085751692da2fef98
export MLLM_LOAD_IN_4BIT=true
python evaluate.py --dataset data/cord/test/annotations.jsonl \
  --output artifacts/runs/cord-test-nf4
export MLLM_LOAD_IN_4BIT=false
python evaluate.py --dataset data/cord/test/annotations.jsonl \
  --output artifacts/runs/cord-test-bf16
python compare_runs.py artifacts/runs/cord-test-nf4/run.json \
  artifacts/runs/cord-test-bf16/run.json
```

测试集 100 个原始样本中，4 个金额标注无法转换为单一非空文本，1 个没有本任务有效金额标签，因此最终评测 95 个；详见 `docs/BENCHMARK.md` 和数据清单。模型 revision、计算精度与数据 revision 均保存在运行记录。

来源：[AutoDL 学术资源加速](https://www.autodl.com/docs/network_turbo/)。

## 保存结果与关机

把运行目录和数据清单下载到本地后，在控制台确认实例关机。平台也支持 `/usr/bin/shutdown`，但不要在尚未下载结果时提前关机。停止 GPU 后仍需核对任何额外存储的计费。

来源：[AutoDL 省钱说明](https://www.autodl.com/docs/save_money/)。
