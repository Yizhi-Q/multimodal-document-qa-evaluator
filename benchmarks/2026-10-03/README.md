# 2026-10-03 实验存档

- `cord-test-nf4`、`cord-test-bf16`：固定配置后的两次完整真实测试，同一批 95 张票据、319 个标注金额字段。
- `cord-validation-nf4`：最终提示词与 BF16 计算的 20 张验证实验。
- 名称含 `initial` 或 `diagnostic` 的目录：验证阶段的旧配置与排错尝试，保留失败记录。
- `comparison.json`：从原始测试记录计算的准确率、耗时与已分配显存摘要。
- `*-manifest.json`：数据版本、源文件哈希、纳入与排除清单。
- `*-annotations.jsonl`：记录各阶段的提取问题、字段 Schema 与标准答案；仅这几个文件不能直接重新评测，因为完整图片不在存档中。
- `model-integrity.json`、`canonical-metadata.json`：下载文件与官方固定版本的对应关系。
- `validation-review-images`、`validation-review.csv`：五张公开验证图片及逐字段人工复核。

查看结果：[实测报告](../../docs/BENCHMARK.md)、[验证排错记录](../../docs/VALIDATION_REVIEW.md)。

复现时按 [AutoDL 说明](../../docs/AUTODL_RUN.md) 使用固定数据 revision 重新导出图片与标注，按记录的模型版本和参数运行；完整模型与 parquet 不随项目包分发。图片哈希会受到图像导出方式及 Pillow 版本影响，需保持记录的导出环境。

公开数据遵循 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，来源和转换见 [ATTRIBUTION.md](ATTRIBUTION.md)。
