# PyTorch Benchmark

`benchmark_torch.py` 使用 `torch.cuda.Event`、warmup 和同步，分开测 `torch_original` 与 `torch_one2one_only`，输出 JSON/CSV、P50/P90/P95/P99、FPS 和峰值显存。

`benchmark_end_to_end.py` 使用 100 张固定真实图像清单，以 batch=1 测量从 JPEG 路径输入到解码后的 `Results` 的 wall time，并同时汇总框架提供的 preprocess/inference/postprocess 分量。正式结论使用训练后 checkpoint、1000 次迭代和三轮独立运行；当前短时合成结果仅为执行链路 smoke。
