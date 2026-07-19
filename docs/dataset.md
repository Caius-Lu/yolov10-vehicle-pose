# Dataset

本次使用用户明确提供的 OpenDataLab CarFusion 包；来源路径、SHA-256 和传输/解压证据见 `reports/data/`。归档是外层 gzip/tar + 内层 `carfusion.tar` 两层结构，必须在远端哈希一致后解压。

转换器接收合并后的官方式 COCO Keypoints JSON，要求每实例 14×3，输出 YOLO pose 格式。`v=0` 无效，`v=1` 遮挡，`v=2` 可见；1/2 均参与坐标损失。源数据的 739 个 `[0,0,0,0]` bbox 只在显式 `drop` 策略下丢弃并保留 ID 记录。

相机后缀 1/2 按同一路口合组：train 为 butler/craig/morewood，val 为 fifth，官方 penn 保留为 test。左右索引未人工确认，因此水平翻转关闭。`run_carfusion_pipeline.sh` 一次完成两层解压、合并、转换、质检、100 图预览和固定清单。
