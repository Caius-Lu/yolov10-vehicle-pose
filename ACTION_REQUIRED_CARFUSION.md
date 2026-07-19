# CarFusion 数据前置条件记录

当前状态：`RESOLVED_BY_USER_SUPPLIED_AUTHORIZED_PACKAGE`

2026-07-17，用户明确提供本地数据路径：

`/Users/caius/Documents/yolopose/data/CarFusion/OpenDataLab___CarFusion`

因此不再等待 `download_carfusion.py`。当前门禁改为：完成可恢复传输、远端 SHA-256 一致性校验、安全解压和真实标注结构盘点。

以下是此前缺少数据时保留的授权流程记录：

1. 打开官方访问申请表：<https://forms.gle/FCUcbt3jD1hB6ja57>。
2. 获批后下载官方 `download_carfusion.py`。
3. 将文件放到 `external/carfusion_to_coco/download_carfusion.py`。
4. 不要把含 token、Cookie 或私密下载链接的脚本提交到 Git。
5. 重新执行数据下载阶段，将原始数据落到 `datasets/CarFusion/raw`。
6. 下载完成后运行 COCO 转换，再运行 `scripts/convert_carfusion.py` 和质量检查。

项目不会使用未授权来源、随机数据或人工 fixture 冒充真实 CarFusion 训练数据。
