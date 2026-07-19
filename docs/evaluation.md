# Evaluation

检测使用 bbox precision/recall/mAP。关键点使用 PCK@0.05、PCK@0.10 与 NME，距离除以 GT bbox 长边；分别输出 visible/occluded、每关键点、small/medium/large 和有效关键点数量分组。

Validator 先以 bbox IoU≥0.5 做按 IoU、置信度稳定排序的贪心一对一配对，输出 `matched_pose_pairs.json`，再输出 `vehicle_pose_metrics.json` 并把可用标量写回训练验证结果。没有权威车辆 sigma 时，框架内 uniform sigma 的 OKS 只能标记为实验性辅助指标，不能称为标准车辆 AP。
