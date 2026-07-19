# Architecture

`yolov10n-vehicle-pose.yaml` 复用 YOLOv10n Backbone/Neck 的 P3/P4/P5。`v10Pose` 保留双检测塔，并增加两套可配置纯卷积关键点塔。训练输出每分支 `(detection_features, keypoint_logits)`；`v10PoseLoss` 分别使用 top-k=10 与 top-k=1 assigner 的 `target_gt_idx` 聚合 Loss。推理只 decode one-to-one，并通过同一 anchor top-k 索引 gather bbox 与 14×3 keypoints。

`one2one_only=True` 仅在 eval 跳过 one-to-many detection/keypoint tower；训练行为不变。

