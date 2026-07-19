# Training

训练门禁顺序为：数据 `READY` 与质检通过 → 128/32 固定清单的 1–3 epoch smoke → 32 图小集过拟合 → full baseline。`create_carfusion_lists.py` 同时保存清单 SHA-256 和 100 张固定测试图，避免用不稳定的 `fraction` 隐式抽样。

脚本固定 seed、关闭未确认的水平翻转，并把结果写入 `vehicle_pose_runs/`。官方检测权重以名称和 shape 交集加载，日志必须保留 transferred/missing 信息。Validator 自动保存 bbox/OKS 辅助指标以及 vehicle PCK/NME 的配对和汇总文件。
