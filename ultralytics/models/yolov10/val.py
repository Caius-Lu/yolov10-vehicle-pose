import json

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import ops
from ultralytics.utils.vehicle_pose_metrics import compute_vehicle_pose_metrics, match_pose_pairs

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            return preds
        else:
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)


class YOLOv10PoseValidator(PoseValidator):
    """Evaluate YOLOv10 one-to-one pose predictions without NMS."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.matched_pose_pairs = []

    def update_metrics(self, preds, batch):
        """Update standard metrics and retain box-matched pairs for vehicle PCK/NME."""
        super().update_metrics(preds, batch)
        for image_index, prediction in enumerate(preds):
            if not len(prediction):
                continue
            prepared = self._prepare_batch(image_index, batch)
            if not len(prepared["bbox"]):
                continue
            pred_native, pred_keypoints = self._prepare_pred(prediction, prepared)
            pairs = match_pose_pairs(
                prepared["bbox"],
                prepared["kpts"],
                pred_native[:, :4],
                pred_keypoints,
                pred_scores=pred_native[:, 4],
                iou_threshold=0.5,
            )
            image_path = str(Path(batch["im_file"][image_index]))
            for pair in pairs:
                pair["image"] = image_path
            self.matched_pose_pairs.extend(pairs)

    def get_stats(self):
        """Return standard mAP plus vehicle PCK/NME and persist traceable matched pairs."""
        stats = super().get_stats()
        metrics = compute_vehicle_pose_metrics(self.matched_pose_pairs)
        self.vehicle_pose_metrics = metrics
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "matched_pose_pairs.json").write_text(
            json.dumps(self.matched_pose_pairs, ensure_ascii=False), encoding="utf-8"
        )
        (self.save_dir / "vehicle_pose_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        scalar_mapping = {
            "metrics/PCK@0.05": metrics["pck@0.05"],
            "metrics/PCK@0.10": metrics["pck@0.10"],
            "metrics/NME": metrics["nme"],
            "metrics/PCK@0.05-visible": metrics["visible_pck@0.05"],
            "metrics/PCK@0.05-occluded": metrics["occluded_pck@0.05"],
        }
        stats.update({key: value for key, value in scalar_mapping.items() if value is not None})
        return stats

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.shape[-1] != 6 + self.kpt_shape[0] * self.kpt_shape[1]:
            preds = ops.v10_pose_postprocess(
                preds.transpose(-1, -2), self.args.max_det, self.nc, self.kpt_shape
            )
        preds = preds.clone()
        preds[..., :4] = ops.xywh2xyxy(preds[..., :4])
        return preds
