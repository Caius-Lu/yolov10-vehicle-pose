from ultralytics.models.yolo.detect import DetectionPredictor
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.models.yolo.pose import PosePredictor


class YOLOv10DetectionPredictor(DetectionPredictor):
    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        if preds.shape[-1] == 6:
            pass
        else:
            preds = preds.transpose(-1, -2)
            bboxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, preds.shape[-1]-4)
            bboxes = ops.xywh2xyxy(bboxes)
            preds = torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            mask = mask & (preds[..., 5:6] == torch.tensor(self.args.classes, device=preds.device).unsqueeze(0)).any(2)
        
        preds = [p[mask[idx]] for idx, p in enumerate(preds)]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


class YOLOv10PosePredictor(PosePredictor):
    """Construct pose Results from YOLOv10 NMS-free one-to-one predictions."""

    def postprocess(self, preds, img, orig_imgs):
        if isinstance(preds, dict):
            preds = preds["one2one"]
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if preds.shape[-1] != 6 + self.model.kpt_shape[0] * self.model.kpt_shape[1]:
            preds = ops.v10_pose_postprocess(
                preds.transpose(-1, -2), self.args.max_det, len(self.model.names), self.model.kpt_shape
            )

        mask = preds[..., 4] > self.args.conf
        if self.args.classes is not None:
            classes = torch.tensor(self.args.classes, device=preds.device)
            mask &= (preds[..., 5:6] == classes.unsqueeze(0)).any(2)
        preds = [prediction[mask[index]] for index, prediction in enumerate(preds)]

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for index, prediction in enumerate(preds):
            orig_img = orig_imgs[index]
            prediction[:, :4] = ops.xywh2xyxy(prediction[:, :4])
            prediction[:, :4] = ops.scale_boxes(img.shape[2:], prediction[:, :4], orig_img.shape)
            keypoints = prediction[:, 6:].view(len(prediction), *self.model.kpt_shape)
            keypoints = ops.scale_coords(img.shape[2:], keypoints, orig_img.shape)
            results.append(
                Results(
                    orig_img,
                    path=self.batch[0][index],
                    names=self.model.names,
                    boxes=prediction[:, :6],
                    keypoints=keypoints,
                )
            )
        return results
