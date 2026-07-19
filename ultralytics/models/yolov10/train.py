from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from .val import YOLOv10DetectionValidator, YOLOv10PoseValidator
from ultralytics.nn.tasks import YOLOv10DetectionModel, YOLOv10PoseModel
from copy import copy
from ultralytics.utils import RANK

class YOLOv10DetectionTrainer(DetectionTrainer):
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_om", "cls_om", "dfl_om", "box_oo", "cls_oo", "dfl_oo", 
        return YOLOv10DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = YOLOv10DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


class YOLOv10PoseTrainer(PoseTrainer):
    """Train YOLOv10 pose with one-to-many and one-to-one keypoint losses."""

    def get_validator(self):
        self.loss_names = (
            "box_om",
            "pose_om",
            "visibility_om",
            "cls_om",
            "dfl_om",
            "box_oo",
            "pose_oo",
            "visibility_oo",
            "cls_oo",
            "dfl_oo",
        )
        return YOLOv10PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOv10PoseModel(
            cfg,
            nc=self.data["nc"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model
