from pathlib import Path
from typing import Union

from cv_bridge import CvBridge
import kwimage
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch

from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import RateTracker

from yolov7.detect_ptg import load_model, preprocess_bgr_img, predict_image
from yolov7.models.experimental import attempt_load
import yolov7.models.yolo
from yolov7.utils.general import (
    check_img_size,
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
)
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox

from angel_utils import declare_and_get_parameters


BRIDGE = CvBridge()


class YoloObjectDetector(Node):
    """
    ROS node that runs the yolov7 object detector model and outputs
    `ObjectDetection2dSet` messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        # Inputs
        param_values = declare_and_get_parameters(
            self,
            [
                ##################################
                # Required parameter (no defaults)
                ("image_topic",),
                ("det_topic",),
                ("net_checkpoint",),
                ##################################
                # Defaulted parameters
                ("inference_img_size", 1280),  # inference size (pixels)
                ("det_conf_threshold", 0.7),  # object confidence threshold
                ("iou_threshold", 0.45),  # IOU threshold for NMS
                ("cuda_device_id", "0"),  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                ("no_trace", True),  # don`t trace model
                ("agnostic_nms", False),  # class-agnostic NMS
            ],
        )
        self._image_topic = param_values["image_topic"]
        self._det_topic = param_values["det_topic"]
        self._model_ckpt_fp = Path(param_values["net_checkpoint"])

        self._inference_img_size = param_values["inference_img_size"]
        self._det_conf_thresh = param_values["det_conf_threshold"]
        self._iou_thr = param_values["iou_threshold"]
        self._cuda_device_id = param_values["cuda_device_id"]
        self._no_trace = param_values["no_trace"]
        self._agnostic_nms = param_values["agnostic_nms"]

        # Model
        self.model: Union[yolov7.models.yolo.Model, TracedModel]
        (self.device, self.model, self.stride, self.imgsz) = load_model(
            str(self._cuda_device_id), self._model_ckpt_fp, self._inference_img_size
        )

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._det_publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        if not self._no_trace:
            self.model = TracedModel(self.model, self.device, self._inference_img_size)

        self.half = half = (
            self.device.type != "cpu"
        )  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16

        self._rate_tracker = RateTracker()
        log.info("Detector initialized")

    def listener_callback(self, image):
        """
        Callback function for image messages. Runs the berkeley object detector
        on the image and publishes an ObjectDetectionSet2d message for the image.
        """
        log = self.get_logger()

        # Convert ROS img msg to CV2 image
        img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")
        height, width = img0.shape[:2]

        # img = preprocess_bgr_img(img0, self.imgsz, self.stride, self.device, self.half)
        # t_end_img = time.monotonic()
        #
        # # Predict
        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        #     pred = self.model(img, augment=False)[0]
        # t_end_pred = time.monotonic()
        #
        # # Apply NMS
        # pred_nms = non_max_suppression(
        #     pred,
        #     self._det_conf_thresh,
        #     self._iou_thr,
        #     classes=None,
        #     agnostic=self._agnostic_nms
        # )
        # t_end_nms = time.monotonic()

        msg = ObjectDetection2dSet()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = image.header.frame_id
        msg.source_stamp = image.header.stamp
        msg.label_vec[:] = self.model.names

        n_classes = len(self.model.names)
        n_dets = 0

        dflt_conf_vec = np.zeros(n_classes, dtype=np.float64)

        for xyxy, conf, cls_id in predict_image(
            img0,
            self.device,
            self.model,
            self.stride,
            self.imgsz,
            self.half,
            False,
            self._det_conf_thresh,
            self._iou_thr,
            None,
            self._agnostic_nms,
        ):
            n_dets += 1
            msg.left.append(xyxy[0])
            msg.top.append(xyxy[1])
            msg.right.append(xyxy[2])
            msg.bottom.append(xyxy[3])

            dflt_conf_vec[cls_id] = conf
            msg.label_confidences.extend(dflt_conf_vec)  # copies data into array
            dflt_conf_vec[cls_id] = 0.0  # reset before next passthrough

        msg.num_detections = n_dets

        # EXAMPLE: Getting max conf labels for each detection:
        # np.asarray(msg.label_vec)[
        #   np.asarray(msg.label_confidences).reshape(msg.num_detections, len(msg.label_vec)).argmax(axis=1)
        # ]

        self._det_publisher.publish(msg)

        self._rate_tracker.tick()
        log.info(
            f"Objects predicted & published (hz: "
            f"{self._rate_tracker.get_rate_avg()})",
            throttle_duration_sec=1,
        )


def main():
    rclpy.init()

    node = YoloObjectDetector()

    # Don't really want to use *all* available threads...
    # 5 threads because:
    # - 3 known subscribers which have their own groups
    # - 1 for default group
    # - 1 for publishers
    executor = MultiThreadedExecutor(num_threads=5)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().debug("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
