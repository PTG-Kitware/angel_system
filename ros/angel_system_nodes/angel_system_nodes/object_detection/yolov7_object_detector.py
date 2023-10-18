from pathlib import Path
import time

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

from yolov7.detect_ptg import load_model, preprocess_bgr_img
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
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
                ("net_weights",),
                ("net_config",),
                ############################
                # Defaulted hyper-parameters
                ("inference_img_size", 1280),   # inference size (pixels)
                ("det_conf_threshold", 0.7),    # object confidence threshold
                ("iou_threshold", 0.45),        # IOU threshold for NMS
                ("cuda_device_id", "0"),        # cuda device, i.e. 0 or 0,1,2,3 or cpu
                ("no_trace", True),             # don`t trace model
                ("agnostic_nms", False),        # class-agnostic NMS
            ]
        )
        self._image_topic = param_values['image_topic']
        self._det_topic = param_values['det_topic']
        self._model_weights_fp = Path(param_values['net_weights'])
        self._model_config_fp = Path(param_values['net_config'])

        self._inference_img_size = param_values['inference_img_size']
        self._det_conf_thresh = param_values['det_conf_threshold']
        self._iou_thr = param_values['iou_threshold']
        self._cuda_device_id = param_values['cuda_device_id']
        self._no_trace = param_values['no_trace']
        self._agnostic_nms = param_values['agnostic_nms']

        # Model
        (
            self.device,
            self.model,
            self.stride,
            self.imgsz,
        ) = load_model(
            str(self._cuda_device_id),
            self._model_weights_fp,
            self._inference_img_size
        )

        self._pred_rate_tracker = RateTracker()

        # Initialize ROS hooks
        self._subscription_cb_group = MutuallyExclusiveCallbackGroup()
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1,
            callback_group=self._subscription_cb_group,
        )
        self._publisher_cb_group = MutuallyExclusiveCallbackGroup()
        self._det_publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1,
            callback_group=self._publisher_cb_group,
        )

        if not self._no_trace:
            self.model = TracedModel(self.model, self.device, self._inference_img_size)

        self.half = half = self.device.type != 'cpu'  # half precision only supported on CUDA
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

        t_start = time.monotonic()

        # Convert ROS img msg to CV2 image
        img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")
        img = preprocess_bgr_img(img0, self.imgsz, self.stride, self.device, self.half)
        t_end_img = time.monotonic()

        # Predict
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        t_end_pred = time.monotonic()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self._det_conf_thresh,
            self._iou_thr,
            classes=[],
            agnostic=self._agnostic_nms
        )
        t_end_nms = time.monotonic()

        t_end = time.monotonic()
        # log.info(
        #     "\n"
        #     f"Start      --> Prediction: {t_end_pred - t_start}\n"
        #     f"Prediction -->        NMS: {t_end_nms - t_end_pred}\n"
        #     f"Total      -->           : {t_end - t_start}",
        #     throttle_duration_sec=1,
        # )

        self._pred_rate_tracker.tick()
        log.info(
            f"Objects predicted (hz: "
            f"{self._pred_rate_tracker.get_rate_avg()})",
            throttle_duration_sec=1,
        )

        # if pred is not None:
        #     # Publish detection set message
        #     self.publish_det_message(pred, image.header)

    def publish_det_message(self, pred, image_header):
        """
        Forms and sends a `angel_msgs/ObjectDetection2dSet` message
        """
        message = ObjectDetection2dSet()

        # Populate message header
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = image_header.frame_id
        message.source_stamp = image_header.stamp

        # Load bboxes
        message.label_vec = []
        label_confidences = []

        message.left = []
        message.right = []
        message.top = []
        message.bottom = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if not len(det):
                continue

            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls_id in reversed(det):
                message.label_vec.append(cls_id)
                label_confidences.append(conf)

                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                norm_xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                cxywh = [norm_xywh[0] * width, norm_xywh[1] * height,
                        norm_xywh[2] * width, norm_xywh[3] * height] # center xy, wh
                xywh = [cxywh[0] - (cxywh[2] / 2), cxywh[1] - (cxywh[3] / 2),
                        cxywh[2], cxywh[3]]

                tlbr = kwimage.Boxes([xywh], "xywh").toformat("tlbr").data[0][0].tolist()
                tl_x, tl_y, br_x, br_y = tlbr
                message.left.append(tl_x)
                message.right.append(br_x)
                message.top.append(tl_y)
                message.bottom.append(br_y)

        message.num_detections = len(message.label_vec)

        if message.num_detections == 0:
            self.get_logger().debug("No detections, nothing to publish")
            self._det_publisher.publish(message)
            return

        message.label_confidences = (
            np.asarray(label_confidences, dtype=np.float64).ravel().tolist()
        )

        # Publish
        self._det_publisher.publish(message)
        self._rate_tracker.tick()
        self.get_logger().info(
            f"Published det] message (hz: " f"{self._rate_tracker.get_rate_avg()})",
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
