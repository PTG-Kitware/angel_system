import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import RateTracker
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device, TracedModel
from utils.plots import plot_one_box
from utils.datasets import letterbox

BRIDGE = CvBridge()


class YoloObjectDetector(Node):
    """
    ROS node that runs the yolov7 object detector model and outputs
    `ObjectDetection2dSet` messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Inputs
        self._image_topic = (
            self.declare_parameter("image_topic", "debug/PVFrames")
            .get_parameter_value()
            .string_value
        )
        self._det_topic = (
            self.declare_parameter("det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._model_weights = (
            self.declare_parameter(
                "weights",
                "best.pth",
            )
            .get_parameter_value()
            .string_value
        )
        self._det_conf_thresh = self.declare_parameter("det_conf_threshold", 0.7).value
        self._iou_thr =  self.declare_parameter("iou_thr", 0.45).value
        self._cuda_device_id = self.declare_parameter("cuda_device_id", 0).value
        self._no_trace = self.declare_parameter("no_trace", False).value
        self._agnostic_nms = self.declare_parameter("agnostic_nms", False).value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Detections topic: {self._det_topic}")
        log.info(f"Weights file: {self._model_weights}")
        log.info(f"Detection confidence threshold: {self._det_conf_thresh}")
        log.info(f"CUDA Device ID: {self._cuda_device_id}")

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

        # Model
        self.device = select_device(self._cuda_device_ide)
        self.model = attempt_load(self._model_weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size

        if not self._no_trace:
            model = TracedModel(self.model, self.device, opt.img_size)

        half = device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16

        self.img_idx = 0
        self._rate_tracker = RateTracker()
        log.info("Detector initialized")

    def listener_callback(self, image):
        """
        Callback function for image messages. Runs the berkeley object detector
        on the image and publishes an ObjectDetectionSet2d message for the image.
        """
        self.img_idx += 1

        # Convert ROS img msg to CV2 image
        img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        s = time.time()
        # Predict
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]
        # Apply NMS
        
        pred = non_max_suppression(pred, self._det_conf_thresh, self._iou_thr, classes=[], agnostic=self._agnostic_nms)

        self.get_logger().debug(f"Detection prediction took: {time.time() - s:.6f} s")
        if pred is not None:
            # Publish detection set message
            self.publish_det_message(pred, image.header)

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
    do_multithreading = True  # TODO add this to cli args
    berkeley_obj_det = BerkeleyObjectDetector()
    if do_multithreading:
        # Don't really want to use *all* available threads...
        # 5 threads because:
        # - 3 known subscribers which have their own groups
        # - 1 for default group
        # - 1 for publishers
        executor = MultiThreadedExecutor(num_threads=5)
        executor.add_node(berkeley_obj_det)
        try:
            executor.spin()
        except KeyboardInterrupt:
            berkeley_obj_det.get_logger().debug("Keyboard interrupt, shutting down.\n")
    else:
        try:
            rclpy.spin(berkeley_obj_det)
        except KeyboardInterrupt:
            berkeley_obj_det.get_logger().debug("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    berkeley_obj_det.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
