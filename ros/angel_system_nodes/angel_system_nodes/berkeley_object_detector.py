import os
import time

import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import RateTracker

from angel_system.berkeley.demo import predictor, model


BRIDGE = CvBridge()


class BerkeleyObjectDetector(Node):
    """
    ROS node that runs the berkeley object detector model and outputs
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
        self._model_config = (
            self.declare_parameter(
              "model_config",
              "angel_system/berkeley/configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_demo.yaml")
            .get_parameter_value()
            .string_value
        )
        self._det_topic = (
            self.declare_parameter("det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._det_conf_thresh = (
            self.declare_parameter("det_conf_threshold", 0.7)
            .value
        )
        self._cuda_device_id = (
            self.declare_parameter("cuda_device_id", 0)
            .value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Model Config file: {self._model_config}")
        log.info(f"Detections topic: {self._det_topic}")
        log.info(f"Detection confidence threshold: {self._det_conf_thresh}")
        log.info(f"CUDA Device ID: {self._cuda_device_id}")

        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )
        self._det_publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1
        )

        # Step classifier
        parser = model.get_parser()

        args = parser.parse_args(
                f"--config-file {self._model_config} --confidence-threshold {self._det_conf_thresh}".split()
        )
        log.info("Arguments: " + str(args))
        cfg = model.setup_cfg(args)
        self.get_logger().info(f'cfg: {cfg}')

        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self._cuda_device_id}'
        self.demo = predictor.VisualizationDemo_add_smoothing(
            cfg,
            last_time=2,
            draw_output=False,
            tracking=False,
        )

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
        bgr_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")

        s = time.time()
        predictions, _, _ = self.demo.run_on_image_smoothing_v2(
            bgr_image, current_idx=self.img_idx)
        decoded_preds = model.decode_prediction(predictions)
        self.get_logger().info(f"Detection prediction took: {time.time() - s:.6f} s")

        if decoded_preds is not None:
            # Publish detection set message
            self.publish_det_message(decoded_preds, image.header)

    def publish_det_message(self, preds, image_header):
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

        message.label_vec = list(preds.keys())
        message.num_detections = len(message.label_vec)

        if message.num_detections == 0:
            self.get_logger().info("No detections, nothing to publish")
            self._det_publisher.publish(message)
            return

        for label, det in preds.items():
            conf_vec = np.zeros(len(message.label_vec))
            conf_vec[message.label_vec.index(label)] = det["confidence_score"]
            label_confidences.append(conf_vec)

            tl_x, tl_y, br_x, br_y = det["bbox"]
            message.left.append(tl_x)
            message.right.append(br_x)
            message.top.append(tl_y)
            message.bottom.append(br_y)

            # Add obj-obj and obj-hand info
            message.obj_obj_contact_state.append(det["obj_obj_contact_state"])
            message.obj_obj_contact_conf.append(det["obj_obj_contact_conf"])
            message.obj_hand_contact_state.append(det["obj_hand_contact_state"])
            message.obj_hand_contact_conf.append(det["obj_hand_contact_conf"])

        message.label_confidences = np.asarray(label_confidences, dtype=np.float64).ravel().tolist()

        # Publish
        self._det_publisher.publish(message)
        self._rate_tracker.tick()
        self.get_logger().info(f"Published det] message (hz: "
                               f"{self._rate_tracker.get_rate_avg()})",
                               throttle_duration_sec=1)


def main():
    rclpy.init()

    berkeley_obj_det = BerkeleyObjectDetector()

    try:
        rclpy.spin(berkeley_obj_det)
    except KeyboardInterrupt:
        berkeley_obj_det.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    berkeley_obj_det.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
