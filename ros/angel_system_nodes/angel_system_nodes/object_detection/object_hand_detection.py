from pathlib import Path
from threading import Event, Lock, Thread
from typing import Union

from cv_bridge import CvBridge
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node, ParameterDescriptor, Parameter
from sensor_msgs.msg import Image

from yolov7.detect_ptg import load_model, predict_image, predict_hands
from yolov7.models.experimental import attempt_load
import yolov7.models.yolo
from yolov7.utils.torch_utils import TracedModel

from angel_system.utils.event import WaitAndClearEvent
from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import declare_and_get_parameters, RateTracker, DYNAMIC_TYPE
from angel_utils import make_default_main


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
                ("cuda_device_id", 0, DYNAMIC_TYPE),  # cuda device: ID int or CPU
                ("no_trace", True),  # don`t trace model
                ("agnostic_nms", False),  # class-agnostic NMS
                # Runtime thread checkin heartbeat interval in seconds.
                ("rt_thread_heartbeat", 0.1),
                # If we should enable additional logging to the info level
                # about when we receive and process data.
                ("enable_time_trace_logging", False),
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

        self._enable_trace_logging = param_values["enable_time_trace_logging"]

        # Model
        self.model: Union[yolov7.models.yolo.Model, TracedModel]
        if not self._model_ckpt_fp.is_file():
            raise ValueError(
                f"Model checkpoint file did not exist: {self._model_ckpt_fp}"
            )
        (self.device, self.model, self.stride, self.imgsz) = load_model(
            str(self._cuda_device_id), self._model_ckpt_fp, self._inference_img_size
        )
        log.info(
            f"Loaded model with classes:\n"
            + "\n".join(f'\t- "{n}"' for n in self.model.names)
        )

        # Single slot for latest image message to process detection over.
        self._cur_image_msg: Image = None
        self._cur_image_msg_lock = Lock()

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

        # Create and start detection runtime thread and loop.
        log.info("Starting runtime thread...")
        # On/Off Switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = param_values["rt_thread_heartbeat"]
        # Condition that the runtime should perform processing
        self._rt_awake_evt = WaitAndClearEvent()
        self._rt_thread = Thread(target=self.rt_loop, name="prediction_runtime")
        self._rt_thread.daemon = True
        self._rt_thread.start()

    def listener_callback(self, image: Image):
        """
        Callback function for image messages. Runs the berkeley object detector
        on the image and publishes an ObjectDetectionSet2d message for the image.
        """
        log = self.get_logger()
        if self._enable_trace_logging:
            log.info(f"Received image with TS: {image.header.stamp}")
        with self._cur_image_msg_lock:
            self._cur_image_msg = image
            self._rt_awake_evt.set()

    def rt_alive(self) -> bool:
        """
        Check that the prediction runtime is still alive and raise an exception
        if it is not.
        """
        alive = self._rt_thread.is_alive()
        if not alive:
            self.get_logger().warn("Runtime thread no longer alive.")
            self._rt_thread.join()
        return alive

    def rt_stop(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()

    def rt_loop(self):
        log = self.get_logger()
        log.info("Runtime loop starting")
        enable_trace_logging = self._enable_trace_logging

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait_and_clear(self._rt_active_heartbeat):
                with self._cur_image_msg_lock:
                    if self._cur_image_msg is None:
                        continue
                    image = self._cur_image_msg
                    self._cur_image_msg = None

                if enable_trace_logging:
                    log.info(f"[rt-loop] Processing image TS={image.header.stamp}")
                # Convert ROS img msg to CV2 image
                img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")
                
                print(f"img0: {img0.shape}")
                width, height = self._inference_img_size

                msg = ObjectDetection2dSet()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = image.header.frame_id
                msg.source_stamp = image.header.stamp
                msg.label_vec[:] = self.model.names

                n_classes = len(self.model.names) + 2 # accomedate 2 hands
                n_dets = 0

                dflt_conf_vec = np.zeros(n_classes, dtype=np.float64)
                right_hand_cid = n_classes - 2
                left_hand_cid = n_classes - 1

                hands_preds = predict_hands(hand_model=self.hand_model, img0=img0, 
                                            img_size=self._inference_img_size, device=self.device)

                hand_centers = [center.xywh.tolist()[0][0] for center in hands_preds.boxes][:2]
                hands_label = []
                if len(hand_centers) == 2:
                    if hand_centers[0] > hand_centers[1]:
                        hands_label.append(right_hand_cid)
                        hands_label.append(left_hand_cid)
                    elif hand_centers[0] <= hand_centers[1]:
                        hands_label.append(left_hand_cid)
                        hands_label.append(right_hand_cid)
                elif len(hand_centers) == 1:
                    if hand_centers[0] > width//2:
                        hands_label.append(right_hand_cid)
                    elif hand_centers[0] <= width//2:
                        hands_label.append(left_hand_cid)
                
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
                    # copies data into array
                    msg.label_confidences.extend(dflt_conf_vec)
                    # reset before next passthrough
                    dflt_conf_vec[cls_id] = 0.0

                msg.num_detections = n_dets

                self._det_publisher.publish(msg)

                self._rate_tracker.tick()
                log.info(
                    f"Objects Detection Rate: {self._rate_tracker.get_rate_avg()} Hz",
                )

    def destroy_node(self):
        print("Stopping runtime")
        self.rt_stop()
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        super().destroy_node()


# Don't really want to use *all* available threads...
# 3 threads because:
# - 1 known subscriber which has their own group
# - 1 for default group
# - 1 for publishers
main = make_default_main(YoloObjectDetector, multithreaded_executor=3)


if __name__ == "__main__":
    main()
