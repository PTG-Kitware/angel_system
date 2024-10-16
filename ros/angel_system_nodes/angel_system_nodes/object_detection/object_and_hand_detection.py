from pathlib import Path
from threading import Event, Lock, Thread
from typing import Union

from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node, ParameterDescriptor, Parameter
from sensor_msgs.msg import Image

from yolov7.detect_ptg import load_model, predict_image
from angel_system.object_detection.yolov8_detect import predict_hands
from yolov7.models.experimental import attempt_load
import yolov7.models.yolo
from yolov7.utils.torch_utils import TracedModel
from ultralytics import YOLO as YOLOv8

from angel_system.utils.event import WaitAndClearEvent
from angel_system.utils.simple_timer import SimpleTimer

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import declare_and_get_parameters, RateTracker  # , DYNAMIC_TYPE
from angel_utils import make_default_main


BRIDGE = CvBridge()


class ObjectAndHandDetector(Node):
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
                ("object_net_checkpoint",),
                ("hand_net_checkpoint",),
                ##################################
                # Defaulted parameters
                ("inference_img_size", 768),  # inference size (pixels)
                ("det_conf_threshold", 0.2),  # object confidence threshold
                ("iou_threshold", 0.35),  # IOU threshold for NMS
                ("cuda_device_id", 0),  # cuda device: ID int or CPU
                ("no_trace", True),  # don`t trace model
                ("agnostic_nms", False),  # class-agnostic NMS
                # Runtime thread checkin heartbeat interval in seconds.
                ("rt_thread_heartbeat", 0.1),
                # If we should enable additional logging to the info level
                # about when we receive and process data.
                ("enable_time_trace_logging", False),
                # Name of the background class to check for.
                ("object_background_class_name", "background"),
            ],
        )
        self._image_topic = param_values["image_topic"]
        self._det_topic = param_values["det_topic"]

        self._object_model_ckpt_fp = Path(param_values["object_net_checkpoint"])
        self._hand_model_chpt_fp = Path(param_values["hand_net_checkpoint"])

        self._inference_img_size = param_values["inference_img_size"]
        self._det_conf_thresh = param_values["det_conf_threshold"]
        self._iou_thr = param_values["iou_threshold"]
        self._cuda_device_id = param_values["cuda_device_id"]
        self._no_trace = param_values["no_trace"]
        self._agnostic_nms = param_values["agnostic_nms"]

        self._enable_trace_logging = param_values["enable_time_trace_logging"]

        ##########################################
        # Object Model
        self.object_model: Union[yolov7.models.yolo.Model, TracedModel]
        if not self._object_model_ckpt_fp.is_file():
            raise ValueError(
                f"Model checkpoint file did not exist: {self._object_model_ckpt_fp}"
            )
        (self.device, self.object_model, self.stride, self.imgsz) = load_model(
            str(self._cuda_device_id),
            self._object_model_ckpt_fp,
            self._inference_img_size,
        )
        log.info(
            f"Loaded object model with classes:\n"
            + "\n".join(f'\t- "{n}"' for n in self.object_model.names)
        )

        if not self._no_trace:
            self.object_model = TracedModel(
                self.object_model, self.device, self._inference_img_size
            )

        self.half = half = (
            self.device.type != "cpu"
        )  # half precision only supported on CUDA
        if half:
            self.object_model.half()  # to FP16

        # The object detection model may have a background class. We do not
        # want to publish that in our communication object detections message
        # as this is implied by the lack of an object or otherwise low
        # confidence class prediction. When formatting our output, we need to
        # discount the index positions for objects *after* the background
        # class.
        self._object_model_background_class_name = param_values[
            "object_background_class_name"
        ]
        self._object_model_background_idx = None
        try:
            self._object_model_background_idx = self.object_model.names.index(
                self._object_model_background_class_name
            )
            log.info(
                f"Background class identified @ index: {self._object_model_background_idx}"
            )
        except ValueError:
            # No background class name in the model's name list. This is
            # acceptable, but we'll just warn about it anyway.
            log.warn(
                f'No background class "{self._object_model_background_class_name}" '
                f"in object model names list: {self.object_model.names}"
            )

        log.info("Object Detector initialized")

        ##########################################
        # Hand model
        self.hand_model = YOLOv8(self._hand_model_chpt_fp)
        log.info(
            f"Loaded hand model with classes:\n"
            + "\n".join(f'\t- "{n}"' for n in self.hand_model.names)
        )

        log.info("Hand Detector initialized")

        ##########################################
        # Other stateful properties

        # Single slot for latest image message to process detection over.
        # This should be written by the image topic subscriber callback, and
        # read from the runtime loop once per iteration.
        self._cur_image_msg: Image = None
        self._cur_image_msg_lock = Lock()

        self._rate_tracker = RateTracker()

        ##########################################
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

        ##########################################
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

        # Create object class label vector for later inclusion in published
        # messages.
        if self._object_model_background_idx is not None:
            label_vector = self.object_model.names[1:]  # remove background label
        else:
            label_vector = self.object_model.names

        label_vector.append("hand (left)")
        label_vector.append("hand (right)")
        n_classes = len(label_vector)

        left_hand_cid = n_classes - 2
        right_hand_cid = n_classes - 1

        hand_cid_label_dict = {
            "hand (right)": right_hand_cid,
            "hand (left)": left_hand_cid,
        }
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
                print(f"img0 type: {type(img0)}")

                msg = ObjectDetection2dSet()
                n_dets = 0
                dflt_conf_vec = np.zeros(n_classes, dtype=np.float64)

                # Detect hands
                # with SimpleTimer("predict_hands", log_func=log.info):
                hand_boxes, hand_labels, hand_confs = predict_hands(
                    hand_model=self.hand_model,
                    img0=img0,
                    device=self.device,
                    imgsz=self._inference_img_size,
                )

                hand_classids = [hand_cid_label_dict[label] for label in hand_labels]

                # Detect objects
                # with SimpleTimer("predict_objects", log_func=log.info):
                object_boxes, object_confs, object_classids = predict_image(
                    img0,
                    self.device,
                    self.object_model,
                    self.stride,
                    self.imgsz,
                    self.half,
                    False,
                    self._det_conf_thresh,
                    self._iou_thr,
                    None,
                    self._agnostic_nms,
                )
                # If we have a background class, decrement the class IDs by one
                # that are after the background class index.
                if self._object_model_background_idx is not None:
                    # This is a little faster than doing a naive list
                    # comprehension.
                    id_arr = np.asarray(object_classids)
                    id_arr[id_arr > self._object_model_background_idx] -= 1
                    object_classids = id_arr.tolist()

                object_boxes.extend(hand_boxes)
                object_confs.extend(hand_confs)
                object_classids.extend(hand_classids)
                for xyxy, conf, cls_id in zip(
                    object_boxes, object_confs, object_classids
                ):
                    n_dets += 1
                    msg.left.append(xyxy[0])
                    msg.top.append(xyxy[1])
                    msg.right.append(xyxy[2])
                    msg.bottom.append(xyxy[3])

                    # Object and Hand detection class prediction output only
                    # consists of a single confidence value for the
                    # most-confident non-background class. Here we slot that
                    # confidence into the appropriate index of a zeroed array
                    # and then extend the output messages confidence matrix
                    # (flattened). The temporary confidence vector is then
                    # zeroed out again in preparation for the next loop.
                    dflt_conf_vec[cls_id] = conf
                    # This extend copies data into msg.label_confidences array
                    msg.label_confidences.extend(dflt_conf_vec)
                    # Reset before next passthrough
                    dflt_conf_vec[cls_id] = 0.0

                msg.num_detections = n_dets

                # set header metadata before publishing to get the correct publish time
                msg.header.frame_id = image.header.frame_id
                msg.source_stamp = image.header.stamp
                msg.label_vec[:] = label_vector
                msg.header.stamp = self.get_clock().now().to_msg()
                self._det_publisher.publish(msg)

                self._rate_tracker.tick()
                log.info(
                    f"Objects Detection Rate: {self._rate_tracker.get_rate_avg()} Hz, Num objects detected: {n_dets}\nnum of hands: {len(hand_boxes)}, other objects: {n_dets - len(hand_boxes)}]"
                )
                log.info(f"msg: {msg}")

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
main = make_default_main(ObjectAndHandDetector, multithreaded_executor=3)


if __name__ == "__main__":
    main()
