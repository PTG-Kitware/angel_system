from threading import Event, Lock, Thread

# !!! FOR SOME REASON CURRENTLY UNKNOWN !!!
# An import of MMPose must happen first before any detectron2 imports.
# Otherwise, something about the ROS2 node handle creation fails with a
# double-free (no idea why...).
# No this import is not being used in this file. Yes it has to be here.
# Maybe there is something more specific in this import chain that is what is
# really necessary, but the investigation is at diminishing returns...
import mmpose.apis

from angel_system.utils.event import WaitAndClearEvent
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, Quaternion
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node, ParameterDescriptor, Parameter
from sensor_msgs.msg import Image
from tcn_hpl.data.utils.pose_generation.generate_pose_data import (
    DETECTION_CLASSES,
    PosesGenerator,
)

from angel_msgs.msg import (
    ObjectDetection2dSet,
    HandJointPosesUpdate,
    HandJointPose,
    ActivityDetection,
)
from angel_utils import declare_and_get_parameters, RateTracker  # , DYNAMIC_TYPE
from angel_utils import make_default_main


BRIDGE = CvBridge()


class PoseEstimator(Node):
    """
    ROS node that runs the pose estimation model and outputs
    `ObjectDetection2dSet` and 'JointKeypoints' messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        print("getting logger")
        log = self.get_logger()

        # Inputs
        print("getting params")
        param_values = declare_and_get_parameters(
            self,
            [
                ##################################
                # Required parameter (no defaults)
                ("image_topic",),
                ("det_topic",),
                ("pose_topic",),
                ("det_net_checkpoint",),
                ("pose_net_checkpoint",),
                ("det_config",),
                ("pose_config",),
                ##################################
                # Defaulted parameters
                ("det_conf_threshold", 0.75),  # object confidence threshold
                ("keypoint_conf_threshold", 0.75),  # keypoint confidence threshold
                ("cuda_device_id", 0),  # cuda device: ID int or CPU
                # Runtime thread checkin heartbeat interval in seconds.
                ("rt_thread_heartbeat", 0.1),
                # If we should enable additional logging to the info level
                # about when we receive and process data.
                ("enable_time_trace_logging", False),
                ("image_resize", False),
                ("image_source_time_threshold", 200),
            ],
        )
        self._image_topic = param_values["image_topic"]
        self._det_topic = param_values["det_topic"]
        self._pose_topic = param_values["pose_topic"]
        self.det_model_ckpt_fp = param_values["det_net_checkpoint"]
        self.pose_model_ckpt_fp = param_values["pose_net_checkpoint"]
        self.det_config = param_values["det_config"]
        self.pose_config = param_values["pose_config"]

        self._ensure_image_resize = param_values["image_resize"]

        self._det_conf_thresh = param_values["det_conf_threshold"]
        self._keypoint_conf_thresh = param_values["keypoint_conf_threshold"]
        self._cuda_device_id = param_values["cuda_device_id"]

        self._image_source_time_threshold = param_values["image_source_time_threshold"]

        print("finished setting params")

        print("Initializing pose models...")
        # Encapsulates detection and pose models.
        self.pose_gen = PosesGenerator(
            det_config_file=self.det_config,
            pose_config_file=self.pose_config,
            det_confidence_threshold=self._det_conf_thresh,
            keypoint_confidence_threshold=self._keypoint_conf_thresh,
            det_model_ckpt=self.det_model_ckpt_fp,
            det_model_device=self._cuda_device_id,
            pose_model_ckpt=self.pose_model_ckpt_fp,
            pose_model_device=self._cuda_device_id,
        )
        print("Initializing pose models... Done")
        self.keypoints_cats = [
            v
            for _, v in sorted(self.pose_gen.pose_dataset_info.keypoint_id2name.items())
        ]
        # Pose estimates considered by this node is constrained to that of the
        # patient class specifically.
        self.patient_class_idx: int = DETECTION_CLASSES.index("patient")

        self._enable_trace_logging = param_values["enable_time_trace_logging"]

        # Single slot for latest image message to process detection over.
        self._cur_image_msg: Image = None
        self._cur_image_msg_lock = Lock()

        print("creating subscription to image topic")
        # Initialize ROS hooks
        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        print("creating publisher to detections")
        self.patient_det_publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        print("creating publisher for poses")
        self.patient_pose_publisher = self.create_publisher(
            HandJointPosesUpdate,
            self._pose_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

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

                img_source_time = image.header.stamp  # store the image timestamp
                # compare the image timestamp to the current time to see if this is a bagged image
                curr_time = self.get_clock().now().to_msg()
                # if it is too old, change the image timestamp to the current time (before processing)
                if (
                    curr_time.sec - img_source_time.sec
                ) > self._image_source_time_threshold:
                    # this must have been a bagged image - use current time (before processing)
                    img_source_time = curr_time

                if enable_trace_logging:
                    log.info(f"[rt-loop] Processing image TS={img_source_time}")
                # Convert ROS img msg to CV2 image
                img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")

                if self._ensure_image_resize:
                    img0 = cv2.resize(
                        img0, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC
                    )

                # print(f"img0: {img0.shape}")
                # height, width, chans = img0.shape

                boxes, scores, classes, keypoints = self.pose_gen.predict_single(img0)

                # Select keypoints for most confidence box among those with
                # keypoints.
                patient_results = np.argwhere(classes == self.patient_class_idx)
                if patient_results.size > 0:
                    p_scores = scores[patient_results]
                    best_idx = patient_results[np.argmax(p_scores)][0]
                    assert (
                        keypoints[best_idx] is not None
                    ), "Patient class should have keypoints but None were found"
                    keypoints = [keypoints[best_idx]]
                else:
                    keypoints = []

                all_poses_msg = HandJointPosesUpdate()
                # note: setting metdata right before publishing below

                # at most, we have 1 set of keypoints for 1 "best" patient
                for keypoints_ in keypoints:
                    for label, keypoint in zip(self.keypoints_cats, keypoints_):
                        position = Point()
                        position.x = float(keypoint[0])
                        position.y = float(keypoint[1])
                        position.z = float(
                            keypoint[2]
                        )  # This is actually the confidence score

                        # Extract the orientation
                        orientation = Quaternion()
                        orientation.x = float(0)
                        orientation.y = float(0)
                        orientation.z = float(0)
                        orientation.w = float(0)

                        # Form the geometry pose message
                        pose_msg = Pose()
                        pose_msg.position = position
                        pose_msg.orientation = orientation

                        # Create the hand joint pose message
                        joint_msg = HandJointPose()
                        joint_msg.joint = label
                        joint_msg.pose = pose_msg
                        all_poses_msg.joints.append(joint_msg)

                    # set the header metadata right before publishing to ensure the correct time
                    all_poses_msg.header.frame_id = image.header.frame_id
                    all_poses_msg.source_stamp = img_source_time
                    all_poses_msg.hand = "patient"
                    all_poses_msg.header.stamp = self.get_clock().now().to_msg()
                    self.patient_pose_publisher.publish(all_poses_msg)

                self._rate_tracker.tick()
                log.info(
                    f"Pose Estimation Rate: {self._rate_tracker.get_rate_avg()} Hz, Poses: {len(keypoints)}, pose message: {all_poses_msg}",
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
print("executing make_default_main")
main = make_default_main(PoseEstimator, multithreaded_executor=3)


if __name__ == "__main__":
    print("executing main")
    # node = PoseEstimator()
    # print(f"before main: {node}")
    main()
