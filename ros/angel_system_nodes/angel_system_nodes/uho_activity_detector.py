from cv_bridge import CvBridge
import cv2
import message_filters as mf
import numpy as np
import rclpy
from rclpy.node import Node
import torch

from angel_msgs.msg import (
    ActivityDetection,
    HandJointPosesUpdate,
    ObjectDetection2dSet
)
from sensor_msgs.msg import Image

from angel_system.uho.src.models.components.transformer import TemTRANSModule
from angel_system.uho.src.models.components.unified_fcn import UnifiedFCNModule
from angel_system.uho.src.models.unified_ho_module import UnifiedHOModule
from angel_system.uho.src.data_helper import create_batch
from angel_utils.conversion import get_hand_pose_from_msg
from angel_utils.sync_msgs import get_frame_synced_hand_poses


BRIDGE = CvBridge()


class UHOActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._image_topic = (
            self.declare_parameter("image_topic", "PVFramesRGB")
            .get_parameter_value()
            .string_value
        )
        self._hand_topic = (
            self.declare_parameter("hand_pose_topic", "HandJointPoseData")
            .get_parameter_value()
            .string_value
        )
        self._obj_det_topic = (
            self.declare_parameter("obj_det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._torch_device = (
            self.declare_parameter("torch_device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self._det_topic = (
            self.declare_parameter("det_topic", "ActivityDetections")
            .get_parameter_value()
            .string_value
        )
        self._frames_per_det = (
            self.declare_parameter("frames_per_det", 32)
            .get_parameter_value()
            .integer_value
        )
        self._model_checkpoint = (
            self.declare_parameter("model_checkpoint",
                                   "/angel_workspace/model_files/uho_epoch_090.ckpt")
            .get_parameter_value()
            .string_value
        )
        self._labels_file = (
            self.declare_parameter("labels_file",
                                   "/angel_workspace/model_files/uho_epoch_090_labels.txt")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Object detections topic: {self._obj_det_topic}")
        log.info(f"Device? {self._torch_device}")
        log.info(f"Frames per detection: {self._frames_per_det}")
        log.info(f"Checkpoint: {self._model_checkpoint}")
        log.info(f"Labels: {self._labels_file}")

        # Image subscriber
        self._image_subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.image_callback,
            1
        )
        # Hand pose subscriber
        self._hand_subscription = self.create_subscription(
            HandJointPosesUpdate,
            self._hand_topic,
            self.hand_callback,
            1
        )
        # Object detections subscriber
        self._obj_det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._obj_det_topic,
            self.obj_det_callback,
            10
        )

        self._slop_ns = (5 / 60.0) * 1e9 # slop (hand msgs have rate of ~60hz per hand)

        self._publisher = self.create_publisher(
            ActivityDetection,
            self._det_topic,
            1
        )

        # Stores the data until we have enough to send to the detector
        self._frames = []
        self._hand_poses = dict(
            lhand=[],
            rhand=[],
        )
        self._hand_pose_stamps = dict(
            lhand=[],
            rhand=[],
        )
        self._frame_stamps = []
        self._obj_dets = []

        # TODO - parameterize this?
        self._topk = 5

        # Instantiate the activity detector models
        fcn = UnifiedFCNModule(net="resnext", num_cpts=21, obj_classes=9, verb_classes=12)
        temporal = TemTRANSModule(act_classes=27, hidden=256, dropout=0.1, depth=6)

        self._detector: UnifiedHOModule = UnifiedHOModule(
            fcn=fcn,
            temporal=temporal,
            checkpoint=self._model_checkpoint,
            device=self._torch_device,
            labels_file=self._labels_file
        )
        self._detector.eval()
        self._detector = self._detector.to(device=self._torch_device)
        log.info(f"UHO Detector initialized")

    def hand_callback(self, hand_pose):
        """
        Callback function for hand poses. Messages are saved in the hand_poses list.
        """
        lhand, rhand = get_hand_pose_from_msg(hand_pose)
        if hand_pose.hand == 'Right':
            self._hand_poses['rhand'].append(rhand)
            self._hand_pose_stamps['rhand'].append(hand_pose.header.stamp)
        elif hand_pose.hand == 'Left':
            self._hand_poses['lhand'].append(lhand)
            self._hand_pose_stamps['lhand'].append(hand_pose.header.stamp)

    def image_callback(self, image):
        """
        Callback function for images. Messages are saved in the images list.
        """
        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)

        self._frames.append(rgb_image_np)

        # Store the image timestamp
        self._frame_stamps.append(image.header.stamp)

    def obj_det_callback(self, msg):
        """
        Callback for the object detection set message. If there are enough frames
        accumulated for the activity detector and there is an object detection
        message received for the last frame in the frame set or after it,
        the activity detector model is called and a new activity detection message
        is published with the current activity predictions.
        """
        log = self.get_logger()

        if msg.num_detections < self._topk:
            log.warn(f"Received msg with less than {self._topk} detections.")
            return

        self._obj_dets.append(msg)

        if len(self._frames) >= self._frames_per_det:
            frame_stamp_set = self._frame_stamps[:self._frames_per_det]
            ready_to_predict = False

            # If the source stamp for this detection message is after or equal to
            # the last frame stamp in the current set, then we have all of the
            # detections and can move onto processing this set of frames.
            msg_nsec = msg.source_stamp.sec * 10e9 + msg.source_stamp.nanosec
            frm_nsec = frame_stamp_set[-1].sec * 10e9 + frame_stamp_set[-1].nanosec
            if msg_nsec >= frm_nsec:
                ready_to_predict = True

            # Need to wait until the object detector has processed all of these frames
            if not ready_to_predict:
                log.info(f"Waiting for more object detection results")
                return

            # Get the frame synchronized hand poses for this set of frames
            frame_set = self._frames[:self._frames_per_det]

            lhand_pose_set, rhand_pose_set = get_frame_synced_hand_poses(
                frame_stamp_set,
                self._hand_poses,
                self._hand_pose_stamps,
                self._slop_ns
            )

            # Get the object detections to use
            first_frm_nsec = frame_stamp_set[0].sec * 10e9 + frame_stamp_set[0].nanosec
            last_frm_nsec = frame_stamp_set[-1].sec * 10e9 + frame_stamp_set[-1].nanosec
            obj_det_idxs_to_remove = []
            obj_det_set = []
            for idx, det in enumerate(self._obj_dets):
                if det.num_detections == 0:
                    log.info(f"no dets, det source: {det.source_stamp}")
                    continue

                det_source_stamp_nsec = det.source_stamp.sec * 10e9 + det.source_stamp.nanosec

                # Check that this detection is within the range of time
                # for the current frame set
                if det_source_stamp_nsec < first_frm_nsec:
                    # Detection is before the first frame in this set,
                    # so we can remove it
                    obj_det_idxs_to_remove.append(idx)
                    continue
                elif det_source_stamp_nsec > last_frm_nsec:
                    # Detection is after the last frame in this set,
                    # so keep it for later
                    continue

                obj_det_idxs_to_remove.append(idx)
                obj_det_set.append(det)

            frame_set_processed, aux_data = create_batch(
                frame_set,
                lhand_pose_set,
                rhand_pose_set,
                obj_det_set,
                self._topk,
            )

            # Inference!
            activities_detected, labels = self._detector.forward(frame_set_processed, aux_data)

            # Create and publish the ActivityDetection msg
            activity_msg = ActivityDetection()

            # This message time
            activity_msg.header.stamp = self.get_clock().now().to_msg()

            # Trace to the source
            activity_msg.header.frame_id = "Activity detection"
            activity_msg.source_stamp_start_frame = frame_stamp_set[0]
            activity_msg.source_stamp_end_frame = frame_stamp_set[-1]

            activity_msg.label_vec = labels
            activity_msg.conf_vec = activities_detected[0].squeeze().tolist()

            # Publish!
            self._publisher.publish(activity_msg)
            log.info(f"Activities detected: {activities_detected}")
            log.info(f"Top activity detected: {activities_detected[1]}")

            # Clear out stored frames, aux_data, and timestamps
            self._frames = self._frames[self._frames_per_det:]
            self._frame_stamps = self._frame_stamps[self._frames_per_det:]

            # Remove old hand poses
            hands = self._hand_pose_stamps.keys()
            last_frm_nsec = frame_stamp_set[-1].sec * 10e9 + frame_stamp_set[-1].nanosec
            for h in hands:
                hand_idxs_to_remove = []
                for idx, stamp in enumerate(self._hand_pose_stamps[h]):
                    h_nsec = stamp.sec * 10e9 + stamp.nanosec
                    if h_nsec <= last_frm_nsec:
                        # Hand pose is before or equal to the last frame
                        # in the set of frames we just processed, so we can
                        # remove it.
                        hand_idxs_to_remove.append(idx)

                for i in sorted(hand_idxs_to_remove, reverse=True):
                    del self._hand_poses[h][i]
                    del self._hand_pose_stamps[h][i]

            for i in sorted(obj_det_idxs_to_remove, reverse=True):
                del self._obj_dets[i]


def main():
    rclpy.init()

    detector = UHOActivityDetector()

    rclpy.spin(detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detector.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
