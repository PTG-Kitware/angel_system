import json
import time
from typing import Dict

# TODO: This is added so that the the angel_system/uho/src folder
# is found when torch attempts to load the saved model checkpoints.
# Is there a better way to do this?
import sys
sys.path.append("angel_system/uho")

from cv_bridge import CvBridge
import cv2
import message_filters as mf
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from smqtk_core.configuration import from_config_dict
import torch

from angel_msgs.msg import ActivityDetection, HandJointPosesUpdate
from sensor_msgs.msg import Image

from angel_system.uho.src.models.components.transformer import TemTRANSModule
from angel_system.uho.src.models.components.unified_fcn import UnifiedFCNModule
from angel_system.uho.src.models.unified_ho_module import UnifiedHOModule


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
            self.declare_parameter("frames_per_det", 32.0)
            .get_parameter_value()
            .double_value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Device? {self._torch_device}")
        log.info(f"Frames per detection: {self._frames_per_det}")

        self.subscription_list = []
        # Image subscription
        self.subscription_list.append(mf.Subscriber(self, Image, self._image_topic))
        # Hand pose subscription
        self.subscription_list.append(mf.Subscriber(self, HandJointPosesUpdate, self._hand_topic))
        self.time_sync = mf.TimeSynchronizer(
            self.subscription_list,
            self._frames_per_det
        )

        self.time_sync.registerCallback(self.multimodal_listener_callback)

        self._publisher = self.create_publisher(
            ActivityDetection,
            self._det_topic,
            1
        )

        # Stores the frames until we have enough to send to the detector
        self._frames = []
        self._aux_data = dict(
            lhand=[],
            rhand=[],
        )
        self._source_stamp_start_frame = -1
        self._source_stamp_end_frame = -1

        # TODO
        checkpoint = "/angel_workspace/angel_system/uho/epoch_046.ckpt"

        # Instantiate the activity detector models
        fcn = UnifiedFCNModule(net="resnext", num_cpts=21, obj_classes=9, verb_classes=12)
        temporal = TemTRANSModule(act_classes=27, hidden=256, dropout=0.1, depth=4)

        self._detector: UnifiedHOModule = UnifiedHOModule(
            fcn=fcn,
            temporal=temporal,
            checkpoint=checkpoint
        )
        self._detector.eval()
        self._detector = self._detector.to(device=self._torch_device)
        log.info(f"Detector {self._detector} initialized")

    def multimodal_listener_callback(self, image, hand_pose):
        log = self.get_logger()
        log.info(f"Got image and hand! {image.header.stamp} {hand_pose.header.stamp}")

        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)

        self._frames.append(rgb_image_np)
        lhand, rhand = self.get_hand_pose_from_msg(hand_pose)
        self._aux_data['lhand'].append(lhand)
        self._aux_data['rhand'].append(rhand)

        # Store the image timestamp
        if self._source_stamp_start_frame == -1:
            self._source_stamp_start_frame = image.header.stamp

        if len(self._frames) >= self._frames_per_det:

            activities_detected = self._detector.forward(self._frames, self._aux_data)

            # Clear out stored frames, aux_data, and timestamps
            self._frames = []
            self._aux_data = dict(
                lhand=[],
                rhand=[],
            )
            self._source_stamp_start_frame = -1

    def get_hand_pose_from_msg(self, msg):
        hand_joints = [{"joint": m.joint,
                        "position": [ m.pose.position.x,
                                      m.pose.position.y,
                                      m.pose.position.z]}
                      for m in msg.joints]

        # Rejecting joints not in OpenPose hand skeleton format
        reject_joint_list = ['ThumbMetacarpalJoint',
                            'IndexMetacarpal',
                            'MiddleMetacarpal',
                            'RingMetacarpal',
                            'PinkyMetacarpal']
        joint_pos = []
        for j in hand_joints:
            if j["joint"] not in reject_joint_list:
                joint_pos.append(j["position"])
        joint_pos = np.array(joint_pos).flatten()

        if msg.hand == 'Right':
            rhand = joint_pos
            lhand = np.zeros_like(joint_pos)
        elif msg.hand == 'Left':
            lhand = joint_pos
            rhand = np.zeros_like(joint_pos)
        else:
            lhand = np.zeros_like(joint_pos)
            rhand = np.zeros_like(joint_pos)

        return lhand, rhand


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
