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

from angel_msgs.msg import (
    ActivityDetection,
    HandJointPosesUpdate,
    ObjectDetection2dSet
)
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

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Object detections topic: {self._obj_det_topic}")
        log.info(f"Device? {self._torch_device}")
        log.info(f"Frames per detection: {self._frames_per_det}")

        self.subscription_list = []
        # Image subscription
        self.subscription_list.append(mf.Subscriber(self, Image, self._image_topic))
        # Hand pose subscription
        self.subscription_list.append(mf.Subscriber(self, HandJointPosesUpdate, self._hand_topic))
        self.time_sync = mf.ApproximateTimeSynchronizer(
            self.subscription_list,
            1500, # queue size
            10  # slop (delay msgs can be synchronized in seconds)
        )

        self.time_sync.registerCallback(self.multimodal_listener_callback)

        # Object detections subscriber
        self._obj_det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._obj_det_topic,
            self.obj_det_callback,
            1000 # queue size
        )

        self._publisher = self.create_publisher(
            ActivityDetection,
            self._det_topic,
            1
        )

        # Stores the frames until we have enough to send to the detector
        self._frames = []
        '''
        self._aux_data = dict(
            lhand=[],
            rhand=[],
            dets=[],
            bbox=[],
        )
        '''
        self._hand_poses = dict(
            lhand=[],
            rhand=[],
        )
        self._frame_stamps = []
        self._obj_dets = []
        self._obj_det_stamps = []

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
        log.info(f"UHO Detector initialized")

    def multimodal_listener_callback(self, image, hand_pose):
        """
        TODO
        """
        log = self.get_logger()

        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)

        self._frames.append(rgb_image_np)

        # Store the image timestamp
        self._frame_stamps.append(image.header.stamp)

        lhand, rhand = self.get_hand_pose_from_msg(hand_pose)
        self._hand_poses['lhand'].append(lhand)
        self._hand_poses['rhand'].append(rhand)

    def obj_det_callback(self, msg):
        """
        Callback for the object detection set message.
        """
        log = self.get_logger()
        self._obj_dets.append(msg)
        self._obj_det_stamps.append(msg.source_stamp)

        if len(self._frames) >= self._frames_per_det:
            # Check if the object descriptor has processed all of the frame for this frame set
            frame_stamp_set = self._frame_stamps[:self._frames_per_det]
            all_stamps_matched = True
            for f in frame_stamp_set:
                print(f, self._obj_det_stamps)
                if f not in self._obj_det_stamps:
                    all_stamps_matched = False
                    break

            # Need to wait until the object detector has processed all of these frames
            if not all_stamps_matched:
                log.info(f"not all frames have detection set msg")
                return

            frame_set = self._frames[:self._frames_per_det]
            lhand_pose_set = self._hand_poses['lhand'][:self._frames_per_det]
            rhand_pose_set = self._hand_poses['rhand'][:self._frames_per_det]

            # TODO - parameterize this
            topk = 5

            aux_data = dict(
                lhand=lhand_pose_set,
                rhand=rhand_pose_set,
                dets=[],
                bbox=[],
            )

            # Check if there are any detections for this frame set
            for d in self._obj_dets:
                if d.num_detections == 0:
                    log.info(f"no dets, det source: {d.source_stamp}")
                    continue

                if d.source_stamp in frame_stamp_set:
                    det_descriptors = torch.Tensor(d.descriptors).reshape(tuple(d.descriptor_dims))
                    #log.info(f"MATCH {det_descriptors.shape}")
                    aux_data['dets'].append(det_descriptors[:topk])

                    bboxes = [
                        torch.Tensor((d.left[i], d.top[i], d.right[i], d.bottom[i])) for i in range(topk)
                    ]
                    bboxes = torch.stack(bboxes)

                    aux_data['bbox'].append(bboxes)

            #log.info(f"aux data {self._aux_data}")

            activities_detected = self._detector.forward(frame_set, aux_data)

            log.info(f"Activities detected: {activities_detected} {activities_detected[0].shape}")

            # Clear out stored frames, aux_data, and timestamps
            self._frames = self._frames[self._frames_per_det:]
            self._frame_stamps = self._frame_stamps[self._frames_per_det:]
            self._hand_poses['lhand'] = self._hand_poses['lhand'][self._frames_per_det:]
            self._hand_poses['rhand'] = self._hand_poses['rhand'][self._frames_per_det:]
            self._obj_dets = []
            self._obj_det_stamps = []

    def get_hand_pose_from_msg(self, msg):
        """
        TODO
        """
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
