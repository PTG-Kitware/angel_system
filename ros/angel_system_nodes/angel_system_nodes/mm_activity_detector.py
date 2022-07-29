import json
import time
from typing import Dict
import pdb

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from smqtk_core.configuration import from_config_dict
import message_filters as mf

from sensor_msgs.msg import Image
from angel_msgs.msg import ActivityDetection, HandJointPosesUpdate
from angel_system.impls.detect_activities.two_stage.two_stage_detect_activities import TwoStageDetector


BRIDGE = CvBridge()
    

class MMActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._image_topic = self.declare_parameter("image_topic", "PVFrames").get_parameter_value().string_value
        self._hand_topic = self.declare_parameter("hand_pose_topic", "HandJointPoseData").get_parameter_value().string_value
        self._use_cuda = self.declare_parameter("use_cuda", True).get_parameter_value().bool_value
        self._det_topic = self.declare_parameter("det_topic", "ActivityDetections").get_parameter_value().string_value
        self._frames_per_det = self.declare_parameter("frames_per_det", 32.0).get_parameter_value().double_value
        self._detector_config = self.declare_parameter("detector_config", "default_activity_det_config.json").get_parameter_value().string_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Use cuda? {self._use_cuda}")
        log.info(f"Frames per detection: {self._frames_per_det}")
        log.info(f"Detector config: {self._detector_config}")

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

        with open(self._detector_config, "r") as f:
            config = json.load(f)

        self._detector: TwoStageDetector = TwoStageDetector.from_config(config)

    def multimodal_listener_callback(self, image, hand_pose):
        log = self.get_logger()
        # log.info("Got image and hand!")

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
            # pdb.set_trace()
            activities_detected = self._detector.detect_activities(self._frames, self._aux_data)

            if len(activities_detected) > 0:
                # Create activity ROS message
                activity_msg = ActivityDetection()

                # This message time
                activity_msg.header.stamp = self.get_clock().now().to_msg()
                # Trace to the source
                activity_msg.header.frame_id = image.header.frame_id

                activity_msg.source_stamp_start_frame = self._source_stamp_start_frame
                activity_msg.source_stamp_end_frame = image.header.stamp

                activity_msg.label_vec = list(activities_detected.keys())
                activity_msg.conf_vec = list(activities_detected.values())

                # Publish activities
                self._publisher.publish(activity_msg)

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

    mm_activity_detector = MMActivityDetector()

    rclpy.spin(mm_activity_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mm_activity_detector.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
