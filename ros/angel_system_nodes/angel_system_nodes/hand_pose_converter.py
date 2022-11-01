import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    HandJointPosesUpdate,
)

from angel_system.utils.matching import descending_match_with_tolerance
from angel_system.utils.simple_timer import SimpleTimer
from angel_utils.conversion import get_hand_pose_from_msg
from angel_utils.conversion import time_to_int


class HandPoseConverter(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._hand_in_topic = (
            self.declare_parameter("hand_pose_in_topic", "HandJointPoseData")
            .get_parameter_value()
            .string_value
        )
        self._head_pose_in_topic = (
            self.declare_parameter("head_pose_in_topic", "HeadsetPoseData")
            .get_parameter_value()
            .string_value
        )
        self._hand_out_topic = (
            self.declare_parameter("hand_pose_out_topic", "HandJointPoseData2d")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Hand input topic: {self._hand_in_topic}")
        log.info(f"Hand output topic: {self._hand_out_topic}")

        # Hand pose subscriber
        self._hand_subscription = self.create_subscription(
            HandJointPosesUpdate,
            self._hand_in_topic,
            self.hand_callback,
            1,
        )

    def hand_callback(self, hand_pose: HandJointPosesUpdate) -> None:
        """
        Callback function for hand poses.
        """
        for joint in hand_pose.joints:
            # Get the hand pose
            joint_position_3d = joint.pose.position
            print(joint_position_3d)

            joint_pose_2d_x = joint_position_3d.x *

            '''

            X' = X * (F/Z)

            Y' = Y * (F/Z)

            bx = ez / dz * dx + ex
            by = ez / dz * dy + ey 
            '''

        # Convert to 2D


def main():
    rclpy.init()

    converter = HandPoseConverter()

    rclpy.spin(converter)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    #  when the garbage collector destroys the node object... if it gets to it)
    converter.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
