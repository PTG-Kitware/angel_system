from threading import Event, RLock, Thread
from typing import List

import numpy as np

import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
    HeadsetPoseData,
)
from angel_system.utils.matching import descending_match_with_tolerance
from angel_system.utils.matrix_conversion import (
    convert_1d_4x4_to_2d_matrix,
    project_3d_pos_to_2d_image,
)
from angel_utils.conversion import time_to_int
from geometry_msgs.msg import (
    Point,
)


# NOTE: These values were extracted from the projection matrix provided by the
# Unity main camera with the Windows MR plugin (now deprecated).
# For some unknown reason, the values provided by the Open XR plugin are
# different and do not provide the correct results. In the future, we should
# figure out why they are different or extract the focal length values from the
# MediaFrameReader which provides the frames, instead of the Unity main camera.
FOCAL_LENGTH_X = 1.6304
FOCAL_LENGTH_Y = 2.5084

PROJECTION_MATRIX = [
    [FOCAL_LENGTH_X, 0.0, 0.0, 0.0],
    [0.0, FOCAL_LENGTH_Y, 0.0, 0.0],
    [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
    [0.0, 0.0, -1.0, 0.0]
]


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
        self._tol = (
            self.declare_parameter("tol", 1 / 30.0)
            .get_parameter_value()
            .double_value
        )

        log = self.get_logger()
        log.info(f"Hand input topic: {self._hand_in_topic}")
        log.info(f"Head pose input topic: {self._head_pose_in_topic}")
        log.info(f"Hand output topic: {self._hand_out_topic}")

        # Hand pose subscriber
        self._hand_subscription = self.create_subscription(
            HandJointPosesUpdate,
            self._hand_in_topic,
            self.hand_callback,
            1,
        )
        # Head pose subscriber
        self._hand_subscription = self.create_subscription(
            HeadsetPoseData,
            self._head_pose_in_topic,
            self.head_callback,
            1,
        )
        self._hand_publisher = self.create_publisher(
            HandJointPosesUpdate,
            self._hand_out_topic,
            1,
        )

        # Stores the messages received
        self._head_poses: List[HeadsetPoseData] = []
        self._hand_poses: List[HandJointPosesUpdate] = []
        self._converter_lock = RLock()

        # Create the runtime thread to trigger processing and publish 2D hand
        # poses.
        log.info("Starting runtime thread...")
        # switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = 0.1  # TODO: Parameterize?
        # Event to notify runtime it should try processing now.
        self._rt_awake_evt = Event()
        self._rt_thread = Thread(
            target=self.thread_convert_hand_poses,
            name="convert_hand_poses"
        )
        self._rt_thread.daemon = True
        self._rt_thread.start()
        log.info("Starting runtime thread... Done")

    def hand_callback(self, hand_pose: HandJointPosesUpdate) -> None:
        """
        Callback function for hand pose messages.
        """
        if self.rt_alive():
            with self._converter_lock:
                self._hand_poses.append(hand_pose)
                self._rt_awake_evt.set()

    def head_callback(self, head_pose: HeadsetPoseData) -> None:
        """
        Callback function for head set pose messages.
        """
        if self.rt_alive():
            with self._converter_lock:
                self._head_poses.append(head_pose)
                self._rt_awake_evt.set()

    def thread_convert_hand_poses(self):
        """
        Hand conversion runtime function.
        """
        log = self.get_logger()
        log.info("Runtime loop starting")

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait(self._rt_active_heartbeat):
                # reset the flag for the next go-around
                self._rt_awake_evt.clear()

                with self._converter_lock:
                    hand_times = [time_to_int(p.header.stamp) for p in self._hand_poses]

                    # Get the matching head pose data for this hand pose
                    head_poses_synced = descending_match_with_tolerance(
                        hand_times,
                        self._head_poses,
                        self._tol,
                        time_from_value_fn=self.head_pose_msg_to_time
                    )

                    idxs_to_remove = []
                    idx = 0
                    first_hand_time = None
                    for head_pose, hand_pose in zip(head_poses_synced, self._hand_poses):
                        if head_pose is None:
                            # No matching head pose yet
                            continue

                        if first_hand_time is None:
                            first_hand_time = hand_times[idx]

                        # Compute the transformation matrices
                        cam_to_world_matrix = convert_1d_4x4_to_2d_matrix(
                            head_pose.world_matrix
                        )
                        world_to_cam_matrix = np.linalg.inv(cam_to_world_matrix)
                        projection_matrix = PROJECTION_MATRIX

                        hand_joint_pose_update_2d = HandJointPosesUpdate()
                        hand_joint_pose_update_2d.header.stamp = hand_pose.header.stamp
                        hand_joint_pose_update_2d.header.frame_id = "2d hand pose"
                        hand_joint_pose_update_2d.hand = hand_pose.hand

                        for j in hand_pose.joints:
                            # Get the hand pose
                            joint_position_3d = np.array(
                                [j.pose.position.x,
                                 j.pose.position.y,
                                 j.pose.position.z,
                                 1]
                            )
                            # Convert to 2d coordinates
                            coords = project_3d_pos_to_2d_image(
                                joint_position_3d,
                                world_to_cam_matrix,
                                projection_matrix,
                            )
                            # Create the new HandJointPose message
                            point_2d = Point()
                            point_2d.x = coords[0]
                            point_2d.y = coords[1]
                            point_2d.z = coords[2] # always 1

                            pose_msg = HandJointPose()
                            pose_msg.joint = j.joint
                            pose_msg.pose.position = point_2d

                            hand_joint_pose_update_2d.joints.append(pose_msg)

                        # Publish 2d joint msg
                        self._hand_publisher.publish(hand_joint_pose_update_2d)

                        idxs_to_remove.append(idx)
                        idx += 1

                    # Clear out old head poses
                    if first_hand_time is not None:
                        log.info(f"Published {len(idxs_to_remove)} 2d hand poses")
                        self.clear_head_poses(first_hand_time)

                        # Clear out these hand poses
                        for i in sorted(idxs_to_remove, reverse=True):
                            del self._hand_poses[i]

    def clear_head_poses(self, t, window: float = -1e9 / 2.0):
        """
        Clear out headset pose messages older than window from the given
        time.
        """
        idxs_to_remove = []
        for idx, pose in enumerate(self._head_poses):
            head_time = self.head_pose_msg_to_time(pose)

            if (head_time - t) <= window:
                idxs_to_remove.append(idx)

        for i in sorted(idxs_to_remove, reverse=True):
            del self._head_poses[i]

    def head_pose_msg_to_time(self, head_pose_msg: HeadsetPoseData) -> int:
        """
        Convert HeadPoseData message to integer ns.
        """
        return time_to_int(head_pose_msg.header.stamp)

    def stop_runtime(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()

    def rt_alive(self) -> bool:
        """
        Check that the runtime thread is still alive and raise an exception
        if it is not.
        """
        alive = self._rt_thread.is_alive()
        if not alive:
            self.get_logger().warn("Runtime thread no longer alive.")
            self._rt_thread.join()
        return alive


def main():
    rclpy.init()

    converter = HandPoseConverter()

    try:
        rclpy.spin(converter)
    except KeyboardInterrupt:
        converter.stop_runtime()
        converter.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    #  when the garbage collector destroys the node object... if it gets to it)
    converter.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
