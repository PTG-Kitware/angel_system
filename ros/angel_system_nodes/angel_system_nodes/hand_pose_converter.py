from threading import Event, RLock, Thread
from typing import List

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
    HeadsetPoseData,
)
from angel_system.utils.matching import descending_match_with_tolerance
from angel_system.utils.matrix_conversion import (
    convert_1d_4x4_to_2d_matrix,
    project_3d_pos_to_2d_image,
    PROJECTION_MATRIX,
)
from angel_utils.conversion import time_to_int
from geometry_msgs.msg import (
    Point,
)


class HandPoseConverter(Node):
    """
    ROS node that converts the MRTK hand joint positions from 3D world space to
    2D RGB camera image space.

    Currently, the input/output messages are of the same type,
    HandJointPosesUpdate, with the 2D version having 1.0 for the z-position
    component. HeadsetPoseMessages are used to get the world-to-camera matrix
    at the time of the hand pose messages.
    """

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
            self.declare_parameter("tol", (5 / 60.0) * 1e9)
            .get_parameter_value()
            .double_value
        )

        log = self.get_logger()
        log.info(f"Hand input topic: {self._hand_in_topic}")
        log.info(f"Head pose input topic: {self._head_pose_in_topic}")
        log.info(f"Hand output topic: {self._hand_out_topic}")

        # Hand pose subscriber
        self._hand_subscription_cb_group = MutuallyExclusiveCallbackGroup()
        self._hand_subscription = self.create_subscription(
            HandJointPosesUpdate,
            self._hand_in_topic,
            self.hand_callback,
            # Hand poses arrive in bursts @ up to 60hz per hand, so this gives
            # a buffer to make sure we don't drop any messages.
            (60 * 2 * 2),
            callback_group=self._hand_subscription_cb_group,
        )
        # Head pose subscriber
        self._head_subscription_cb_group = MutuallyExclusiveCallbackGroup()
        self._head_subscription = self.create_subscription(
            HeadsetPoseData,
            self._head_pose_in_topic,
            self.head_callback,
            (30 * 2), # 2 seconds of frames
            callback_group=self._head_subscription_cb_group,
        )
        self._hand_publisher_cb_group = MutuallyExclusiveCallbackGroup()
        self._hand_publisher = self.create_publisher(
            HandJointPosesUpdate,
            self._hand_out_topic,
            1,
            callback_group=self._hand_publisher_cb_group,
        )

        # Stores the messages received
        self._head_poses: List[HeadsetPoseData] = []
        self._hand_poses_left: List[HandJointPosesUpdate] = []
        self._hand_poses_right: List[HandJointPosesUpdate] = []
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
                if hand_pose.hand == "Left":
                    self._hand_poses_left.append(hand_pose)
                elif hand_pose.hand == "Right":
                    self._hand_poses_right.append(hand_pose)
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
                    frame_times_ns = [time_to_int(p.header.stamp) for p in self._head_poses]

                    lhand_poses_synced = descending_match_with_tolerance(
                        frame_times_ns,
                        self._hand_poses_left,
                        self._tol,
                        time_from_value_fn=self.head_pose_msg_to_time
                    )
                    rhand_poses_synced = descending_match_with_tolerance(
                        frame_times_ns,
                        self._hand_poses_right,
                        self._tol,
                        time_from_value_fn=self.head_pose_msg_to_time
                    )

                    hand_lists = [lhand_poses_synced, rhand_poses_synced]
                    first_hand_time = None
                    for hand_poses in hand_lists:
                        for head_pose, hand_pose in zip(self._head_poses, hand_poses):
                            if hand_pose is None:
                                # No matching head pose yet
                                continue

                            if first_hand_time is None:
                                first_hand_time = time_to_int(hand_pose.header.stamp)

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
                            log.info("Publish 2d hand pose msg")

                            # Remove this hand pose from the global list
                            if hand_pose.hand == "Left":
                                self._hand_poses_left.remove(hand_pose)
                            else:
                                self._hand_poses_right.remove(hand_pose)

                    # Clear out old head poses
                    if first_hand_time is not None:
                        self.clear_head_poses(first_hand_time)

    def clear_head_poses(
        self,
        t,
        window: float = -1e9 / 2.0 # 0.5 seconds
    ):
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
        self._rt_thread.join()

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

    # If things are going wrong, set this False to debug in a serialized setup.
    do_multithreading = True

    if do_multithreading:
        # Don't really want to use *all* available threads...
        # 4 threads because:
        # - 2 known subscribers which have their own groups
        # - 1 for default group
        # - 1 for publishers
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(converter)
        try:
            executor.spin()
        except KeyboardInterrupt:
            converter.stop_runtime()
            converter.get_logger().info("Keyboard interrupt, shutting down.\n")
    else:
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
