from threading import Event, Lock, Thread
import json

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from angel_system.utils.event import WaitAndClearEvent
from angel_utils.conversion import time_to_float, time_to_int
from builtin_interfaces.msg import Time

from angel_msgs.msg import ObjectDetection2dSet, HandJointPosesUpdate, ActivityDetection, TaskUpdate
from angel_utils import (
    declare_and_get_parameters,
    RateTracker,  # DYNAMIC_TYPE
    make_default_main,
)
from std_msgs.msg import String as ros2_string
from angel_msgs.msg import ImageMetadata


class LatencyTracker(Node):
    """
    ROS node that tracks latency of ros2 messages.
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
                ("image_md_topic",),
                ("det_topic",),
                ("pose_topic",),
                ("activity_topic",),
                ("latency_topic",),
                ##################################
                # Defaulted parameters
                ("rt_thread_heartbeat", 0.1),
                # If we should enable additional logging to the info level
                # about when we receive and process data.
                ("enable_time_trace_logging", False),
                ("gsp_topic", "TaskUpdates"),
            ],
        )
        self._image_md_topic = param_values["image_md_topic"]
        self._det_topic = param_values["det_topic"]
        self._pose_topic = param_values["pose_topic"]
        self._act_topic = param_values["activity_topic"]
        self._latency_topic = param_values["latency_topic"]
        self._gsp_topic = param_values["gsp_topic"]

        self._enable_trace_logging = param_values["enable_time_trace_logging"]

        ##########################################
        # Other stateful properties

        # Single slot for latest image message to process detection over.
        # This should be written by the image topic subscriber callback, and
        # read from the runtime loop once per iteration.
        self._cur_det_msg_lock = Lock()
        self._cur_pose_msg_lock = Lock()
        self._cur_act_msg_lock = Lock()
        self._image_lookup_lock = Lock()
        self._task_msg_lock = Lock()

        self._rate_tracker = RateTracker()

        self._det = None
        self._pose = None
        self._act = None
        self._task = None
        self._image_lookup = {}

        ##########################################
        # Initialize ROS hooks
        log.info("Setting up ROS subscriptions and publishers...")
        self._img_ts_subscriber = self.create_subscription(
            ImageMetadata,
            self._image_md_topic,
            self.img_md_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.det_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._pose_subscriber = self.create_subscription(
            HandJointPosesUpdate,
            self._pose_topic,
            self.pose_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._act_subscriber = self.create_subscription(
            ActivityDetection,
            self._act_topic,
            self.act_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._task_subscriber = self.create_subscription(
            TaskUpdate,
            self._gsp_topic,
            self.task_callback,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self._latency_publisher = self.create_publisher(
            ros2_string,
            self._latency_topic,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        ##########################################
        # Create and start runtime thread and loop.
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

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait_and_clear(self._rt_active_heartbeat):

                msg = ros2_string()

                # get latency from image source stamp vs the time the message was created
                det_lat = None
                if self._det:
                    with self._cur_det_msg_lock:
                        det_msg = self._det
                    dt_time = time_to_float(det_msg.header.stamp)
                    img_time = self.get_msg_time_from_source(det_msg.source_stamp)
                    if img_time is not None:
                        det_lat = dt_time - time_to_float(img_time)

                pose_lat = None
                if self._pose:
                    with self._cur_pose_msg_lock:
                        pose_msg = self._pose
                    ps_time = time_to_float(pose_msg.header.stamp)
                    img_time = self.get_msg_time_from_source(pose_msg.source_stamp)
                    if img_time is not None:
                        pose_lat = ps_time - time_to_float(img_time)

                act_lat_start = None
                act_lat_end = None
                if self._act:
                    with self._cur_act_msg_lock:
                        act_msg = self._act
                    act_time = time_to_float(act_msg.header.stamp)
                    img_time = self.get_msg_time_from_source(
                        act_msg.source_stamp_start_frame
                    )
                    if img_time is not None:
                        act_lat_start = act_time - time_to_float(img_time)
                    img_time = self.get_msg_time_from_source(
                        act_msg.source_stamp_end_frame
                    )
                    if img_time is not None:
                        act_lat_end = act_time - time_to_float(img_time)

                task_lat = None
                if self._task:
                    with self._task_msg_lock:
                        task_msg = self._task
                    task_time = time_to_float(task_msg.header.stamp)
                    img_time = self.get_msg_time_from_source(task_msg.latest_sensor_input_time)
                    if img_time is not None:
                        task_lat = task_time - time_to_float(img_time)

                # save the info to the message
                data = {
                    "detection": det_lat,
                    "pose:": pose_lat,
                    "activity_start": act_lat_start,
                    "activity_end": act_lat_end,
                    "task": task_lat,
                }
                det_str = f"{det_lat:.3f}" if det_lat else "NA"
                pose_str = f"{pose_lat:.3f}" if pose_lat else "NA"
                acts_str = f"{act_lat_start:.3f}" if act_lat_start else "NA"
                acte_str = f"{act_lat_end:.3f}" if act_lat_end else "NA"
                task_str = f"{task_lat:.3f}" if task_lat else "NA"
                log.info(
                    f"Detection: {det_str}, Pose: {pose_str}, Activity.start: {acts_str}, Activity.end: {acte_str}"
                    f", Task monitor: {task_str}"
                )
                msg.data = json.dumps(data, indent=0)

                self._latency_publisher.publish(msg)

                self._rate_tracker.tick()
                log.info(f"Latency Rate: {self._rate_tracker.get_rate_avg()} Hz")

    def img_md_callback(self, msg: ImageMetadata) -> None:
        """
        Capture a detection source image timestamp message.
        """
        log = self.get_logger()
        if self._enable_trace_logging:
            log.info(f"Received image with TS: {msg.header.stamp}")

        with self._image_lookup_lock:
            self._image_lookup[time_to_int(msg.image_source_stamp)] = msg.header.stamp
            self._rt_awake_evt.set()

    def det_callback(self, msg: ObjectDetection2dSet) -> None:
        with self._cur_det_msg_lock:
            self._det = msg

    def pose_callback(self, msg: HandJointPosesUpdate) -> None:
        with self._cur_pose_msg_lock:
            self._pose = msg

    def act_callback(self, msg: ActivityDetection) -> None:
        with self._cur_act_msg_lock:
            self._act = msg

    def task_callback(self, msg: TaskUpdate) -> None:
        with self._task_msg_lock:
            self._task = msg

    def get_msg_time_from_source(self, source_stamp: Time) -> Time:
        with self._image_lookup_lock:
            return self._image_lookup.get(time_to_int(source_stamp))

    def destroy_node(self):
        print("Stopping runtime")
        self.rt_stop()
        print("Shutting down runtime thread...")
        self._rt_active.clear()  # make RT active flag "False"
        self._rt_thread.join()
        print("Shutting down runtime thread... Done")
        super().destroy_node()


main = make_default_main(LatencyTracker, multithreaded_executor=3)


if __name__ == "__main__":
    main()
