import uuid
from threading import RLock

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from angel_msgs.msg import (
    ActivityDetection,
    ObjectDetection3dSet,
    TaskUpdate,
    AruiUpdate,
    AruiObject3d,
    VisionBoundingBox3d
)


class FeedbackGenerator(Node):
    """
    ROS node responsible for sending activity, detection, and task information to the ARUI when the information has updated.

    Takes in information from the `angel_msgs/ActivityDetection`, `angel_msgs/ObjectDetection3dSet`, and `angel_msgs/TaskUpdate` messages.

    Publishes `angel_msgs/AruiUpdate` representing the current activity, detections, and task.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._activity_detector_topic = self.declare_parameter("activity_detector_topic", "ActivityDetections").get_parameter_value().string_value
        self._object_detection_topic = self.declare_parameter("object_detection_topic", "ObjectDetections3d").get_parameter_value().string_value
        self._task_monitor_topic = self.declare_parameter("task_monitor_topic", "TaskUpdates").get_parameter_value().string_value
        self._arui_update_topic = self.declare_parameter("arui_update_topic", "AruiUpdates").get_parameter_value().string_value
        # TODO: add topic input for predicted user intents

        # logger
        self.log = self.get_logger()
        self.log.info(f"Activity detector topic: {self._activity_detector_topic}")
        self.log.info(f"Object detection topic: {self._object_detection_topic}")
        self.log.info(f"Task monitor topic: {self._task_monitor_topic}")
        self.log.info(f"AruiUpdate topic: {self._arui_update_topic}")

        # subscribers
        self.activity_subscriber = self.create_subscription(
            ActivityDetection,
            self._activity_detector_topic,
            self.activity_callback,
            1
        )

        self.object_detector_subscriber = self.create_subscription(
            ObjectDetection3dSet,
            self._object_detection_topic,
            self.object_callback,
            1
        )

        self.task_monitor_subscriber = self.create_subscription(
            TaskUpdate,
            self._task_monitor_topic,
            self.task_callback,
            1
        )

        # publisher
        self.arui_update_publisher = self.create_publisher(
            AruiUpdate,
            self._arui_update_topic,
            1
        )

        # message
        # Stateful message components that are carried through from update to
        # update. Content here is locked via the `self.lock` below.
        self._arui_update_header_frame_id = "ARUI Update"
        self._arui_update_latest_activity = ActivityDetection()
        self._arui_update_task_update = TaskUpdate()
        # Set the default value for task update current step ID to the "real"
        # base value of -1
        # TODO: This is fragile. Dependent on implementation detail of task
        #       monitor v2.
        self._arui_update_task_update.current_step_id = -1

        # Lock for stateful ARUI update message parts.
        # Reentrant to allow for chaining updates with the publish method.
        self.lock = RLock()

        # detection uuids
        self.uuids = dict()

    def _make_common_header(self) -> Header:
        """
        Make a header sub-message in a common way across usages.
        """
        return Header(
            frame_id=self._arui_update_header_frame_id,
            stamp=self.get_clock().now().to_msg(),
        )

    def publish_update(self, object3d_remove=(), object3d_update=(),
                       notifications=(), intents_for_confirmation=()) -> None:
        """
        Central message publishing method.

        Aggregates any stateful storage into a new message instance.
        Stateless message components should be provided as arguments.
        """
        this_header = self._make_common_header()
        with self.lock:
            self.log.info(f"Publishing AruiUpdate")
            self.arui_update_publisher.publish(AruiUpdate(
                header=this_header,
                object3d_remove=object3d_remove,
                object3d_update=object3d_update,
                latest_activity=self._arui_update_latest_activity,
                # TODO: expertise_level=,
                task_update=self._arui_update_task_update,
                notifications=notifications,
                intents_for_confirmation=intents_for_confirmation,
            ))

    def activity_callback(self, activity):
        with self.lock:
            self._arui_update_latest_activity = activity
            self.publish_update()

    def object_callback(self, object_msg):
        # convert ObjectDetection3dSet to AruiObject3d
        detections = []
        for i in range(object_msg.num_objects):
            detection = AruiObject3d()

            detection.label = object_msg.object_labels[i]

            # TODO: Update this to real tracking
            # For now, assumes only one type of object will be in the scene at a time
            if detection.label in self.uuids.keys():
                detection.uid = self.uuids[detection.label]
            else:
                detection.uid = str(uuid.uuid4())
                self.uuids[detection.label] = detection.uid

            detection.stamp = object_msg.source_stamp

            detection.bbox = VisionBoundingBox3d()

            # min = sorted[0], max = sorted[-1]
            xs = sorted([object_msg.right[i].x,  object_msg.left[i].x,  object_msg.top[i].x, object_msg.bottom[i].x])
            ys = sorted([object_msg.right[i].y,  object_msg.left[i].y,  object_msg.top[i].y, object_msg.bottom[i].y])
            zs = sorted([object_msg.right[i].z,  object_msg.left[i].z,  object_msg.top[i].z, object_msg.bottom[i].z])

            detection.bbox.size.x = xs[-1] - xs[0] # width
            detection.bbox.size.y = ys[-1] - ys[0] # height
            detection.bbox.size.z = zs[-1] - zs[0] # depth

            detection.bbox.center.position.x = xs[0] + (0.5 * detection.bbox.size.x)
            detection.bbox.center.position.y = ys[0] + (0.5 * detection.bbox.size.y)
            detection.bbox.center.position.z = zs[0] + (0.5 * detection.bbox.size.z)

            detections.append(detection)

        self.publish_update(object3d_update=detections)

    def task_callback(self, task):
        """
        Update the ARUI message with the given task update.
        """
        with self.lock:
            self._arui_update_task_update = task
            self.publish_update()


def main():
    rclpy.init()

    feedback_generator = FeedbackGenerator()

    rclpy.spin(feedback_generator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    feedback_generator.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
