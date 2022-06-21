import time

import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Pose, Point

from angel_msgs.msg import (
    ActivityDetection,
    ObjectDetection3dSet,
    TaskUpdate,
    AruiUpdate,
    AruiObject3d
)

class FeedbackGenerator(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("activity_detector_topic", "ActivityDetections")
        self.declare_parameter("object_detection_topic", "ObjectDetections3d")
        # TODO: task needs to be updated to emit TaskNode
        self.declare_parameter("task_monitor_topic", "TaskUpdates")
        self.declare_parameter("arui_update_topic", "AruiUpdates")

        self._activity_detector_topic = self.get_parameter("activity_detector_topic").get_parameter_value().string_value
        self._object_detection_topic = self.get_parameter("object_detection_topic").get_parameter_value().string_value
        self._task_monitor_topic = self.get_parameter("task_monitor_topic").get_parameter_value().string_value
        self._arui_update_topic = self.get_parameter("arui_update_topic").get_parameter_value().string_value

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
        self.arui_update_message = AruiUpdate()

    def publish_update(self):
        self.arui_update_message.header.stamp = self.get_clock().now().to_msg()
        self.log.info(f"Publishing AruiUpdate: {self.arui_update_message}")
        self.arui_update_publisher.publish(self.arui_update_message)

    def activity_callback(self, activity):
        self.arui_update_message.latest_activity = activity
        self.arui_update_message.header.frame_id = activity.header.frame_id
        self.publish_update()

    def object_callback(self, object_msg):
        self.arui_update_message.object3d_remove = self.arui_update_message.object3d_update
        self.log.info(f"object message type: {type(object_msg)}")
        # convert ObjectDetection3dSet to AruiObject3d
        detections = []
        for i in range(object_msg.num_objects):
            detection = AruiObject3d()
            #detection.uid =
            detection.stamp = object_msg.source_stamp
            detection.label = object_msg.object_labels[i]

            detection.bbox = BoundingBox3D()

            # min = sorted[0], max = sorted[-1]
            xs = sorted([object_msg.right[i].x,  object_msg.left[i].x,  object_msg.top[i].x, object_msg.bottom[i].x])
            ys = sorted([object_msg.right[i].y,  object_msg.left[i].y,  object_msg.top[i].y, object_msg.bottom[i].y])
            zs = sorted([object_msg.right[i].z,  object_msg.left[i].z,  object_msg.top[i].z, object_msg.bottom[i].z])

            w = xs[-1] - xs[0]
            h = ys[-1] - ys[0]
            d = zs[-1] - zs[0]

            detection.bbox.size.x = w
            detection.bbox.size.y = h
            detection.bbox.size.z = d

            detection.bbox.center.position.x = xs[0] + (0.5 * w)
            detection.bbox.center.position.y = ys[0] + (0.5 * h)
            detection.bbox.center.position.z = zs[0] + (0.5 * d)

            self.log.info(f"detection: {detection}")
            detections.append(detection)

        self.arui_update_message.object3d_update = detections
        self.arui_update_message.header.frame_id = object_msg.header.frame_id
        self.publish_update()

    def task_callback(self, task):
        # TODO: Update this to TaskNode type
        self.arui_update_message.current_task_uid = task.task_name
        self.arui_update_message.header.frame_id = task.header.frame_id
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

