import uuid
from threading import Lock

import rclpy
from rclpy.node import Node
from vision_msgs.msg import BoundingBox3D

from angel_msgs.msg import (
    ActivityDetection,
    ObjectDetection3dSet,
    TaskUpdate,
    AruiUpdate,
    AruiObject3d
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
        # TODO: task needs to be updated to emit TaskNode
        self._task_monitor_topic = self.declare_parameter("task_monitor_topic", "TaskUpdates").get_parameter_value().string_value
        self._arui_update_topic = self.declare_parameter("arui_update_topic", "AruiUpdates").get_parameter_value().string_value

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
        self.arui_update_message.header.frame_id = "ARUI Update"

        # lock
        self.lock = Lock()

        # detection uuids
        self.uuids = dict()

    def publish_update(self):
        self.lock.acquire()

        self.arui_update_message.header.stamp = self.get_clock().now().to_msg()
        self.log.info(f"Publishing AruiUpdate: {self.arui_update_message}\n")
        self.arui_update_publisher.publish(self.arui_update_message)

        self.lock.release()

    def activity_callback(self, activity):
        self.lock.acquire()
        self.arui_update_message.latest_activity = activity
        self.lock.release()

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

            detection.bbox = BoundingBox3D()

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

        self.lock.acquire()
        self.arui_update_message.object3d_update = detections
        self.lock.release()

        self.publish_update()

    def task_callback(self, task):
        # TODO: Update this to TaskNode type
        self.lock.acquire()
        self.arui_update_message.current_task_uid = task.task_name
        self.lock.release()

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

