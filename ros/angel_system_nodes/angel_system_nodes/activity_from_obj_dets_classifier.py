import rclpy
from rclpy.node import Node

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ActivityDetection,
)


PARAM_DET_TOPIC = "det_topic"
PARAM_ACT_TOPIC = "act_topic"
PARAM_CLASSIFIER_FILE = "classifier_file"


class ActivityFromObjectDetectionsClassifier(Node):
    """
    ROS node that publishes `ActivityDetection` messages using a classifier and
    `ObjectDetection2dSet` messages.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        parameter_names = [
            PARAM_DET_TOPIC,
            PARAM_ACT_TOPIC,
            PARAM_CLASSIFIER_FILE,
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        # Check for not-set parameters
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._det_topic = self.get_parameter(PARAM_DET_TOPIC).value
        self._act_topic = self.get_parameter(PARAM_ACT_TOPIC).value
        self._classifier_file = self.get_parameter(PARAM_CLASSIFIER_FILE).value
        log.info(f"Detections topic: "
                      f"({type(self._det_topic).__name__}) "
                      f"{self._det_topic}")
        log.info(f"Activity detections topic: "
                      f"({type(self._act_topic).__name__}) "
                      f"{self._act_topic}")
        log.info(f"Classifier: "
                      f"({type(self._classifier_file).__name__}) "
                      f"{self._classifier_file}")

        # Create activity det publisher
        self._det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.det_callback,
            1
        )
        self._activity_publisher = self.create_publisher(
            ActivityDetection,
            self._act_topic,
            1
        )

        # Would load the angel_system/sklearn module here from the classifier file
        self.classifier = None

    def det_callback(self, msg: ObjectDetection2dSet) -> None:
        """
        Callback function for `ObjectDetection2dSet` messages. Runs the classifier,
        creates an `ActivityDetection` message from the results the classifier,
        and publish the `ActivityDetection` message.
        """
        log = self.get_logger()

        # TODO: Call stub activity classifier function. It is expected that this
        # function will return a dictionary mapping activity labels to confidences.
        #label_conf_dict = self.classifier.classify(msg)
        label_conf_dict = {
            "activity1": 0.99,
            "activity2": 0.74,
            "activity3": 0.3,
            "activity4": 0.1,

        }

        activity_det_msg = ActivityDetection()
        activity_det_msg.header.stamp = self.get_clock().now().to_msg()
        activity_det_msg.header.frame_id = "ActivityDetection"

        # Set the activity det start/end frame time to this frame's source stamp
        activity_det_msg.source_stamp_start_frame = msg.source_stamp
        activity_det_msg.source_stamp_end_frame = msg.source_stamp

        activity_det_msg.label_vec = list(label_conf_dict.keys())
        activity_det_msg.conf_vec = list(label_conf_dict.values())

        self._activity_publisher.publish(activity_det_msg)
        log.info("Publish activity detection msg")



def main():
    rclpy.init()

    activity_classifier = ActivityFromObjectDetectionsClassifier()

    try:
        rclpy.spin(activity_classifier)
    except KeyboardInterrupt:
        activity_classifier.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    activity_classifier.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
