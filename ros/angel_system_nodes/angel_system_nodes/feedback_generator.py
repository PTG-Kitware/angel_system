import uuid
from threading import RLock

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from angel_msgs.msg import (
    ActivityDetection,
    AruiObject3d,
    AruiUpdate,
    AruiUserNotification,
    InterpretedAudioUserIntent,
    ObjectDetection3dSet,
    SystemTextResponse,
    TaskUpdate,
    VisionBoundingBox3d,
    DialogueUtterance,
)
from angel_utils import declare_and_get_parameters


# Parameter name constants
PARAM_ACTIVITY_DET_TOPIC = "activity_detector_topic"
PARAM_OBJECT_DET_TOPIC = "object_detection_topic"
PARAM_TASK_MONITOR_TOPIC = "task_monitor_topic"
PARAM_ARUI_UPDATE_TOPIC = "arui_update_topic"
PARAM_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"
PARAM_SYSTEM_TEXT_RESPONSE_TOPIC = "system_text_response_topic"
PARAM_UTTERANCE_TOPIC = "utterances_topic"


class FeedbackGenerator(Node):
    """
    ROS node responsible for sending activity, detection, and task information
    to the ARUI when the information has updated.

    Takes in information from the `angel_msgs/ActivityDetection`,
    `angel_msgs/ObjectDetection3dSet`, `angel_msgs/TaskUpdate`,
    `angel_msgs/InterpretedAudioUserIntent`, and `angel_msgs/SystemTextResponse`
    messages.

    Publishes `angel_msgs/AruiUpdate` representing the current activity,
    detections, and task.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_ACTIVITY_DET_TOPIC,),
                (PARAM_OBJECT_DET_TOPIC,),
                (PARAM_TASK_MONITOR_TOPIC,),
                (PARAM_ARUI_UPDATE_TOPIC,),
                (PARAM_INTERP_USER_INTENT_TOPIC,),
                (PARAM_SYSTEM_TEXT_RESPONSE_TOPIC,),
                (PARAM_UTTERANCE_TOPIC,),
            ],
        )

        self._activity_detector_topic = param_values[PARAM_ACTIVITY_DET_TOPIC]
        self._object_detection_topic = param_values[PARAM_OBJECT_DET_TOPIC]
        self._task_monitor_topic = param_values[PARAM_TASK_MONITOR_TOPIC]
        self._arui_update_topic = param_values[PARAM_ARUI_UPDATE_TOPIC]
        self._interp_uintent_topic = param_values[PARAM_INTERP_USER_INTENT_TOPIC]
        self._system_text_response_topic = param_values[
            PARAM_SYSTEM_TEXT_RESPONSE_TOPIC
        ]
        self._utterance_topic = param_values[
            PARAM_UTTERANCE_TOPIC
        ]

        # subscribers
        self.activity_subscriber = self.create_subscription(
            ActivityDetection, self._activity_detector_topic, self.activity_callback, 1
        )

        self.object_detector_subscriber = self.create_subscription(
            ObjectDetection3dSet, self._object_detection_topic, self.object_callback, 1
        )

        self.task_monitor_subscriber = self.create_subscription(
            TaskUpdate, self._task_monitor_topic, self.task_callback, 1
        )

        self.interp_uintent_subscriber = self.create_subscription(
            InterpretedAudioUserIntent,
            self._interp_uintent_topic,
            self.user_intent_callback,
            1,
        )

        self.system_text_subscriber = self.create_subscription(
            SystemTextResponse,
            self._system_text_response_topic,
            self.system_text_response_callback,
            1,
        )

        self.utterance_subscriber = self.create_subscription(
            DialogueUtterance,
            self._utterance_topic,
            self.utterance_callback,
            1,
        )

        # publisher
        self.arui_update_publisher = self.create_publisher(
            AruiUpdate, self._arui_update_topic, 1
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

    def publish_update(
        self,
        object3d_remove=(),
        object3d_update=(),
        notifications=(),
        intents_for_confirmation=(),
    ) -> None:
        """
        Central message publishing method.

        Aggregates any stateful storage into a new message instance.
        Stateless message components should be provided as arguments.
        """
        this_header = self._make_common_header()
        with self.lock:
            self.log.debug(f"Publishing AruiUpdate")
            self.arui_update_publisher.publish(
                AruiUpdate(
                    header=this_header,
                    object3d_remove=object3d_remove,
                    object3d_update=object3d_update,
                    latest_activity=self._arui_update_latest_activity,
                    # TODO: expertise_level=,
                    task_update=self._arui_update_task_update,
                    notifications=notifications,
                    intents_for_confirmation=intents_for_confirmation,
                )
            )

    def activity_callback(self, activity: ActivityDetection) -> None:
        with self.lock:
            self._arui_update_latest_activity = activity
            self.publish_update()

    def object_callback(self, object_msg: ObjectDetection3dSet) -> None:
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
            xs = sorted(
                [
                    object_msg.right[i].x,
                    object_msg.left[i].x,
                    object_msg.top[i].x,
                    object_msg.bottom[i].x,
                ]
            )
            ys = sorted(
                [
                    object_msg.right[i].y,
                    object_msg.left[i].y,
                    object_msg.top[i].y,
                    object_msg.bottom[i].y,
                ]
            )
            zs = sorted(
                [
                    object_msg.right[i].z,
                    object_msg.left[i].z,
                    object_msg.top[i].z,
                    object_msg.bottom[i].z,
                ]
            )

            detection.bbox.size.x = xs[-1] - xs[0]  # width
            detection.bbox.size.y = ys[-1] - ys[0]  # height
            detection.bbox.size.z = zs[-1] - zs[0]  # depth

            detection.bbox.center.position.x = xs[0] + (0.5 * detection.bbox.size.x)
            detection.bbox.center.position.y = ys[0] + (0.5 * detection.bbox.size.y)
            detection.bbox.center.position.z = zs[0] + (0.5 * detection.bbox.size.z)

            detections.append(detection)

        self.publish_update(object3d_update=detections)

    def task_callback(self, task: TaskUpdate) -> None:
        """
        Update the ARUI message with the given task update.
        """
        with self.lock:
            self._arui_update_task_update = task
            self.publish_update()

    def user_intent_callback(self, msg: InterpretedAudioUserIntent) -> None:
        """
        Publish an ARUI update message with a *single* predicted user intent.

        TODO: Pending use-case definition for multiple simultaneous user intent
              inputs as opposed to must handling over multiple input messages,
              translating into multiple outgoing ARUI Update messages.
        """
        self.publish_update(intents_for_confirmation=[msg])

    def system_text_response_callback(self, msg: SystemTextResponse) -> None:
        """
        Publish an ARUI update message with a *single* ARUI user notification.
        The ARUI will read the `description` field aloud.
        """
        # Create an AruiUserNotification msg with this information
        notification = AruiUserNotification()

        notification.category = notification.N_CAT_NOTICE
        notification.context = notification.N_CONTEXT_USER_MODELING

        notification.title = f"{msg.utterance_text}"
        notification.description = f"{msg.response}"

        self.publish_update(notifications=[notification])

    def utterance_callback(self, msg: DialogueUtterance) -> None:
        """
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        """
        keyword_check = msg.utterance_text[0:8]
        if (keyword_check.contains("Angel,") or keyword_check.contains("angel,") or keyword_check.contains("Angela") or keyword_check.contains("angela,") or keyword_check.contains("Angel,") or keyword_check.contains("angel") or keyword_check.contains("Angela") or keyword_check.contains("angela")):
            arui_message = msg.utterance_text
            arui_message= arui_message.replace("Angel, ", "")
            arui_message= arui_message.replace("angel, ", "")
            arui_message= arui_message.replace("Angela, ", "")
            arui_message= arui_message.replace("angela, ", "")
            arui_message= arui_message.replace("Angel ", "")
            arui_message= arui_message.replace("angel ", "")
            arui_message= arui_message.replace("Angela ", "")
            arui_message= arui_message.replace("angela ", "")
            arui_message= arui_message.capitalize()

            # Create an AruiUserNotification msg with this information
            notification = AruiUserNotification()

            notification.category = notification.N_CAT_NOTICE
            notification.context = notification.N_CONTEXT_USER_MODELING

            notification.title = f"{arui_message}"
            notification.description = ""

            self.publish_update(notifications=[notification])

        

def main():
    rclpy.init()

    feedback_generator = FeedbackGenerator()

    rclpy.spin(feedback_generator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    feedback_generator.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
