import threading
from threading import Lock

from pynput import keyboard
from rclpy.node import Node

from angel_msgs.msg import AnnotationEvent
from angel_utils import make_default_main


class AnnotationEventMonitor(Node):
    """
    ROS node that monitors the keyboard to generate AnnotationEvent
    messages. Event messages for annotation start and stop are generated
    when the up arrow key is pressed. Event messages for error start
    and stop are are generated when the down arrow key is pressed.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._annotation_event_topic = (
            self.declare_parameter("annotation_event_topic", "AnnotationEvents")
            .get_parameter_value()
            .string_value
        )

        # Initialize ROS hooks
        self._publisher = self.create_publisher(
            AnnotationEvent, self._annotation_event_topic, 1
        )

        # Whether or not an annotation is currently ongoing
        self._annotation_active = False

        # Whether or not an error is currently ongoing
        self._error_active = False

        self._keyboard_lock = Lock()

    def monitor_keypress(self):
        log = self.get_logger()
        log.info("Starting keyboard monitor")
        log.info("Press the up arrow key to toggle annotation recording")
        log.info("Press the down arrow key to toggle error recording")

        # Collect events until released
        with keyboard.Listener(on_press=self.on_press) as listener:
            listener.join()

    def on_press(self, key):
        """
        Callback function for keypress events. The up arrow key controls
        annotation events (start/stop). The down arrow key controls error
        events (start/stop).
        """
        log = self.get_logger()

        with self._keyboard_lock:
            if key == keyboard.Key.up:
                msg = AnnotationEvent()
                msg.header.frame_id = "Annotation Event"
                msg.header.stamp = self.get_clock().now().to_msg()

                if not self._annotation_active:
                    log.info("Generating start annotation event message")
                    self._annotation_active = True
                    msg.description = "Start annotation"
                else:
                    log.info("Generating stop annotation event message")
                    self._annotation_active = False
                    msg.description = "Stop annotation"

                self._publisher.publish(msg)

            elif key == keyboard.Key.down:
                msg = AnnotationEvent()
                msg.header.frame_id = "Annotation Event"
                msg.header.stamp = self.get_clock().now().to_msg()

                if not self._error_active:
                    log.info("Generating start error event message")
                    self._error_active = True
                    msg.description = "Start error"
                else:
                    log.info("Generating stop error event message")
                    self._error_active = False
                    msg.description = "Stop error"

                self._publisher.publish(msg)


def init_kb_thread(node):
    """
    Initialize the
    """
    keyboard_t = threading.Thread(target=node.monitor_keypress)
    keyboard_t.daemon = True
    keyboard_t.start()


main = make_default_main(AnnotationEventMonitor, pre_spin_callback=init_kb_thread)


if __name__ == "__main__":
    main()
