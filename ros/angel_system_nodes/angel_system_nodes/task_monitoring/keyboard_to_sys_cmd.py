from threading import Thread

from pynput import keyboard
from rclpy.node import Node

from angel_msgs.msg import (
    SystemCommands,
)
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main


PARAM_SYS_CMD_TOPIC = "system_command_topic"


class KeyboardToSystemCommands(Node):
    """
    ROS node that monitors key presses to output `SystemCommand` messages that
    control task progress.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_SYS_CMD_TOPIC,),
            ],
        )
        self._sys_cmd_topic = param_values[PARAM_SYS_CMD_TOPIC]

        # Initialize ROS hooks
        self._sys_cmd_publisher = self.create_publisher(
            SystemCommands, self._sys_cmd_topic, 1
        )

        # Start the keyboard monitoring thread
        log.info("Starting keyboard threads")
        self._keyboard_t = Thread(target=self.monitor_keypress)
        self._keyboard_t.daemon = True
        self._keyboard_t.start()
        log.info("Starting keyboard threads... done")

    def monitor_keypress(self) -> None:
        log = self.get_logger()
        log.info(
            f"Starting keyboard monitor. Use the 0-9 number keys to advance"
            f" between tasks."
        )

        def for_task_direction(t_id, forward):
            log.info(
                f"Registering command to move task {t_id} "
                f"{'forward' if forward else 'backward'}"
            )
            return lambda: self.publish_sys_cmd(t_id, forward)

        with keyboard.GlobalHotKeys(
            {
                "<ctrl>+<shift>+!": for_task_direction(0, False),
                "<ctrl>+<shift>+@": for_task_direction(0, True),
                "<ctrl>+<shift>+#": for_task_direction(1, False),
                "<ctrl>+<shift>+$": for_task_direction(1, True),
                "<ctrl>+<shift>+%": for_task_direction(2, False),
                "<ctrl>+<shift>+^": for_task_direction(2, True),
                "<ctrl>+<shift>+&": for_task_direction(3, False),
                "<ctrl>+<shift>+*": for_task_direction(3, True),
                "<ctrl>+<shift>+(": for_task_direction(4, False),
                "<ctrl>+<shift>+)": for_task_direction(4, True),
            }
        ) as h:
            h.join()

    def publish_sys_cmd(self, task_id: int, forward: bool) -> None:
        """
        Publishes the SystemCommand message to the configured ROS topic.
        """
        log = self.get_logger()
        msg = SystemCommands()
        msg.task_index = task_id
        if forward:
            msg.next_step = True
            msg.previous_step = False
            cmd_str = "forward"
        else:
            msg.next_step = False
            msg.previous_step = True
            cmd_str = "backward"

        log.info(f"Publishing command for task {task_id} to move {cmd_str}")
        self._sys_cmd_publisher.publish(msg)


main = make_default_main(KeyboardToSystemCommands)


if __name__ == "__main__":
    main()
