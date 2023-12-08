"""
Consume system messages and log to a file according to the Eval2 requirements
as documented here in the "Log Format Specification" section:
    https://docs.google.com/document/d/1efuWwEvVXWJ-0H1nAV_3kyCDjkW9YY93/edit

This node is currently only compatible with the `global_step_predictor` task
monitoring node due to leveraging specific implementation/output semantics.
"""
import csv
import math
from pathlib import Path
import re
import time
from threading import RLock
from typing import Optional

import numpy as np
from rclpy.node import Node

from angel_msgs.msg import (
    AruiUserNotification,
    TaskUpdate,
)
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main
from angel_utils.conversion import time_to_float


###############################################################################
# Parameter names/docs

# Input topic for task updates
PARAM_TOPIC_TASK_UPDATES = "topic_task_updates"
# Input topic for notifications, some of which are error notifications.
PARAM_TOPIC_NOTIFICATIONS = "topic_notifications"
# Directory in which to write our log file.
PARAM_OUTPUT_DIR = "log_output_dir"


###############################################################################

# Expected string name for our team.
TEAM_NAME = "KITWARE"

# Mapping of our recipe task name to expected logged value.
RECIPE_TO_ID = {
    "Pinwheel": "A",
    "Coffee": "B",
    "Tea": "C",
    "Oatmeal": "D",
    "Dessert Quesadilla": "E",
}
RECIPE_NULL = "null"

NO_STEP_NUMBER = "null"

RE_ERR_DESC = re.compile(
    r"Recipe: (?P<task_name>.*), activity: (?P<activity_str>.*), "
    r"broad step: \(id=(?P<broad_step_id>\d+)\) (?P<broad_step_str>.*)$"
)

STATUS_ACTIVE = "active"
STATUS_ERROR = "error"
STATUS_NULL = "null"


###############################################################################


def ts_str(t: Optional[float] = None) -> str:
    """
    Generate "now" timestamp as a string, used in both filename and log lines.
    :return: String "now" timestamp.
    """
    if t is None:
        t = time.time()
    tl = time.localtime(t)
    ts_fmt = time.strftime(r"%Y-%m-%dT%H:%M:%S.{decimal}Z", tl)
    # `.3f` format will guarantee a decimal point in the string even if `t` is
    # an effective integer or in the [0,1] range. `modf[0]` will always be less
    # than 1, thus can guarantee returned value string will always start with
    # "0.", using `[2:]` to get string after the decimal point.
    return ts_fmt.format(decimal=f"{math.modf(t)[0]:.3f}"[2:])


class Eval2LoggingNode(Node):
    __doc__ = __doc__  # equal to module doc-string.

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        params = declare_and_get_parameters(
            self,
            [
                (PARAM_TOPIC_TASK_UPDATES,),
                (PARAM_TOPIC_NOTIFICATIONS,),
                (PARAM_OUTPUT_DIR,),
            ],
        )

        # Unix timestamp of this "trial" for logging.
        self._trial_timestamp = t = time.time()
        self._log_output_dir = Path(params[PARAM_OUTPUT_DIR])
        self._log_output_filepath = (
            self._log_output_dir / f"kitware_trial_log_{ts_str(t)}.log"
        )
        log.info(f"Writing to log file: {self._log_output_filepath}")

        # Open file to our logging lines to. Open in non-binary mode for
        # writing.
        self._log_file = open(self._log_output_filepath, "w")
        self._log_csv = csv.writer(self._log_file)
        # Lock for synchronizing log file writing.
        self._log_lock = RLock()

        self._sub_task_update = self.create_subscription(
            TaskUpdate,
            params[PARAM_TOPIC_TASK_UPDATES],
            self.cb_task_update,
            1,
        )
        self._sub_error_notifications = self.create_subscription(
            AruiUserNotification,
            params[PARAM_TOPIC_NOTIFICATIONS],
            self.cb_arui_notification,
            1,
        )
        log.info("Init complete")

    def log_line(self, t, task_name, step_number, current_status, comment=None) -> None:
        """
        Log an individual line to the file.

        Thread-safe.

        :param t: unix time the logging is associated with.
        :param task_name: Recipe task name. This is expected to follow the\
            given spec, otherwise we will not log anything.
        :param step_number:
        :param current_status:
        :param comment: Optional additional comment, like what an error is
            about.
        """
        log = self.get_logger()

        # Translate inputs into required format values
        try:
            recipe_id = (
                RECIPE_TO_ID[task_name] if task_name != RECIPE_NULL else task_name
            )
        except KeyError:
            log.error(
                f'No recipe identifier for task name "{task_name}". '
                f"Skipping logging. Otherwise input: "
                f"step_number={step_number}, current_status={current_status}, "
                f"comment={comment}"
            )
            return

        row = [
            ts_str(t),
            TEAM_NAME,
            recipe_id,
            step_number,
            current_status,
        ]
        if comment is not None:
            row.append(comment)

        log.info(f"Logging row: {row}")

        with self._log_lock:
            if not self._log_file.closed:
                self._log_csv.writerow(row)
                self._log_file.flush()

    def cb_task_update(self, msg: TaskUpdate) -> None:
        log = self.get_logger()

        t = time_to_float(msg.latest_sensor_input_time)

        # If we are on step "0" and current_step="background", we are in the
        # background state, in which state the logging wants "nulls" in places.
        if msg.current_step_id == 0 and msg.current_step == "background":
            # In background, transmits "null"s appropriately
            self.log_line(t, RECIPE_NULL, NO_STEP_NUMBER, STATUS_NULL)
        else:
            if (
                msg.task_name == "Pinwheel"
                and msg.current_step_id == 10
                and not msg.completed_steps[-1]
            ):
                # Known special case for Pinwheel task where we have omitted a
                # step in our configuration due to algorithm performance.
                self.log_line(t, RECIPE_NULL, NO_STEP_NUMBER, STATUS_NULL)
                self.log_line(t, msg.task_name, msg.current_step_id + 2, STATUS_ACTIVE)
            elif np.all(msg.completed_steps):
                # If all steps are completed, output nulls to indicate the
                # final "done" state.
                self.log_line(
                    t,
                    RECIPE_NULL,
                    NO_STEP_NUMBER,
                    STATUS_NULL,
                    f"{msg.task_name} task completed",
                )
            else:
                # Emit a null line to indicate the previous task was completed,
                # Except if the last step was background.
                if msg.previous_step != "background":
                    self.log_line(
                        t,
                        RECIPE_NULL,
                        NO_STEP_NUMBER,
                        STATUS_NULL,
                        f"Stopped performing: {msg.previous_step}",
                    )
                self.log_line(
                    t,
                    msg.task_name,
                    # Steps are 0-index based coming out of the TaskMonitor, bring
                    # it back into 1-indexed for logging spec.
                    msg.current_step_id + 1,
                    STATUS_ACTIVE,
                    f"Started performing: {msg.current_step}",
                )

    def cb_arui_notification(self, msg: AruiUserNotification) -> None:
        # "Error" notification message has the broad step ID in it, so we can
        # parse that out via regex.
        if msg.context == AruiUserNotification.N_CONTEXT_TASK_ERROR:
            m = RE_ERR_DESC.search(msg.description)
            if m:
                md = m.groupdict()
                self.log_line(
                    time.time(),
                    md["task_name"],
                    md["broad_step_id"],
                    STATUS_ERROR,
                    msg.description,
                )
            else:
                self.get_logger().error(
                    f"Failed to parse error notification for logging: "
                    f"{msg.description}"
                )

    def destroy_node(self) -> None:
        """
        Clean-up resources
        """
        log = self.get_logger()
        with self._log_lock:
            log.info(f"Closing log file: {self._log_file.name}")
            self._log_file.close()
        super().destroy_node()


main = make_default_main(Eval2LoggingNode)


if __name__ == "__main__":
    main()
