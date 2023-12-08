from typing import Set

import numpy as np

from angel_system.data.common.config_structs import load_object_label_set

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils import declare_and_get_parameters, RateTracker
from angel_utils import make_default_main
from angel_utils.object_detection import max_labels_and_confs


# The filepath to the object labels config that defines the labels to let
# "pass through" this filter.
PARAM_CONFIG_FP = "object_labels_config"
# Input topic name
PARAM_TOPIC_INPUT = "topic_input"
# Output topic name
PARAM_TOPIC_OUTPUT = "topic_output"


class ObjectDetectionFilterNode(Node):
    """
    Node that will filter received ObjectDetection2dSet messages to only
    contain those included in the specified input configuration.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)

        param_vales = declare_and_get_parameters(
            self,
            (
                (PARAM_CONFIG_FP,),
                (PARAM_TOPIC_INPUT,),
                (PARAM_TOPIC_OUTPUT,),
            ),
        )

        # Load object labels structure. Store set of labels to filter
        # detections by.
        object_label_set = load_object_label_set(param_vales[PARAM_CONFIG_FP])
        self._label_whitelist: Set[str] = {ol.label for ol in object_label_set.labels}

        self._rt = RateTracker()

        # Establish subscriber/publisher
        self._pub = self.create_publisher(
            ObjectDetection2dSet,
            param_vales[PARAM_TOPIC_OUTPUT],
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self._sub = self.create_subscription(
            ObjectDetection2dSet,
            param_vales[PARAM_TOPIC_INPUT],
            self.filter,
            1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

    def filter(self, msg: ObjectDetection2dSet):
        """
        Filter messages and push them back out.
        """
        log = self.get_logger()
        label_whitelist_set = self._label_whitelist

        det_indices = []
        all_max_labels = max_labels_and_confs(msg)[0]
        for det_i, max_label in enumerate(all_max_labels):
            if max_label in label_whitelist_set:
                det_indices.append(det_i)

        if len(det_indices) != msg.num_detections:
            log.info(
                f"Filtering input detections from {all_max_labels.tolist()} "
                f"to {all_max_labels[det_indices].tolist()}"
            )

            # Create a new message only containing the detections with labels in
            # the white-list set.
            new_msg = ObjectDetection2dSet()
            new_msg.header = msg.header
            new_msg.source_stamp = msg.source_stamp
            new_msg.label_vec = msg.label_vec

            new_msg.num_detections = len(det_indices)

            # Only array slice if there is anything left post-filtering.
            if new_msg.num_detections > 0:
                new_msg.left = np.asarray(msg.left)[det_indices].tolist()
                new_msg.right = np.asarray(msg.right)[det_indices].tolist()
                new_msg.top = np.asarray(msg.top)[det_indices].tolist()
                new_msg.bottom = np.asarray(msg.bottom)[det_indices].tolist()
                new_msg.label_confidences = (
                    np.asarray(msg.label_confidences)
                    .reshape(msg.num_detections, len(msg.label_vec))[det_indices, :]
                    .ravel()
                    .tolist()
                )

            msg = new_msg

        else:
            log.info(
                "All input detections passed filter, simply forwarding input "
                "message."
            )

        self._pub.publish(msg)

        self._rt.tick()
        log.info(
            f"Detection Filter Rate: {self._rt.get_rate_avg()} Hz",
        )


# Don't really want to use *all* available threads...
# 3 threads because:
# - 1 known subscriber which has their own group
# - 1 for default group
# - 1 for publishers
main = make_default_main(ObjectDetectionFilterNode, multithreaded_executor=3)


if __name__ == "__main__":
    main()
