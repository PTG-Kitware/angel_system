from rclpy.node import Node

from angel_msgs.msg import DialogueUtterance


class AbstractDialogueNode(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

    def _copy_dialogue_utterance(
        self, src_msg: DialogueUtterance, node_name: str, copy_time
    ) -> DialogueUtterance:
        msg = DialogueUtterance()

        msg.header.frame_id = node_name
        msg.header.stamp = copy_time

        msg.utterance_text = src_msg.utterance_text
        msg.pov_frame = src_msg.pov_frame

        # Copy all optional fields below.

        # Copy over intent classification information if present.
        if src_msg.intent:
            msg.intent = src_msg.intent
            msg.intent_confidence_score = src_msg.intent_confidence_score

        # Copy over emotion classification information if present.
        if msg.emotion:
            msg.emotion = src_msg.emotion
            msg.emotion_confidence_score = src_msg.emotion_confidence_score

        return msg
