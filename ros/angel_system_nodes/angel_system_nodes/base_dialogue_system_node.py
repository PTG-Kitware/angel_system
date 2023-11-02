from abc import ABC
import rclpy
from rclpy.node import Node

from angel_msgs.msg import DialogueUtterance

class BaseDialogueSystemNode(Node):
    """
    This class is used for all dialogue system nodes to inherit similar
    functionality.
    """
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()
    
    def get_intent_or(self, src_msg: DialogueUtterance, or_value: str = "not available") -> str:
        """
        Returns the src_msg intent classification information. If the value is absent,
        the or_value is passed in.
        """
        return src_msg.intent if src_msg.intent else or_value

    def get_emotion_or(self, src_msg: DialogueUtterance, or_value: str = "not available") -> str:
        """
        Returns the src_msg emotion classification information. If the value is absent,
        the or_value is passed in.
        """
        return src_msg.emotion if src_msg.emotion else or_value

    def copy_dialogue_utterance(self,
                                src_msg: DialogueUtterance,
                                node_name: str = "Dialogue System Node"
                                ) -> DialogueUtterance:
        msg = DialogueUtterance()
        msg.header.frame_id = node_name
        msg.utterance_text = src_msg.utterance_text

        # Assign new time for publication.
        msg.header.stamp = self.get_clock().now().to_msg()

        # Copy over intent classification information if present.
        if src_msg.intent:
            msg.intent = src_msg.intent
            msg.intent_confidence_score = src_msg.intent_confidence_score

        # Copy over intent classification information if present.
        if src_msg.emotion:
            msg.emotion = src_msg.emotion
            msg.emotion_confidence_score = src_msg.emotion_confidence_score

        return msg

def main():
    rclpy.init()
    base_dialogue_node = BaseDialogueSystemNode()
    rclpy.spin(base_dialogue_node)
    base_dialogue_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
