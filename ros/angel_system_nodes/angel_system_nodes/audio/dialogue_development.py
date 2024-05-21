from termcolor import colored
import time
from typing import List

from angel_msgs.msg import DialogueUtterance
from angel_system_nodes.audio import dialogue
from angel_utils import declare_and_get_parameters
from angel_utils import make_default_main


INTERVAL_SECONDS = "interval_seconds"
OUTPUT_TOPIC = "dev_dialogue_output_topic"

MESSAGES = [
    "The following messages are for development and debugging use only.",
    "Every 3-5 seconds, a new message will be emitted from this node.",
    "This will provide data to downstream nodes in the absence of rosbag data."
]

class DevelopmentDialoguePublisherNode(dialogue.AbstractDialogueNode):
    """
    This node is for development purposes and will continously output
    textual utterances for downstream dialogue system processing.
    """

    def __init__(self):
        super().__init__()

        # Handle parameterization.
        param_values = declare_and_get_parameters(
            self,
            [
                (OUTPUT_TOPIC,),
                (INTERVAL_SECONDS, 5)
            ],
        )

        self._output_topic = param_values[OUTPUT_TOPIC]
        self._interval_seconds = int(param_values[INTERVAL_SECONDS])

        # Handle subscription/publication topics.
        self._publisher = self.create_publisher(
            DialogueUtterance, self._output_topic, 1
        )
        self._forever_publish_messages(MESSAGES)

    def _forever_publish_messages(self, messages: List[str]):
        """
        Handles message publishing for an utterance with a detected emotion classification.
        """
        while True:
            for message_text in messages:
                development_msg = DialogueUtterance()
                development_msg.header.frame_id = "Development Dialogue Publisher Node"
                development_msg.header.stamp = self.get_clock().now().to_msg()
                development_msg.utterance_text = message_text
                self._publisher.publish(development_msg)
                colored_utterance = colored(message_text, "light_blue")
                self.log.info(
                    f'Publishing \"{colored_utterance}\" to {self._output_topic}.')
                time.sleep(self._interval_seconds)


main = make_default_main(DevelopmentDialoguePublisherNode)

if __name__ == "__main__":
    main()
