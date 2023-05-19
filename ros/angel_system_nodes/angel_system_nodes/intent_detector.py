import queue
import rclpy
from rclpy.node import Node
import threading

from angel_msgs.msg import InterpretedAudioUserIntent, Utterance

# Please refer to labels defined in
# https://docs.google.com/document/d/1uuvSL5de3LVM9c0tKpRKYazDxckffRHf7IAcabSw9UA .
NEXT_STEP_KEYPHRASES = ['skip', 'next', 'next step']
PREV_STEP_KEYPHRASES = ['previous', 'previous step', 'last step', 'go back']
QUESTION_KEYPHRASES = ['question', '?']
OVERRIDE_KEYPHRASES = ['angel', 'angel system']

# TODO(derekahmed): Please figure out how to keep this sync-ed with
# config/angel_system_cmds/user_intent_to_sys_cmd_v1.yaml.
LABELS = [
    "Go to next step",
    "Go to previous step",
    "User inquiry"
]

UTTERANCES_TOPIC = "utterances_topic"
PARAM_EXPECT_USER_INTENT_TOPIC = "expect_user_intent_topic"
PARAM_INTERP_USER_INTENT_TOPIC = "interp_user_intent_topic"

class IntentDetector(Node):
    '''
    As of Q12023, intent detection is derived heuristically. This will be shifted
    to a model-based approach in the near-future.
    '''

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        # Handle parameterization. 
        parameter_names = [
            UTTERANCES_TOPIC,
            PARAM_EXPECT_USER_INTENT_TOPIC,
            PARAM_INTERP_USER_INTENT_TOPIC,
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._utterances_topic = \
            self.get_parameter(UTTERANCES_TOPIC).value
        self._expect_uintent_topic = \
            self.get_parameter(PARAM_EXPECT_USER_INTENT_TOPIC).value
        self._interp_uintent_topic = \
            self.get_parameter(PARAM_INTERP_USER_INTENT_TOPIC).value
        self.log.info(f"Utterances topic: "
                      f"({type(self._utterances_topic).__name__}) "
                      f"{self._utterances_topic}")
        self.log.info(f"Expected User Intent topic: "
                      f"({type(self._expect_uintent_topic).__name__}) "
                      f"{self._expect_uintent_topic}")
        self.log.info(f"Interpreted User Intent topic: "
                      f"({type(self._interp_uintent_topic).__name__}) "
                      f"{self._interp_uintent_topic}")
        
        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            Utterance,
            self._utterances_topic,
            self.listener_callback,
            1)
        self._expected_publisher = self.create_publisher(
            InterpretedAudioUserIntent,
            self._expect_uintent_topic,
            1)
        self._interp_publisher = self.create_publisher(
            InterpretedAudioUserIntent,
            self._interp_uintent_topic,
            1)

        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

    def listener_callback(self, msg):
        '''
        This is the main ROS node listener callback loop that will process
        all messages received via subscribed topics.
        '''
        self.log.info(f"Received message:\n\n{msg.value}")
        self.message_queue.put(msg)

    def process_message_queue(self):
        '''
        Constant loop to process received messages.
        '''
        while True:
            msg = self.message_queue.get()
            self.log.info(f"Processing message:\n\n{msg.value}")
            intent, score = self.detect_intents(msg)
            if not intent:
                continue
            self.publish_msg(msg.value, intent, score)

    def detect_intents(self, msg):
        '''
        Core logic for intent detection and publishing.
        '''
        lower_utterance = msg.value.lower()
        interp_intents = []
        confidences =  []
        if self._contains_phrase(lower_utterance, NEXT_STEP_KEYPHRASES):
            interp_intents.append(LABELS[0])
            confidences.append(0.5)
        if self._contains_phrase(lower_utterance, PREV_STEP_KEYPHRASES):
            interp_intents.append(LABELS[1])
            confidences.append(0.5)
        if self._contains_phrase(lower_utterance, QUESTION_KEYPHRASES):
            interp_intents.append(LABELS[2])
            confidences.append(0.5)
        
        if not interp_intents:
            self.log.info(f"No intents detected for:\n\n\"{msg.value}\":")
            return None, -1.0
        
        classified_intent = interp_intents[0]
        confidence = confidences[0]
        if len(interp_intents) > 1:
            if LABELS[2] in interp_intents:
                classified_intent = LABELS[2]
            self.log.info(f"Detected multiple intents: \n{interp_intents}\n" +\
                            f"Defaulting: \"{classified_intent}\".")
        return classified_intent, confidence
    
    def publish_msg(self, utterance, intent, score):
        '''
        Handles message publishing for an utterance with a detected intent.
        '''
        intent_msg = InterpretedAudioUserIntent()
        intent_msg.utterance_text = utterance
        intent_msg.user_intent = intent
        intent_msg.confidence = score
        if self._contains_phrase(utterance.lower(), OVERRIDE_KEYPHRASES):
            intent_msg.confidence = 1.0
            self._expected_publisher.publish(intent_msg)
            self.log.info(f"Publishing intents to {PARAM_EXPECT_USER_INTENT_TOPIC} " +\
                          f"for\n\n\"{utterance}\"" +\
                          f"\n\n\"{self._red_font(intent_msg.user_intent)}\": " +\
                          f"{intent_msg.confidence}")
        else:
            self._interp_publisher.publish(intent_msg)
            self.log.info(f"Publishing intents to {PARAM_INTERP_USER_INTENT_TOPIC} " +\
                          f"for\n\n\"{utterance}\"" +\
                          f"\n\n\"{self._red_font(intent_msg.user_intent)}\": " +\
                          f"{intent_msg.confidence}")

    def _contains_phrase(self, utterance, phrases):
        for phrase in phrases:
            if phrase in utterance:
                return True
        return False
    
    def _red_font(self, text):
        return f"\033[91m{text}\033[0m"

def main():
    rclpy.init()
    intent_detector = IntentDetector()
    rclpy.spin(intent_detector)
    intent_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
