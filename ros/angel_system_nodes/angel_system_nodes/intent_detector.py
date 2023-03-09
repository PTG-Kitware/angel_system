import json
import time

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from angel_msgs.msg import IntentDetection, Utterance

# Please refer to labels defined in
# https://docs.google.com/document/d/1uuvSL5de3LVM9c0tKpRKYazDxckffRHf7IAcabSw9UA .
NEXT_STEP_KEYPHRASES = ['skip', 'next step']
PREV_STEP_KEYPHRASES = ['prev step', 'last step', 'go back']
CHECK_OFF_STEP_KEYPHRASES = ['already', 'done']
LABELS = ["Go to Next Step", "Go to Prev Step", "Check off X", "Other"]

class IntentDetector(Node):
    '''
    As of Q12023, intent detection is derived heuristically. This will be shifted
    to a model-based approach soon.
    '''

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("utterances_topic", "Utterances")
        self.declare_parameter("det_topic", "IntentDetections")

        self._utterances_topic = self.get_parameter("utterances_topic").get_parameter_value().string_value
        self._det_topic = self.get_parameter("det_topic").get_parameter_value().string_value
        
        self.subscription = self.create_subscription(
            Utterance,
            self._utterances_topic,
            self.listener_callback,
            10)
        
        self._publisher = self.create_publisher(
            IntentDetection,
            self._det_topic,
            1
        )

    def listener_callback(self, msg):
        log = self.get_logger()
        utterance = msg.value.lower()
        intent_detection_msg = IntentDetection()
        intent_detection_msg.label_vec = LABELS
        if self.contains_phrase(utterance, NEXT_STEP_KEYPHRASES):
            intent_detection_msg.conf_vec = [1.0, 0.0, 0.0, 0.0]
        elif self.contains_phrase(utterance, PREV_STEP_KEYPHRASES):
            intent_detection_msg.conf_vec = [0.0, 1.0, 0.0, 0.0]
        elif self.contains_phrase(utterance, CHECK_OFF_STEP_KEYPHRASES):
            intent_detection_msg.conf_vec = [0.0, 0.0, 1.0, 0.0]
        else:
            intent_detection_msg.conf_vec = [0.0, 0.0, 0.0, 1.0]
        
        log.info(f"Detected intents for \"{utterance}\":\n" +
                 ", ".join([f"\"{label}\": {score}" for label, score in 
                            zip(intent_detection_msg.label_vec, intent_detection_msg.conf_vec)]))
        self._publisher.publish(intent_detection_msg)
        
    def contains_phrase(self, utterance, phrases):
        for phrase in phrases:
            if phrase in utterance:
                return True
        return False


def main():
    rclpy.init()

    intentDetector = IntentDetector()
    
    rclpy.spin(intentDetector)
    
    intentDetector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
