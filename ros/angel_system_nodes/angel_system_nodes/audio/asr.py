import json
import queue
import requests
import soundfile
import tempfile
from termcolor import colored
import threading
import base64
import PIL.Image
import io
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import rclpy

from angel_msgs.msg import HeadsetAudioData, DialogueUtterance, SystemTextResponse
from angel_system_nodes.audio import dialogue
from angel_utils import make_default_main

BRIDGE = CvBridge()

AUDIO_TOPIC = "audio_topic"
UTTERANCES_TOPIC = "utterances_topic"
FEEDBACK_TOPIC = "feedback_topic"
ASR_SERVER_URL = "asr_server_url"
ASR_REQ_SEGMENT_SECONDS_DURATION = "asr_req_segment_duration"
IS_SENTENCE_TOKENIZE = "is_sentence_tokenize"
DEBUG_MODE = "debug_mode"
IMAGE_TOPIC = "image_topic"

# TODO (derekahmed) We should figure out how this value was derived
# and make this a constant accordingly.
WAV_SAMPLE_WIDTH = 4  # Derived from audio_player.py


class ASR(dialogue.AbstractDialogueNode):
    def __init__(self):
        super().__init__()

        parameter_names = [
            AUDIO_TOPIC,
            UTTERANCES_TOPIC,
            FEEDBACK_TOPIC,
            ASR_SERVER_URL,
            ASR_REQ_SEGMENT_SECONDS_DURATION,
            IS_SENTENCE_TOKENIZE,
            DEBUG_MODE,
            IMAGE_TOPIC,
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        # Check for not-set parameters.
        some_not_set = False
        for p in set_parameters:
            if p.type_ is rclpy.parameter.Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")

        self._audio_topic = (
            self.get_parameter(AUDIO_TOPIC).get_parameter_value().string_value
        )
        self._utterances_topic = (
            self.get_parameter(UTTERANCES_TOPIC).get_parameter_value().string_value
        )
        self._asr_server_url = (
            self.get_parameter(ASR_SERVER_URL).get_parameter_value().string_value
        )
        self._segmentation_duration = self.get_parameter(
            ASR_REQ_SEGMENT_SECONDS_DURATION
        ).value
        self._is_sentence_tokenize_mode = (
            self.get_parameter(IS_SENTENCE_TOKENIZE).get_parameter_value().bool_value
        )
        self._debug_mode = (
            self.get_parameter(DEBUG_MODE).get_parameter_value().bool_value
        )
        self._feedback_topic = (
            self.get_parameter(FEEDBACK_TOPIC).get_parameter_value().string_value
        )
        self._image_topic = (
            self.get_parameter(IMAGE_TOPIC).get_parameter_value().string_value
        )

        self.log.info(
            f"Audio topic: "
            f"({type(self._audio_topic).__name__}) "
            f"{self._audio_topic}"
        )
        self.log.info(
            f"Utterances topic: "
            f"({type(self._utterances_topic).__name__}) "
            f"{self._utterances_topic}"
        )
        self.log.info(
            f"Columbia ASR Server URL: "
            f"({type(self._asr_server_url).__name__}) "
            f"{self._asr_server_url}"
        )
        self.log.info(
            f"Columbia ASR Server Request Segmentation Duration: "
            f"({type(self._segmentation_duration).__name__}) "
            f"{self._segmentation_duration}"
        )
        self.log.info(
            f"Is Sentence Tokenization On: "
            f"({type(self._is_sentence_tokenize_mode).__name__}) "
            f"{self._is_sentence_tokenize_mode}"
        )
        self.log.info(
            f"Is Debugging On: "
            f"({type(self._debug_mode).__name__}) "
            f"{self._debug_mode}"
        )

        # Handle subscription/publication topics.
        self.subscription = self.create_subscription(
            HeadsetAudioData, self._audio_topic, self.listener_callback, 1
        )
        self._publisher = self.create_publisher(
            DialogueUtterance, self._utterances_topic, 1
        )
        self._feedback_publisher = self.create_publisher(
            SystemTextResponse, self._feedback_topic, 1
        )

        # Single slot for latest image message to process detection over.
        self.pov_frame: Image = ""

        self.log.info("Creating subscription to image topic")
        # Initialize ROS hooks
        self.subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.process_image_callback,
            1,
        )

        self.audio_stream = []
        self.t = threading.Thread()
        self.prev_timestamp = None
        self.audio_stream_duration = 0.0
        # This locks self.audio_stream, audio_stream_duration,
        # self.prev_timestamp, and self.t to faciilitate accumulation of
        # audio data prior to ASR processing.
        self.audio_stream_lock = threading.RLock()

        # This locks ASR server requesting (requests on self._asr_server_url)
        # to ensure only 1 ASR request gets processed at a time.
        # TODO: Future iterations of this node will phase out use of
        # this external server.
        self.asr_server_lock = threading.RLock()

        self.message_queue = queue.Queue()
        self.handler_thread = threading.Thread(target=self.process_message_queue)
        self.handler_thread.start()

    def listener_callback(self, msg):
        self.message_queue.put(msg)

    def process_image_callback(self, image: Image):
        # image is type sensor_msgs.msg encoding BGR8
        img0 = BRIDGE.imgmsg_to_cv2(image, desired_encoding="bgr8")

        # Convert img0 into RGB and create a PIL image instance.
        img_rgb = PIL.Image.fromarray(img0[:, :, ::-1], mode="RGB")
        img_rgb = img_rgb.resize(np.divide(img_rgb.size, 4).astype(int))
        jpg_container = io.BytesIO()
        img_rgb.save(jpg_container, format="JPEG")
        self.pov_frame = base64.b64encode(jpg_container.getvalue()).decode("utf-8")

    def process_message_queue(self):
        while True:
            msg = self.message_queue.get()
            with self.audio_stream_lock:
                # Check that this message comes temporally after the previous message.
                message_order_valid = False
                if self.prev_timestamp is None:
                    message_order_valid = True
                else:
                    if msg.header.stamp.sec > self.prev_timestamp.sec:
                        message_order_valid = True
                    elif msg.header.stamp.sec == self.prev_timestamp.sec:
                        # Seconds are same, so check nanoseconds.
                        if msg.header.stamp.nanosec > self.prev_timestamp.nanosec:
                            message_order_valid = True

                if message_order_valid:
                    self.audio_stream.extend(msg.data)
                    self.audio_stream_duration += msg.sample_duration
                else:
                    self.log.info(
                        "Warning! Out of order messages.\n"
                        + f"Prev: {self.prev_timestamp} \nCurr: {msg.header.stamp}"
                    )
                    return

                self.prev_timestamp = msg.header.stamp
                if self.audio_stream_duration >= self._segmentation_duration and not (
                    self.t.is_alive()
                ):
                    # Make a copy of the current data so we can run ASR
                    # while more data is accumulated.
                    audio_data = self.audio_stream

                    # Remove the data that we just copied from the stream.
                    self.audio_stream = []
                    self.audio_stream_duration = 0.0

                    # Start the ASR server request thread.
                    self.log.info("Spawning ASR request thread.")
                    self.t = threading.Thread(
                        target=self.asr_server_request_thread,
                        args=(audio_data, msg.channels, msg.sample_rate),
                    )
                    self.t.start()

    def asr_server_request_thread(self, audio_data, num_channels, sample_rate):
        with self.asr_server_lock:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav", delete=(not self._debug_mode)
            )
            self.log.info(f"Writing audio to temporary file: {temp_file.name}")
            channels_data = []
            for channel_i in range(num_channels):
                channels_data.append(audio_data[channel_i::num_channels])
            channels_data = np.array(channels_data).T
            soundfile.write(temp_file, channels_data, sample_rate)

            with open(temp_file.name, "rb") as temp_file:
                response = requests.post(
                    self._asr_server_url,
                    files={"audio_data": temp_file},
                    params={"preprocessing": "mfcc_up"},
                )
            if not self._debug_mode:
                temp_file.close()

            if not response.ok:
                self.log.error("ASR Server Response contains an error.")
                return
            if response:
                response_text = json.loads(response.text)["text"]
                self.log.info("Complete ASR text is:\n" + f'"{response_text}"')
                if self._is_sentence_tokenize_mode:
                    for sentence in sent_tokenize(response_text):
                        self._publish_text(sentence)
                else:
                    self._publish_text(response_text)

    def _publish_text(self, text: str):

        self.log.info("Utterance was: " + f'"{text}"')

        if (
            "angela" not in text.lower()
            and "angel" not in text.lower()
            and "angela," not in text.lower()
            and "angel," not in text.lower()
        ):
            # If Angel keyword is not found, don't publish the utterance
            return

        self.log.info("Publish thinking feedback")
        self.publish_feedback_response()

        published_msg = DialogueUtterance()
        published_msg.header.frame_id = "ASR"
        published_msg.header.stamp = self.get_clock().now().to_msg()

        # Find the index of the first occurrence of the word
        result_text = text
        keywords = ["angela", "angela,", "angel", "angel,"]
        for word in keywords:
            index = text.lower().find(word)
            if index != -1:
                # Remove everything before the word
                result_text = text[index + 6 :]
                break

        published_msg.utterance_text = result_text

        if self.pov_frame is None or len(self.pov_frame) <= 1:
            published_msg.pov_frame = ""
            self.log.info("No pov frame available")
        else:
            published_msg.pov_frame = self.pov_frame
            self.log.info("Adding pov frame to utterance..")
        colored_utterance = colored(published_msg.utterance_text, "light_blue")
        self.log.info("Publishing message: " + f'"{colored_utterance}"')

        self._publisher.publish(published_msg)

    def publish_feedback_response(self):
        publish_msg = SystemTextResponse()
        publish_msg.header.frame_id = "GPT thinking"
        publish_msg.header.stamp = self.get_clock().now().to_msg()
        publish_msg.utterance_text = ""
        publish_msg.response = "thinking"
        self._feedback_publisher.publish(publish_msg)


main = make_default_main(ASR)


if __name__ == "__main__":
    main()
