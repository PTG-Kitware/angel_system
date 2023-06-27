import pyaudio
import requests
import tempfile
import threading
import time
import wave

from nltk.tokenize import sent_tokenize
import rclpy
from rclpy.node import Node
import simpleaudio as sa

from angel_msgs.msg import HeadsetAudioData, Utterance


AUDIO_TOPIC = "audio_topic"
UTTERANCES_TOPIC = "utterances_topic"
ASR_SERVER_URL = "asr_server_url"
ASR_REQ_SEGMENT_SECONDS_DURATION = "asr_req_segment_duration"


class ASR(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            AUDIO_TOPIC,
            UTTERANCES_TOPIC,
            ASR_SERVER_URL,
            ASR_REQ_SEGMENT_SECONDS_DURATION,
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
        self.segmentation_duration = self.get_parameter(
            ASR_REQ_SEGMENT_SECONDS_DURATION
        ).value
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
            f"({type(self.segmentation_duration).__name__}) "
            f"{self.segmentation_duration}"
        )

        self.subscription = self.create_subscription(
            HeadsetAudioData, self._audio_topic, self.listener_callback, 1
        )

        # TODO(derekahmed): Add internal buffering to reduce publisher queue
        # size to 1.
        self._publisher = self.create_publisher(Utterance, self._utterances_topic, 10)

        self.audio_stream = bytearray()
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

    def listener_callback(self, msg):
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
            if self.audio_stream_duration >= self.segmentation_duration and not (
                self.t.is_alive()
            ):
                # Make a copy of the current data so we can run ASR
                # while more data is accumulated.
                audio_data = self.audio_stream

                # Remove the data that we just copied from the stream.
                self.audio_stream = bytearray()
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
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
            self.log.info(f"Writing audio to temporary file: {temp_file.name}")
            wav_write = wave.open(temp_file, "wb")
            wav_write.setnchannels(num_channels)
            # TODO (derekahmed) We should figure out how this value was derived
            # and make this a constant accordingly.
            wav_write.setsampwidth(4)  # Derived from audio_player.py
            wav_write.setframerate(sample_rate)
            wav_write.writeframes(audio_data)

            with open(temp_file.name, "rb") as temp_file:
                response = requests.post(
                    self._asr_server_url, files={"audio_data": temp_file}
                )
            temp_file.close()

            if not response.ok:
                self.log.error("ASR Server Response contains an error.")
                return
            if response:
                self.log.info("Complete ASR text is:\n" + f'"{response.text}"')
                for sentence in sent_tokenize(response.text):
                    utterance_msg = Utterance()
                    utterance_msg.value = sentence
                    self.log.info("Publishing message: " + f'"{sentence}"')
                    self._publisher.publish(utterance_msg)


def main():
    rclpy.init()

    asr = ASR()

    rclpy.spin(asr)

    asr.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
