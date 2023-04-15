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


IN_AUDIO_TOPIC = "input_audio_topic"
OUT_VOICE_ACTIVITY_TOPIC = "output_voice_activity_topic"
VAD_SERVER_URL = "vad_server_url"

#This parameter controls how frequently Voice Activity Detection should be
# queried, e.g. every 1s of Input Audio should be accumulated for voice
# activity detection.
INTERVAL_UNIT_SEC_DURATION = "interval_unit_sec_duration"

# TODO (derekahmed) We should figure out how this value was derived
# and make this a constant accordingly.
WAV_SAMPLE_WIDTH = 4  # Derived from audio_player.py

class VoiceActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_AUDIO_TOPIC,
            OUT_VOICE_ACTIVITY_TOPIC,
            VAD_SERVER_URL
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
        
        self._in_audio_topic =\
            self.get_parameter(IN_AUDIO_TOPIC).get_parameter_value().string_value
        self._out_voice_activity_topic =\
            self.get_parameter(OUT_VOICE_ACTIVITY_TOPIC).get_parameter_value().string_value
        self._asr_server_url =\
            self.get_parameter(VAD_SERVER_URL).get_parameter_value().string_value
        self.log.info(f"Input Audio topic: "
                      f"({type(self._in_audio_topic).__name__}) "
                      f"{self._in_audio_topic}")
        self.log.info(f"Output Voice Activity topic: "
                      f"({type(self._out_voice_activity_topic).__name__}) "
                      f"{self._out_voice_activity_topic}")
        self.log.info(f"Columbia Voice Activity Detection Server URL: "
                      f"({type(self._asr_server_url).__name__}) "
                      f"{self._asr_server_url}")

        self.subscription = self.create_subscription(
            HeadsetAudioData,
            self._in_audio_topic,
            self.listener_callback,
            1)
        
        # TODO(derekahmed): Add internal buffering to reduce publisher queue
        # size to 1.
        self._publisher = self.create_publisher(
            HeadsetAudioData,
            self._out_voice_activity_topic,
            10
        )

        self.audio_stream = bytearray()
        self.t = threading.Thread()
        self.prev_timestamp = None
        self.audio_stream_duration = 0.0
        # This locks self.audio_stream, audio_stream_duration,
        # self.prev_timestamp, and self.t to faciilitate accumulation of
        # audio data prior to ASR processing.
        self.audio_stream_lock = threading.RLock()

        # This locks Voice Activity Detection (VAD) server requesting 
        # (requests on self._vad_server_url)
        # to ensure only 1 VAD request gets processed at a time.
        # TODO: Future iterations of this node will phase out use of
        # this external server.
        self.vad_server_lock = threading.RLock()

    def listener_callback(self, msg):
        pass

    def vad_server_request_thread(self, audio_data, num_channels, sample_rate):
        pass

def main():
    rclpy.init()
    vad = VoiceActivityDetector()
    rclpy.spin(vad)    
    vad.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
