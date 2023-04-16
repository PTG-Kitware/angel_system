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
            VAD_SERVER_URL,
            INTERVAL_UNIT_SEC_DURATION
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
        self.max_interval_duration =\
            self.get_parameter(INTERVAL_UNIT_SEC_DURATION).value
        self._vad_server_url =\
            self.get_parameter(VAD_SERVER_URL).get_parameter_value().string_value
        self.log.info(f"Input Audio topic: "
                      f"({type(self._in_audio_topic).__name__}) "
                      f"{self._in_audio_topic}")
        self.log.info(f"Output Voice Activity topic: "
                      f"({type(self._out_voice_activity_topic).__name__}) "
                      f"{self._out_voice_activity_topic}")
        self.log.info(f"Columbia Voice Activity Detection Server URL: "
                      f"({type(self._vad_server_url).__name__}) "
                      f"{self._vad_server_url}")
        self.log.info(f"Columbia VAD Interval Unit Duration: "
                      f"({type(self.max_interval_duration).__name__}) "
                      f"{self.max_interval_duration}")

        # TODO(derekahmed): Add these as node parameters
        self.sample_rate = -1
        self.num_channels = 2
        self.bytes_per_sample = 4
        self.sec_padding = 0.25
        self.max_utterance_sec_length = 20
        self.debug_mode = True
        self.debugging_n_chunks = 0

        self.subscription = self.create_subscription(
            HeadsetAudioData,
            self._in_audio_topic,
            self.listener_callback,
            3000)
        
        # TODO(derekahmed): Add internal buffering to reduce publisher queue
        # size to 1.
        self._publisher = self.create_publisher(
            HeadsetAudioData,
            self._out_voice_activity_topic,
            3000
        )

        self.request_thread = threading.Thread()

        self.audio_stream = bytearray()
        self.prev_timestamp = None
        self.audio_stream_duration = 0.0
        self.within_interval_accumulation = 0.0
        self.n_intervals = 0
        # This locks self.audio_stream, audio_stream_duration,
        # self.prev_timestamp, and self.t to faciilitate accumulation of
        # audio data prior to ASR processing.
        self.audio_stream_lock = threading.RLock()
        self.msg_i = 0

        # This locks Voice Activity Detection (VAD) server requesting 
        # (requests on self._vad_server_url)
        # to ensure only 1 VAD request gets processed at a time.
        # TODO: Future iterations of this node will phase out use of
        # this external server.
        self.vad_server_lock = threading.RLock()

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
                self.within_interval_accumulation += msg.sample_duration
            else:
                self.log.info("Warning! Out of order messages.\n"
                         + f"Prev: {self.prev_timestamp} \nCurr: {msg.header.stamp}")
                return
            
            self.msg_i = self.msg_i + 1

            self.prev_timestamp = msg.header.stamp
            # Every self._interval_unit_sec_duration query for voice activity
            # detection.
            if self.within_interval_accumulation >= self.max_interval_duration\
                and not(self.request_thread.is_alive()):
                # print(f"It's copying time for segment #{self.n_chunks} @ {self.audio_stream_duration}")
                req_data = self.audio_stream.copy()
                req_duration = self.audio_stream_duration
                req_n_channels = msg.channels
                req_sample_rate = msg.sample_rate
                self.log.info(f"Stream has {self.audio_stream_duration}s with {len(req_data)} floats on {req_n_channels} channels")
                # self.log.info(f"{req_duration} vs {(1 / (req_sample_rate)) * (len(req_data) / (4 * req_n_channels))}")

                # Start the VAD server request thread.
                self.request_thread = threading.Thread(target=self.vad_server_request_thread,
                    args=(req_data, req_n_channels, req_sample_rate, req_duration))
                self.request_thread.start()

                # Reset the within-interval-accumulation counter, regardless
                # if voice activity was detected.
                self.within_interval_accumulation = 0.0
                self.n_intervals += 1

    def vad_server_request_thread(self, audio_data, num_channels, sample_rate,
                                  audio_duration):
        self.log.info(f"Saving audio data ({len(audio_data)}B) ")
        with self.vad_server_lock:
            # TODO(derekahmed) DELETE ME.
            temp_file =\
                self._create_temp_audio_file(audio_data, sample_rate,
                                             num_channels,
                                             prefix=f"main-{self.n_intervals}_")

            # Experiment with a dummy split.
            # TODO change me
            voice_active_segments =\
                self.vad_server_request(temp_file, audio_duration < 7)
            split_timestamp = self.max_utterance_sec_length
            if voice_active_segments:
                # Take the ("next") first segment. If the detected voice
                # activity is reasonably before the end of the audio_data,
                # we can split the data, publish, and continue.
                next_voice_activity_interval = voice_active_segments[0]
                _, end = next_voice_activity_interval
                if end >= audio_duration - self.sec_padding:
                    # Ignore this interval if it is too close to the end because
                    # the user may still be speaking.
                    return
                self._handle_publication(audio_data, end,
                                         self.bytes_per_sample, num_channels,
                                         sample_rate)
                split_timestamp = end
            elif not voice_active_segments and\
                audio_duration > self.max_utterance_sec_length:
                # If no vocal detection has been identified before the
                # utterance end with reasonable padding, but the accumulated
                # audio is too long, publish anyways.
                self._handle_publication(audio_data, 
                                         split_timestamp,
                                         self.bytes_per_sample, num_channels,
                                         sample_rate)
            else:
                return

            # Apply the necessary transformations to the accumulated audio.
            with self.audio_stream_lock:
                split_idx = self._time_to_byte_index(split_timestamp,
                    self.bytes_per_sample, num_channels, sample_rate)
                
                # Update the audio stream to remove the split data.
                debugging_old_duration = self.audio_stream_duration
                self.audio_stream = self.audio_stream[split_idx:]
                self.audio_stream_duration -= split_timestamp
                self.log.info(f"Audio data chopped from {debugging_old_duration} " +\
                            f"-> {self.audio_stream_duration}s...")

            if not self.debug_mode:
                temp_file.close()
    
    def vad_server_request(self, file, 
                         return_empty=False):
        '''
        Assumes there will only be two splits
        TODO(derekahmed) return_empty was a hack that should be removed to test functionality.
        '''
        # with open(file.name, 'rb') as f:
        #     response = requests.post(self._vad_server_url,
        #                             files={'audio_data': f})
        if return_empty:
            return []
        return [[0.0, 6.0], [6.0, 8.0]]
    
    def _handle_publication(self, audio_data, split_timestamp,
                     sample_byte_length, num_channels, sample_rate):
        '''
        TODO(derekahmed): Should publish messages for each split but here
        we will do temporary file saving instead.
        '''
        with self.audio_stream_lock:
            split_idx = self._time_to_byte_index(split_timestamp,
                sample_byte_length, num_channels, sample_rate)
            split_data = audio_data[:split_idx]
            self.log.info(f"Splitting @ {split_timestamp}s: ({split_idx})")
            if self.debug_mode:
                self._create_temp_audio_file(
                    split_data, sample_rate, num_channels,
                    prefix=f"split-{self.debugging_n_chunks}-")
            self.debugging_n_chunks += 1

        # TODO: publish me

    def _create_temp_audio_file(self, audio_data, sample_rate, num_channels,
                                prefix=None):
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', prefix=prefix,
                                                delete=False)
        self.log.info(f"Writing audio data to temporary file: {temp_file.name}")
        wav_write = wave.open(temp_file, 'wb')
        wav_write.setnchannels(num_channels)
        wav_write.setsampwidth(WAV_SAMPLE_WIDTH) 
        wav_write.setframerate(sample_rate)
        wav_write.writeframes(audio_data)
        return temp_file

    def _time_to_byte_index(self, seconds_timestamp,
                            sample_byte_length,
                            num_channels, sample_rate):
        '''
        Maps a timestamp to corresponding index in a bytearray of audio data.
        This mapping requires the sample frequency, the number of Bytes per
        sample, and the number of channels are inherently necessary.
        '''
        # Each sample holds `sample_byte_length` Bytes with `num_channels`
        # interleaved.
        return int((seconds_timestamp * sample_rate)\
            * sample_byte_length * num_channels)

def main():
    rclpy.init()
    vad = VoiceActivityDetector()
    rclpy.spin(vad)    
    vad.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
