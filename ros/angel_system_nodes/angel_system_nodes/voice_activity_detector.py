import json
import pyaudio
import requests
import struct
import tempfile
import threading
import time
import wave

from nltk.tokenize import sent_tokenize
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import simpleaudio as sa

from angel_msgs.msg import HeadsetAudioData, Utterance

FLOAT32_BYTE_LENGTH = 4 # Number of Bytes in a float32.

IN_AUDIO_TOPIC = "input_audio_topic"
OUT_VOICE_ACTIVITY_TOPIC = "output_voice_activity_topic"

# This parameter points to a server that returns a list of [start, end]
# timestamp intervals of detected voice activity. Returned responses can be
# emtpy if no voice activity was detected.
VAD_SERVER_URL = "vad_server_url"

# This parameter controls how frequently Voice Activity Detection should be
# queried. For example, a value of 1 specifies that each second's worth of audio
# should be appended to an accumulating bytearray. The accumulated bytearray
# should then be sent to the VAD_SERVER_URL to request voice activity detection.
VAD_SEC_CADENCE = "vad_cadence"

# This parameter helps determine when accumulated audio that has detected vocal
# inactivity should be published. If detected vocal inactivity begins
# VAD_SEC_MARGIN seconds before the end of the "current" VAD_SEC_CADENCE,
# then vocal activity is likely not ongoing / has ended. The vocal activity
# should then be sliced from accumulated audio and published. Otherwise,
# vocal activity may have been detected and is ongoing. Smaller values
# of this parameter imply more faith in the vocal activity detection model
# at VAD_SERVER_URL.
VAD_SEC_MARGIN = "vad_margin"

# This parameter controls the max length of accumulated audio before publishing.
# If the accumulated audio breaches this threshold, it is published as an audio
# message for downstream consumption.
MAX_ACCUMULATION_SEC_LENGTH = "max_accumulation_length"

class VoiceActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)
        self.log = self.get_logger()

        parameter_names = [
            IN_AUDIO_TOPIC,
            OUT_VOICE_ACTIVITY_TOPIC,
            VAD_SERVER_URL,
            VAD_SEC_CADENCE,
            VAD_SEC_MARGIN,
            MAX_ACCUMULATION_SEC_LENGTH
        ]
        set_parameters = self.declare_parameters(
            namespace="",
            parameters=[(p,) for p in parameter_names],
        )
        # Check for not-set parameters.
        some_not_set = False
        for p in set_parameters:
            if p.type_ is Parameter.Type.NOT_SET:
                some_not_set = True
                self.log.error(f"Parameter not set: {p.name}")
        if some_not_set:
            raise ValueError("Some parameters are not set.")
        
        self._in_audio_topic =\
            self.get_parameter(IN_AUDIO_TOPIC).get_parameter_value().string_value
        self._out_voice_activity_topic =\
            self.get_parameter(OUT_VOICE_ACTIVITY_TOPIC).get_parameter_value().string_value
        self._vad_server_url =\
            self.get_parameter(VAD_SERVER_URL).get_parameter_value().string_value
        self.vad_cadence =\
            self.get_parameter(VAD_SEC_CADENCE).value
        self._vad_sec_margin =\
            self.get_parameter(VAD_SEC_MARGIN).value

        self._max_accumulation_sec_length =\
            self.get_parameter(MAX_ACCUMULATION_SEC_LENGTH).value
        self.log.info(f"Input Audio topic: "
                      f"({type(self._in_audio_topic).__name__}) "
                      f"{self._in_audio_topic}")
        self.log.info(f"Output Voice Activity topic: "
                      f"({type(self._out_voice_activity_topic).__name__}) "
                      f"{self._out_voice_activity_topic}")
        self.log.info(f"Columbia Voice Activity Detection Server URL: "
                      f"({type(self._vad_server_url).__name__}) "
                      f"{self._vad_server_url}")
        self.log.info(f"Columbia VAD Cadence (Seconds): "
                      f"({type(self.vad_cadence).__name__}) "
                      f"{self.vad_cadence}")
        self.log.info(f"Columbia VAD Margin (Seconds): "
                      f"({type(self._vad_sec_margin).__name__}) "
                      f"{self._vad_sec_margin}")
        
        self.is_audio_metadata_set = False
        self.sample_rate = -1
        self.num_channels = -1
        self.bytes_per_sample = FLOAT32_BYTE_LENGTH
        
        self.debug_mode = True # Set this field to turn on debugging.
        # Used to keep track of number of vocal activity detection splits.
        self.split_counter = 0

        # TODO(derekahmed): Add internal buffering to reduce subscriber queue
        # size to 1.
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
        self.prev_timestamp = None
        self.accumulated_audio = bytearray()
        self.accumulated_timestamps = []
        self.accumulated_audio_duration = 0.0
        self.intracadence_duration = 0.0
        self.n_cadence_steps = 0
        # This locks self.audio_stream, audio_stream_duration,
        # self.prev_timestamp, and self.t to faciilitate accumulation of
        # audio data prior to ASR processing.
        self.audio_stream_lock = threading.RLock()
        self.msg_i = 0

        # This locks Voice Activity Detection (VAD) server requesting
        # to ensure only 1 VAD request gets processed at a time.
        # TODO: Future iterations of this node will phase out use of
        # this external server.
        self.vad_server_lock = threading.RLock()

    def listener_callback(self, msg):

        self._check_and_update_audio_metadata_fields(msg)
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
                self.accumulated_audio.extend(msg.data)
                self.accumulated_timestamps.append(msg.header.stamp)
                self.accumulated_audio_duration += msg.sample_duration
                self.intracadence_duration += msg.sample_duration
            else:
                self.log.info("Warning! Out of order messages.\n"
                         + f"Prev: {self.prev_timestamp} \nCurr: {msg.header.stamp}")
                return
            
            self.msg_i = self.msg_i + 1
            self.prev_timestamp = msg.header.stamp
            if self.intracadence_duration >= self.vad_cadence\
                and not(self.request_thread.is_alive()):
                self.log.info(f"{self.n_cadence_steps}-th cadence occurred " +\
                              f"with {self.accumulated_audio_duration} sec " +\
                              "currently accumulated. Requesting voice activity detection.")
                
                # Make a copy of the current data so we can run VAD
                # while more data is accumulated.
                req_data = self.accumulated_audio[:]
                req_duration = self.accumulated_audio_duration

                # Start the VAD server request thread.
                self.request_thread = threading.Thread(target=self.vad_server_request_thread,
                    args=(req_data, self.num_channels, self.sample_rate,
                          req_duration))
                self.request_thread.start()

                # Reset the intercadence-accumulation counter, regardless
                # if voice activity was detected.
                self.intracadence_duration = 0.0
                self.n_cadence_steps += 1

    def vad_server_request_thread(self, audio_data, num_channels, sample_rate,
                                  audio_duration):
        with self.vad_server_lock:

            temp_file = self._create_temp_audio_file(
                audio_data, sample_rate, num_channels,
                prefix=f"main-{self.n_cadence_steps}_")
            voice_active_segments = self.vad_server_request(temp_file)
            split_timestamp = self._max_accumulation_sec_length
            if voice_active_segments:
                # Take the ("next") first segment. If the detected voice
                # activity ends reasonably before the end of the audio_data,
                # we can publish the audio data before the point of inactivity.
                next_voice_activity_interval = voice_active_segments[0]
                _, end = next_voice_activity_interval
                if end >= audio_duration - self._vad_sec_margin:
                    self.log.info(f"Vocal segment end={end} while " +\
                                  f"audio_duration={audio_duration} with  "+\
                                  f"margin={self._vad_sec_margin}.")
                    return
                self._handle_publication(audio_data, end,
                                         self.bytes_per_sample, num_channels,
                                         sample_rate)
                split_timestamp = end
            elif not voice_active_segments and\
                audio_duration > self._max_accumulation_sec_length:
                # If no vocal detection was identified but the accumulated
                # audio is too long, publish anyways.
                self._handle_publication(audio_data, 
                                         split_timestamp,
                                         self.bytes_per_sample, num_channels,
                                         sample_rate)
            else:
                return

            # Update the accumulated audio stream to remove any split data.
            self._split_audio(split_timestamp, num_channels, sample_rate)
            if not self.debug_mode:
                temp_file.close()

    def _split_audio(self, split_timestamp, num_channels, sample_rate):
        '''
        Split the accumulated audio given a timestamp for the split.
        Also updates the accumulated timestamps
        '''
        split_idx = self._time_to_index(split_timestamp, sample_rate)
        split_byte_idx = self._time_to_byte_index(split_timestamp,
                    self.bytes_per_sample, num_channels, sample_rate)
        with self.audio_stream_lock:
            old_duration = self.accumulated_audio_duration
            self.accumulated_audio = self.accumulated_audio[split_byte_idx:]
            self.accumulated_timestamps =\
                self.accumulated_timestamps[split_idx:]
            self.accumulated_audio_duration -= split_timestamp
            self.log.info(f"Audio data chopped from {old_duration} " +\
                                    f"-> {self.accumulated_audio_duration}s...")
    
    def vad_server_request(self, file):
        '''
        Encloses the VAD server request logic.
        '''
        with open(file.name, 'rb') as f:
            self.log.info("Querying Columbia VAD server...")
            response = requests.post(self._vad_server_url,
                                    files={'audio_data': f})
            if response:
                segments = json.loads(response.content)['segments']
                self.log.info(f"Received VAD response: {segments}")
                return segments
    
    def _handle_publication(self, audio_data, split_timestamp,
                     sample_byte_length, num_channels, sample_rate):
        
        # Obtain the audio "prefix" with detected voice activity (before split).
        audio_prefix = None
        split_byte_idx = self._time_to_byte_index(split_timestamp,
                sample_byte_length, num_channels, sample_rate)
        with self.audio_stream_lock:
            audio_prefix = audio_data[:split_byte_idx]
            self.log.info(f"Split @ {split_timestamp}s: ({split_byte_idx})")            
            if self.debug_mode:
                self.log.info("Writing VOCAL ACTIVITY DETECTED prefix of " +\
                          "currently accumulated audio data to \"split\" " +\
                          "temporary file.")
                self._create_temp_audio_file(
                    audio_prefix, sample_rate, num_channels,
                    prefix=f"split-{self.split_counter}-")
        self.split_counter += 1

        # Publish the audio "prefix" with detected voice activity
        # (before split).
        audio_msg = HeadsetAudioData()
        audio_msg.header.stamp = self.accumulated_timestamps[0]
        audio_msg.header.frame_id = "AudioData"
        audio_msg.channels = num_channels
        audio_msg.sample_rate = sample_rate
        n_samples = split_byte_idx / (num_channels * sample_byte_length)
        audio_msg.sample_duration = n_samples * sample_rate
        audio_msg.data = struct.unpack(
            'f' * int(split_byte_idx / sample_byte_length), audio_prefix)
        self._publisher.publish(audio_msg)
        self.log.info(f"Buffer of {int(split_byte_idx / sample_byte_length)} float32 was published")

    def _create_temp_audio_file(self, audio_data, sample_rate, num_channels,
                                prefix=None):
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav', prefix=prefix, delete=False)
        self.log.info(f"Writing to {temp_file.name}...")
        wav_write = wave.open(temp_file, 'wb')
        wav_write.setnchannels(num_channels)
        wav_write.setsampwidth(FLOAT32_BYTE_LENGTH)
        wav_write.setframerate(sample_rate)
        wav_write.writeframes(audio_data)
        return temp_file

    def _check_and_update_audio_metadata_fields(self, msg):
        '''
        Ideally the affected metadata fields are not updated frequently.
        Such may result due to unexpected changes in the hardware. e.g. a new
        microphone getting used for recording. This functionality encapsulates
        how this is handled (e.g. logging, error raising, etc.)
        '''
        if not self.is_audio_metadata_set:
            if msg.channels != self.num_channels:
                self.log.info("Audio number of channels is being changed " +\
                            f"{self.num_channels} -> {msg.channels}")
                self.num_channels = msg.channels
            
            if msg.sample_rate != self.sample_rate:
                self.log.info("Audio sample rate is being changed " +\
                            f"{self.sample_rate} -> {msg.sample_rate}")
                self.sample_rate = msg.sample_rate
            
            self.is_audio_metadata_set = True

        elif msg.channels != self.num_channels or\
                msg.sample_rate != self.sample_rate:
            self.log.error("Audio message metadata has suddenly changed. " +\
                           f"Node audio metadata is {self.num_channels} " +\
                           f"channels and {self.sample_rate} frequency. " +\
                           f"Msg audio metadata is {msg.num_channels} " +\
                           f"channels and {msg.sample_rate} frequency. ")

    def _time_to_byte_index(self, seconds_timestamp,
                            sample_byte_length,
                            num_channels, sample_rate):
        '''
        Maps a timestamp to corresponding index in a bytearray of audio data.
        This mapping requires the sample frequency, the number of Bytes per
        sample, and the number of channels are inherently necessary.
        '''
        return self._time_to_index(seconds_timestamp, sample_rate) *\
                   sample_byte_length * num_channels

    def _time_to_index(self, seconds_timestamp, sample_rate):
        '''
        Maps a timestamp to corresponding index in a float32[] of audio data.
        '''
        return int(seconds_timestamp * sample_rate)

def main():
    rclpy.init()
    vad = VoiceActivityDetector()
    rclpy.spin(vad)    
    vad.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
