import pyaudio
import rclpy
from rclpy.node import Node
import requests
import simpleaudio as sa
import threading
import time
import wave

from angel_msgs.msg import HeadsetAudioData

FORMAT = pyaudio.paInt16


class ASR(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("audio_topic", "HeadsetAudioData")
        self.declare_parameter("asr_server_url", "http://communication.cs.columbia.edu:8058/home")
        self.declare_parameter("tmp_wav_filename", "/tmp/asr.wav")
        

        self._audio_topic = self.get_parameter("audio_topic").get_parameter_value().string_value
        self._asr_server_url = self.get_parameter("asr_server_url").get_parameter_value().string_value
        self._tmp_wav_filename = self.get_parameter("tmp_wav_filename").get_parameter_value().string_value

        self.subscription = self.create_subscription(
            HeadsetAudioData,
            self._audio_topic,
            self.listener_callback,
            1)

        self.audio_stream = bytearray()
        self.audio_stream_lock = threading.RLock()
        self.asr_server_lock = threading.RLock()
        self.t = threading.Thread()

        self.audio_stream_duration = 0.0
        self.segmentation_duration = 30.0

        self.prev_timestamp = None

    def listener_callback(self, msg):
        """
        Add the audio to the cumulative audio stream and spawns a new playback
        thread if the cumulative recording has reached more than 30 seconds.
        """
        log = self.get_logger()
        
        with self.audio_stream_lock:
            # Check that this message comes temporally after the previous message
            message_order_valid = False
            if self.prev_timestamp is None:
                message_order_valid = True
            else:
                if msg.header.stamp.sec > self.prev_timestamp.sec:
                    message_order_valid = True
                elif msg.header.stamp.sec == self.prev_timestamp.sec:
                    # Seconds are same, so check nanoseconds
                    if msg.header.stamp.nanosec > self.prev_timestamp.nanosec:
                        message_order_valid = True

            if message_order_valid:
                self.audio_stream.extend(msg.data)
                self.audio_stream_duration += msg.sample_duration
            else:
                log.info("Warning! Out of order messages.\n"
                         + f"Prev: {self.prev_timestamp} \nCurr: {msg.header.stamp}")
                return

            self.prev_timestamp = msg.header.stamp
            if self.audio_stream_duration >= self.segmentation_duration and not(self.t.is_alive()):
                # Make a copy of the current data so we can play it back
                # while more data is accumulated.
                audio_data = self.audio_stream

                # Remove the data that we just copied from the stream.
                self.audio_stream = bytearray()
                self.audio_stream_duration = 0.0

                # Start the ASR server request thread.
                self.t = threading.Thread(target=self.asr_server_request_thread,
                    args=(audio_data, msg.channels, msg.sample_rate, self._tmp_wav_filename))
                self.t.start()


    def asr_server_request_thread(self, audio_data, num_channels, 
        sample_rate, wav_output_filename):
        """

        """
        # Ensure only one ASR is happening at a time(?)
        with self.asr_server_lock:

            wf = wave.open(wav_output_filename, 'wb')
            wf.setnchannels(num_channels)
            # TODO (derekahmed) We should figure out how this value was derived
            # and make this a constant accordingly.
            wf.setsampwidth(4) # Derived from audio_player.py
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
            wf.close()

            with open(wav_output_filename, 'rb') as f:
                response = requests.post(self._asr_server_url, files={'audio_data': f})
            
            print(response.text)
            # TODO(@derekahmed): Delete the saved audio file.
            return response.text


def main():
    rclpy.init()

    asr = ASR()
    
    rclpy.spin(asr)
    
    asr.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()