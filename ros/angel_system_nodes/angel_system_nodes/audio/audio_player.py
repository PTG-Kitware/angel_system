import threading

import simpleaudio as sa

from rclpy.node import Node

from angel_msgs.msg import HeadsetAudioData
from angel_utils import make_default_main


class AudioPlayer(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("audio_topic", "HeadsetAudioData")

        self._audio_topic = (
            self.get_parameter("audio_topic").get_parameter_value().string_value
        )

        self.subscription = self.create_subscription(
            HeadsetAudioData, self._audio_topic, self.listener_callback, 1
        )

        self.audio_stream = bytearray()
        self.audio_stream_lock = threading.RLock()
        self.audio_player_lock = threading.RLock()
        self.t = threading.Thread()

        self.audio_stream_duration = 0.0
        self.playback_duration = 1.0

        self.prev_timestamp = None

    def listener_callback(self, msg):
        """
        Add the audio to the cumulative audio stream and spawns a new playback
        thread if the cumulative recording has reached more than a second and
        there is no ongoing playback.
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
                log.info(
                    "Warning! Out of order messages.\n"
                    + f"Prev: {self.prev_timestamp} \nCurr: {msg.header.stamp}"
                )
                return

            self.prev_timestamp = msg.header.stamp

            if self.audio_stream_duration >= self.playback_duration and not (
                self.t.is_alive()
            ):
                # Make a copy of the current data so we can play it back
                # while more data is accumulated
                audio_data = self.audio_stream

                # Remove the data that we just copied from the stream
                self.audio_stream = bytearray()
                self.audio_stream_duration = 0.0

                # Start the audio playback thread
                self.t = threading.Thread(
                    target=self.audio_playback_thread,
                    args=(
                        audio_data,
                        msg.channels,
                        msg.sample_rate,
                    ),
                )
                self.t.start()

    def audio_playback_thread(self, audio_data, num_channels, sample_rate):
        """
        Thread that plays back the received audio data with SimpleAudio.
        Waits for the audio playback to finish before exiting.
        """
        # Ensure only one playback is happening at a time
        with self.audio_player_lock:
            audio_player_object = sa.play_buffer(
                audio_data, num_channels, 4, sample_rate
            )  # bytes per sample

            audio_player_object.wait_done()


main = make_default_main(AudioPlayer)


if __name__ == "__main__":
    main()
