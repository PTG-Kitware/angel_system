#
# Message containing audio data captured from the HL2 microphones.
#
# NOTE: There does not seem to be an audio message type in the default
# ROS message packages. This message is based on the audio messages
# defined in https://github.com/ros-drivers/audio_common
#

# Header with time stamp
std_msgs/Header header

# Audio meta data
int32 channels

# Sampling rate [Hz]
int32 sample_rate

# Duration of sample (s)
float32 sample_duration

# Audio data
float32[] data
