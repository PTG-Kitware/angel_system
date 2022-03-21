#
# An activity detection on a set of frames.
#

# Header frame_id should indicate the source these detections were predicted
# over.
std_msgs/Header header

# Timestamps of the first image and the last image these predictions pertain to.
builtin_interfaces/Time source_stamp_start_frame
builtin_interfaces/Time source_stamp_end_frame

# Prediction classification label ordered from most confident to least.
string[] label_vec
