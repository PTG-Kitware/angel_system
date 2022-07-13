#
# An activity detection on a set of frames.
#

# Header frame_id should indicate the source these detections were predicted
# over.
std_msgs/Header header

# Timestamps of the first image and the last image these predictions pertain to.
builtin_interfaces/Time source_stamp_start_frame
builtin_interfaces/Time source_stamp_end_frame

# Class labels and prediction confidences.
# conf_vec[i] corresponds to the prediction confidence for the class at
# label_vec[i].
string[] label_vec
float32[] conf_vec
