#
# Message that contains a snapshot of the patient joint poses
#

# Header frame_id should indicate the source these detections were predicted
# over.
std_msgs/Header header

# Timestamp of the source image these predictions pertain to.
builtin_interfaces/Time source_stamp

# List of joints
float64[] keypoints
