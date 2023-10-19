#
# A collection of 2D object detections predicted on a source 2D image.
#
# This message does not use ObjectDetection2D as a means of vectorizing
# components for efficient packing reasons.
#
# Box [left,right,top,bottom] origin is coordinate is the upper-left of the
# source image.
#
# Example: Getting the max-conf labels for each detection (Python)
# >>> np.asarray(msg.label_vec)[
# ...     np.asarray(msg.label_confidences)
# ...     .reshape(msg.num_detections, len(msg.label_vec))
# ...     .argmax(axis=1)
# ... ]
#

# Header frame_id should indicate the source these detections were predicted
# over.
std_msgs/Header header

# Timestamp of the source image these predictions pertain to.
builtin_interfaces/Time source_stamp

# Prediction classification labels.
# Each detection in this set should have a confidence prediction for each of
# these labels, in the [0,1] range.
# If a detector does not yield a confidence for a label, then a 0 should be
# used for labels it did not predict.
string[] label_vec

# Number of detections contained in this set.
int64 num_detections

# Vector of detection axis aligned bounding box bounds coordinates.
# Each vector here should be of congruent length, where value `i` corresponds
# to the coordinate for detection `i`.
float32[] left
float32[] right
float32[] top
float32[] bottom

# Predicted label confidences for detections.
# This is a flattened 2D row-major matrix with shape:
#   [num_detections, len(label_vec)]
float64[] label_confidences

###############################################################################
# DEPRECATED
#

# Vector length of each descriptor.
int32 descriptor_dim

# Flattened descriptor vector. The length of this list should be equal to
# num_detections * descriptor_dims.
float64[] descriptors

# Optional fields for Berkeley object detector.
# Each of these vectors is expected to be of length num_detections.
bool[] obj_obj_contact_state
float32[] obj_obj_contact_conf
bool[] obj_hand_contact_state
float32[] obj_hand_contact_conf
