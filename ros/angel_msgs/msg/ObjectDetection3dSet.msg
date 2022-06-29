#
# A collection of 3D object detections obtained by projecting the 2D detections
# onto the world scene.
#
# Box [left,right,top,bottom] origin is coordinate is the upper-left of the
# source image.
#

# Header frame_id should indicate the source these detections were predicted
# over.
std_msgs/Header header

# Timestamp of the source image these predictions pertain to.
builtin_interfaces/Time source_stamp

# Vector of object labels with length = num_objects
string[] object_labels

# Number of objects contained in this set.
int64 num_objects

# Vector of detection axis aligned bounding box bounds point coordinates.
# Each vector here should be of congruent length, where value `i` corresponds
# to the 3D coordinate for detection `i`.
geometry_msgs/Point[] left
geometry_msgs/Point[] right
geometry_msgs/Point[] top
geometry_msgs/Point[] bottom
