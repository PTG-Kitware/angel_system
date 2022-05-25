#
# Message for transporting the current eye gaze information for the user.
# The info in this message is provided by the MRTK EyeGazeProvider API.
#

std_msgs/Header header

# Eye gaze position
geometry_msgs/Point gaze_origin

# Eye gaze direction
geometry_msgs/Vector3 gaze_direction

# Head movement info
# Head info is included along with the gaze info by the EyeGazeProvider API,
# so we are passing it along here as well.
geometry_msgs/Vector3 head_movement_direction
geometry_msgs/Vector3 head_velocity

# Object hit info
bool is_object_hit
geometry_msgs/Point hit_object_position
