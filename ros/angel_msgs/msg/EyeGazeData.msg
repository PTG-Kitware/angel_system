#
# Message for transporting the current eye gaze information for the user.
#

std_msgs/Header header

# Eye gaze position
geometry_msgs/Point gaze_origin

# Eye gaze direction
geometry_msgs/Vector3 gaze_direction

# Head movement info
geometry_msgs/Vector3 head_movement_direction
geometry_msgs/Vector3 head_velocity
