#
# Message for updating hand joint information from the HoloLens hand tracking
# provider. There are 26 tracked joints for each hand. Position and rotation
# are provided for each joint.
#

std_msgs/Header header

# Which hand this update is for ('left' or 'right')
string hand

# List of joints
angel_msgs/HandJointPose[] joints
