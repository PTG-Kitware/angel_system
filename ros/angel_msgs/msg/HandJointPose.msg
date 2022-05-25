#
# Message for transmitting a single hand joint pose.
#

# Which joint this pose represents
# If we experience network congestion issues, we could change this to the int
# in the MRTK enumeration
string joint

# Pose (position and rotation)
geometry_msgs/Pose pose
