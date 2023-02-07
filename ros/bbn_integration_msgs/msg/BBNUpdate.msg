# BBN's demo yaml includes a "header" field that is in conflict with ROS's
# normal "header" field. This should be translated in the integration node that
# is interfacing with YAML+ZMQ. In this ROS specific message, we'll separate
# into "header" (ROS header) and "bbn_header", the latter incorporating the
# BBN- specific information.

# ROS2 standard header info.
# - frame_id is not defined yet.
#std_msgs/Header header

string message

# TODO: Translate the remainder of the header content that is appropriate here.
