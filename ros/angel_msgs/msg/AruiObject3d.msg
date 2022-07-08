#
# Object state update message to be sent to the AR-UI.
#
# This message is intended to represent a single object in 3D space
#
# Currently assumed a sub-component of the `AruiUpdate` message.
#

# Unique ID of this object.
# This value is likely a hash-like string, not something that has
# human-semantic meaning.
# This ID should be unique among all currently "active" 3D objects, e.g. no
# other 3D object that has not already been removed should share this ID.
string uid

# Latest update time.
builtin_interfaces/Time stamp

# Semantic label of this object.
string label

# 3D location of this object as a series of bounding-box coordinates.
# Following REP-103, the bbox "size" vector axis follow the right-hand rule.
# See "Axis Orientation" here: https://www.ros.org/reps/rep-0103.html#id21
angel_msgs/VisionBoundingBox3d bbox
