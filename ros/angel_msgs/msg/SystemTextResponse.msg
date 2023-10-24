#
# Message indicating a generated textual answer for a provided query 
# utterance received from a user.
#

# The header primarily encapsulates when this message was emitted.
# The time component of this may be utilized as an identifier for this user
# intent and utterance.
std_msgs/Header header

# User utterance received as input to a ROS node.
string utterance_text

# Generated system response text for received user utterance input.
string response