#
# Message communicating an interpreted User Audio from an audio-based user
# utterance.
#

# The header primarily encapsulates when this message was emitted.
# The time component of this may be utilized as an identifier for this user
# intent and utterance.
std_msgs/Header header

# Speech-to-text of the user utterance we have interpreted
string utterance_text

# Canonical user emotion that has been interpreted. "Canonical" in this context
# is to mean that this string may be used as an identifier of this type of
# emotion
string user_emotion

# Confidence we think this intent was targetted at the ANGEL system.
# Should be in the range [0,1] where 1.0 means absolute confidence the target is the ANGEL
# system and 0.0 is absolute confidence the target IS NOT the ANGEL system.
float64 confidence