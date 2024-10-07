#
# Dialogue Utterance with additional information about the environmental state
# and user model. 
#

# The header primarily encapsulates when this message was emitted.
# The time component of this may be utilized as an identifier for this user
# intent and utterance.
std_msgs/Header header

# Speech-to-text of the user utterance we have interpreted
string utterance_text

# Image frame at the time the user utterance was captured
string pov_frame

# Below are optional fields 

# Canonical user intent that has been interpreted. "Canonical" in this context
# is to mean that this string may be used as an identifier of this type of
# user intent. Should be in the range [0,1] where 1.0 means absolute confidence.
string intent
float64 intent_confidence_score

# Emotion classification. Should be in the range [0,1] where 1.0 means absolute confidence.
string emotion
float64 emotion_confidence_score
