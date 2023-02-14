# 2c. WHICH OF THESE SKILLS ARE DONE (optional/required?)
# THIS WILL CHANGE AS TREATMENT PROGRESSES

# false if there are no skills detected
bool populated false

# Dictionary of a list (key is casualty number),
#  each element of the list is a list of two items.
#  The two items are skill number (string) and belief the skill is complete (0-1.0).
#  This is your systems belief that we've seen the end of activity on this skill.
#  It is not a measure of progress, or of correctness.  Just that we're done with the skill.
#  If you are certain the skill is completed, your belief it is complete is 1.0.
#  If you are certain the skill is being worked on your belief it is complete is 0.0.
#  If you have no measure of this, your belief will always stay at 0.5
#
# The following two vectors are parallel in association. `casualty_ids` defines
# the mapping keys and `skill_confidences` defines the mapping values.
#
uint32[] casualty_ids
BBNSkillConfidenceList[] skill_confidences
