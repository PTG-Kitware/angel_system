# 2b. HARD CODED FOR DEMO 1 (required)

# false if there are no skills detected
bool populated false

# Dictionary of a list (key is casualty number),
#  each element of the list is a list of two items.
#  The two items are skill number (string) and confidence (0-1.0).
#  Confidence is the measure of certainty that the skill listed is really the skill.
#
# The following two vectors are parallel in association. `casualty_ids` defines
# the mapping keys and `skill_confidences` defines the mapping values.
#
uint32[] casualty_ids
BBNSkillConfidenceList[] skill_confidences
