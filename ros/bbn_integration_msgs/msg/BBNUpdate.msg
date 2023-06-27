# BBN's demo yaml includes a "header" field that is in conflict with ROS's
# normal "header" field. This should be translated in the integration node that
# is interfacing with YAML+ZMQ. In this ROS specific message, we'll separate
# into "header" (ROS header) and "bbn_header", the latter incorporating the
# BBN- specific information.

# ROS2 standard header info.
# - frame_id is not defined yet.
std_msgs/Header header

# 1. BASIC HEADER STUFF (required)
BBNHeader bbn_header

# 2a. CASUALTY COUNT / WORK STATIONS
# HARD CODED FOR DEMO 1 (required)
BBNCasualties casualties

# 2b. HARD CODED FOR DEMO 1 (required)
BBNSkillsOpenPerCasualty skills_open_per_casualty

# 2c. WHICH OF THESE SKILLS ARE DONE (optional/required?)
BBNSkillsDonePerCasualty skills_done_per_casualty

# 2d. WHAT IS THE CURRENT SKILL STEP ACTIVITY? (required)
BBNCurrentUserActions current_user_actions

# 3. MEASURE OF PROGRESS TOWARDS NEXT STEP (optional)
BBNNextStepProgress next_step_progress

# 4. DO WE BELIEVE THERE ARE ANY ERRORS?  (optional)
BBNCurrentErrors current_errors

# 5. TA1 needs to consult with me about this.  (future)
BBNCurrentUserState current_user_state
