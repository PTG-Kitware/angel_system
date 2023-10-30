#
# Represents the current status of the task being performed.
#

# Standard ROS message header
std_msgs/Header header

# Task name
string task_name

# Task description
string task_description

# Timestamp of the latest input that has gone into this task update estimation.
builtin_interfaces/Time latest_sensor_input_time

# The index of the step currently in progress.
# A value of `-1` indicates that no step has been started yet.
int8 current_step_id

# String of the step currently in progress.
string current_step
# Previous step is the step worked on before the current step.
string previous_step

# Current confidence from the HMM that a recipe is complete.
# Only task_monitor_v2 will fill this in.
float32 task_complete_confidence

# Array of length n_steps (same length as `steps` above) that indicates which
# steps, by index association, are complete.
bool[] completed_steps

###############################################################################
#
# DEPRECATED FIELDS
#
# List of steps for this task
# --> Users should instead query the QueryTaskGraph service.
string[] steps

# Current activity classification prediction
string current_activity
string next_activity

# Items required
TaskItem[] task_items

# Time remaining to move to next task (e.g. waiting for tea to steep)
# -1 means that this is not a time based task
int32 time_remaining_until_next_task

# Array of relevant task step HMM step confidence scores. This includes the HMM
# background class as index 0.
# TODO: This should be its own separate message, implementation dependent
float32[] hmm_step_confidence
