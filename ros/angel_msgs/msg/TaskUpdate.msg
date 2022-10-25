#
# Represents the current status of the task being performed.
#

# Standard ROS message header
std_msgs/Header header

# Task name
string task_name

# Task description
string task_description

# Items required
TaskItem[] task_items

# List of steps for this task
# DEPRECATED: Users should instead query the QueryTaskGraph service.
string[] steps

# The index of the step most recently completed.
# A value of `-1` indicates that no step has been completed yet.
# TODO: The use of the word "current" is misleading: We don't predict a
#       step-in-action, but the step most recently completed. Maybe rename to
#       "recently_completed_step_id"
int8 current_step_id

# String of the most recently completed step.
# TODO: Maybe rename to "recently_completed_step"
string current_step
# Previous step is the step before the most recently completed
string previous_step

# Current activity
string current_activity
string next_activity

# Time remaining to move to next task (e.g. waiting for tea to steep)
# -1 means that this is not a time based task
int32 time_remaining_until_next_task

# Current confidence from the HMM that a recipe is complete.
# Only task_monitor_v2 will fill this in.
float32 task_complete_confidence
