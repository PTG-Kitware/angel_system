#
# Represents the current status of the task being performed.
#

# Standard ROS message header
std_msgs/Header header

# Task name
string task_name

# List of steps for this task
string[] steps

# Current/previous steps
string current_step
string previous_step

# Current activity
string current_activity
