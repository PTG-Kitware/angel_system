#
# Fields that may be interpreted by various system components as commands to be
# followed.
#
# Each command must have some sort of defined "default" state which is
# interpreted as "no command issued" so that a consumer may understand whether
# a command needs to be enacted on or has just not been requested via the
# received message instance.
#

##############################
# Task Monitor commands

# Index of the task to which the commands here pertain.
int8 task_index
# Reset the referenced task's state and start over from the beginning.
# TODO: This is currently unused.
bool reset_task
# Revert our step tracking for the referenced task to the previous step.
bool previous_step
# Progress our step tracking for the referenced task to the next step.
bool next_step
