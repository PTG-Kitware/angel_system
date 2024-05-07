#
# Fields that may be interpreted by various system components as commands to be
# followed.
#
# Each command must have some sort of defined "default" state which is
# interpreted as "no command issued" so that a consumer may understand whether
# a command needs to be enacted on or has just not been requested via the
# received message instance.
#


# =====================
# Task Monitor commands
# =====================

# Index of GSP tasks being concurrently handled to indicate step progression/regression for.
int8 task_index

# If the current task progress should be reset.
bool reset_current_task

# If the current task progress should be regressed a step,
bool previous_step

# If the current task progress should be progressed a step.
bool next_step


# =============
# Task Changing
# =============

# Enumeration of sorts for the tasks that may be switched to.
#
# This is an enumeration here in order to be a source of coordination between
# nodes that support this functionality as just simply passing a string can
# easily be subject to spelling errors or misalignment between nodes.
#
uint32 TASK_NO_CHANGE   =   0
uint32 TASK_COOKING     =   1   # Cooking is "meta" atm because it is multi-task under the hood.
uint32 TASK_MEDICAL_M2  =   2
uint32 TASK_MEDICAL_M3  =   3
uint32 TASK_MEDICAL_R18 =   4

# The task that is being changed to.
uint32 change_task
