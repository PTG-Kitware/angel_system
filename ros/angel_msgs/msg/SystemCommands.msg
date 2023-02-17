#
# Fields that may be interpreted by various system components as commands to be
# followed.
#
# Each command must have some sort of defined "default" state which is
# interpreted as "no command issued" so that a consumer may understand whether
# a command needs to be enacted on or has just not been requested via the
# received message instance.
#

# Task Monitor commands
bool reset_current_task false
bool previous_step      false
bool next_step          false
