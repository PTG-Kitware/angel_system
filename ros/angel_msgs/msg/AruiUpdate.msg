#
# Message encapsulating a monolithic update to send back to the AR-UI so it may
# be able to update it's user-facing rendering appropriately, and provide the
# appropriate signaling.
#
# This should broadly include:
#   * 3D object updates
#   * 3D activity updates
#   * task graph summary
#   * current task state
#   * New error messages
#   *
#
# NOTE: This message is initially in a state of "drafting" out what the ARUI
# needs, so some aspects are over-specified in this message and may be more
# appropriately broken out into a separate message file.
#
# meeting notes:
# provide goals / trying to accomplish, but not specifically what to do
# want to also know where all virtual things -- ARUI produces/manages these?
# probably want to also add spatial bounds to activity messages
#
#

std_msgs/Header header


#
# 3D Objects
#
# These objects represent incremental updates and removals to 3D objects that
# the system has observed. Objects have a UID to reflect entity persistence as
# determined by the system. If an object does not receive any updates, we
# expect that to be interpreted that they object still exists as the same as
# the last update.
#
# NOTE: All objects may be retrieved in batch by using the
# `angel_msgs/QueryAllObjects3d` service.
#

# Array of 3D objects that explicitly do not exist. UIDs are "reclaimed" in
# that new objects may re-use old UIDs.
angel_msgs/AruiObject3d[] object3d_remove

# Existing or new objects whose state has changed in some way.
angel_msgs/AruiObject3d[] object3d_update


#
# Activities
#
angel_msgs/ActivityDetection latest_activity


#
# User modeling
#
# This is probably more a reminder that user-modeling info will come in through
# this channel.
#
uint8 expertise_level


#
# Task Graph and State
#
# Message representing the current status of the task being performed.
#
# NOTE: The task graph may be queried by using the `QueryTaskGraph` service.
#
angel_msgs/TaskUpdate task_update


#
# User Communications
#
# Broad category of information that should be conveyed to a user.
# Different categories of communications have their own messages.
#

# Notifications / Directives
angel_msgs/AruiUserNotification[] notifications

# Field encapsulating interpreted user intents that we want confirmed by the user.
# These likely have some confidence less than
angel_msgs/InterpretedAudioUserIntent[] intents_for_confirmation
