#
# Message communicating a single notification for the ARUI to present to the
# user.
#
# A notification may be handed in certain ways based on the combination of both
# the notification context and category.
#

###############################################################################
# Notification category constants.
# Basically the priority in which the notification should be treated.
# https://www.mysafetysign.com/danger-caution-warning-safety-sign-headers
uint8 N_CAT_DANGER=0
uint8 N_CAT_WARNING=1
uint8 N_CAT_CAUTION=2
uint8 N_CAT_NOTICE=3

# Notification category. See constants above.
uint8 category

###############################################################################
# Notification context constants.
# Different notifications can have different contexts from which they are
# emitted that the UI will want to handle differently.

# There is some error the user performed in the task
uint8 N_CONTEXT_TASK_ERROR=0
# There is something in the environment this notification pertains to. Likely spatial in nature.
uint8 N_CONTEXT_ENV_ATTENTION=1
# This notification is in regards to the user modeling (e.g. user frustration).
uint8 N_CONTEXT_USER_MODELING=2

# Notification context. See constants above.
uint8 context

###############################################################################
# Textual details to go with this notification.

# A short message about this notification.
string title

# Potentially longer description of this notification.
string description

###############################################################################
# Spatial associations for this notification.

# 0 or more  3D objects this notification is associated with, by UID.
string[] associated_object_uids

# 0 or more spatial polygons this notification is associated with.
# Expected to mark surfaces.
geometry_msgs/Polygon[] associated_polys
