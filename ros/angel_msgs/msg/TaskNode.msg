#
# A single task node
#

string uid
string name

# NOTE: The following are speculative. Awaiting system-side task data-structure
# details to better inform the appropriate details to put here.

# Related objects
string[] object_uid

# Action instruction
string action_name
float32 action_time  # may be zero if not applicable.
