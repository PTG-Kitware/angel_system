# STATE OF A STEP


# State identifiers
# -- Task step is complete.
uint8 STATE_DONE = 0
# -- Task step is implied complete.
uint8 STATE_IMPLIED = 1
# -- Task step is currently active.
uint8 STATE_CURRENT = 2
# -- Task step is unobserved at this time.
# ** Default state
uint8 STATE_UNOBSERVED = 3


# Index of the task step
uint32 number

# Name of the task step
string name

# Step state identifier, see below for constants expected.
uint8 state 3  # default: STATE_UNOBSERVED (see above)

# Confidence of the step state determination, in the [0, 1] range.
float64 confidence
