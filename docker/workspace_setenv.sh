# This encapsulates environment setup that should be set for both build-time
# and run-time.

# Set Cyclone DDS for middleware
# * We were finding that this middleware was slow for local node connections
#   that fastRTPS is more appropriate for which should be using shared memory.
# export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
