#
# Message for transporting the world matrices and projection matrices provided
# with each frame from the HoloLens.
#

std_msgs/Header header

float32[] world_matrix
float32[] projection_matrix
