#
# Message for transporting the world matrices and projection matrices provided
# with each frame from the HoloLens.
#

std_msgs/Header header

# Camera-to-world projection matrix. This is a flattened 4x4 matrix in
# row-major order (e.g. [00,01,02,03,10,...]). The world coordinate system
# described by this transform is left-handed (Unity).
float32[] world_matrix

# Camera matrix (flattened 4x4 format).
# See: https://docs.unity3d.com/ScriptReference/Camera-projectionMatrix.html
float32[] projection_matrix
