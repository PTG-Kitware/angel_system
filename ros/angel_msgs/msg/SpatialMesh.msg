#
# Message for transporting spatial mapping mesh updates received from
# the HoloLens.
#

int32 mesh_id

# Signals if this mesh should be removed from the world map
bool removal

shape_msgs/Mesh mesh
