#
# Message exporting the task (directed) graph structure.
#
# TODO: format likely depends on decisions regarding programatic structure, e.g.
#       there are parallel threads looking into PDDL loading / representation,
#       etc.
#

angel_msgs/TaskNode[] task_nodes

# Pairs of integers mapping parent-child node relationships in the directed
# graph.
# Integer values are indices into the above `task_nodes` array.
# This array should always have an even number of elements.
uint32[] node_edges
