System configuration files for integrating with BBN.

Naming Convention
=================
We have traditionally had two configuration files for every integration with
BBN.
Each configuration "pair" should be identical except for how sensor data is
input into the ROS2 system.

The configurations for use *with* BBN are prefixed with ``BBN-`` such that the
file name might look like ``BBN-<task_name>.yml``. These configurations should
pull sensor data from the ``ros2 run angel_system_nodes redis_ros_bridge`` node
and have a "BBN Interface" section that uses the the BBN integration nodes to
output to their systems.

The configurations for use on the Kitware side, i.e. with the HoloLens2, are
prefixed with ``Kitware-`` such that the file name might look like
``Kitware-<task_name>.yml``.
