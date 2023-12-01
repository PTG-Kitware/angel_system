#!/bin/bash
# Workspace build component -- ROs workspace build via colcon
#
# For additional debugging of build products, try adding:
#   --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script will be installed into the directory that houses the `ros` source
# directory, i.e. the location that `colcon build` should be run from.
cd "$SCRIPT_DIR"

# Activate the base ROS context and build our local workspace.
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
colcon build --base-paths "${SCRIPT_DIR}/ros" --continue-on-error --merge-install "$@"
