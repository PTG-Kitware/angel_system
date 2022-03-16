#!/bin/bash
#
# Standard workspace build.
#
# This script is expecting to be run inside the container workspace root.
# Standard advice is to perform builds before sourcing install setup scripts.
#
# Additional parameters are passed on to `colcon build`.
#
# For additional debugging of build products, try adding:
#   --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
colcon build --continue-on-error --merge-install "$@"
