#!/bin/bash
# Workspace build component -- rosdep installation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# This script will be installed into the directory that houses the
# ROS2 workspace source directory.
cd "$SCRIPT_DIR"

# Run update if we have no cache yet
if [[ ! -d ${HOME}/.ros/rosdep ]]
then
  rosdep update
fi

rosdep install -i --from-path "${ANGEL_WORKSPACE_DIR}" --rosdistro "${ROS_DISTRO}" -y
