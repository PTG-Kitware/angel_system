#!/bin/bash
set -e

# Source installed ROS distro setup
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.sh"

# Activate our workspace
source /angel_workspace/workspace_setenv.sh

exec "$@"
