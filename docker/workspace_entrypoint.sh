#!/bin/bash
#
# PTG ANGEL container entrypoint.
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source installed ROS distro setup
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# If the variable is set, activate our workspace install setup
if [[ -n "$SETUP_WORKSPACE_INSTALL" ]]
then
  source "${ANGEL_WORKSPACE_DIR}/install/setup.bash"
fi

# Activate our workspace
source "${ANGEL_WORKSPACE_DIR}/workspace_setenv.sh"

exec "$@"
