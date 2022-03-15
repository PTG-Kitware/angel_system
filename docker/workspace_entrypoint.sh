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

# If CYCLONE_DDS_INTERFACE is defined to a value, then template
if [[ -n "$CYCLONE_DDS_INTERFACE" ]]
then
  envsubst <"${SCRIPT_DIR}/cyclonedds_profile.xml.tmpl" >"${SCRIPT_DIR}/cyclonedds_profile.xml"
  export CYCLONEDDS_URI=file://${SCRIPT_DIR}/cyclonedds_profile.xml
fi

# TODO: there is probably a better way to do this
# Install the angel system package
pip install .

exec "$@"
