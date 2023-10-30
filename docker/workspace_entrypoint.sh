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

# Report on ROS localhost only mode.
if [[ "${ROS_LOCALHOST_ONLY}" -ne 0 ]]
then
  >&2 echo "[INFO] Setting ROS to localhost-only communication."
else
  >&2 echo "[INFO] ROS host communication unconstrained."
fi

# Configure RMW-specific configurations depending on what is set to be used.
if [[ "${RMW_IMPLEMENTATION}" = "rmw_fastrtps_cpp" ]]
then
  >&2 echo "[INFO] RMW set to FastRTPS, generating config XML."
  envsubst <"${SCRIPT_DIR}/fastrtps_profile.xml.tmpl" >"${SCRIPT_DIR}/fastrtps_profile.xml"
  export FASTRTPS_DEFAULT_PROFILES_FILE="${SCRIPT_DIR}/fastrtps_profile.xml"
  export RMW_FASTRTPS_USE_QOS_FROM_XML=1
elif [[ "${RMW_IMPLEMENTATION}" = "rmw_cyclonedds_cpp" ]]
then
  >&2 echo "[INFO] RMW set to Cyclone DDS, generating config XML."
  if [[ -z "$CYCLONE_DDS_INTERFACE" ]]
  then
    >&2 echo "[ERROR] Cyclone DDS RMW specified, but not interface specified in var CYCLONE_DDS_INTERFACE."
    exit 1
  fi
  envsubst <"${SCRIPT_DIR}/cyclonedds_profile.xml.tmpl" >"${SCRIPT_DIR}/cyclonedds_profile.xml"
  export CYCLONEDDS_URI="file://${SCRIPT_DIR}/cyclonedds_profile.xml"
else
  >&2 echo "[INFO] No explicit RMW set, using stock defaults."
fi

exec "$@"
