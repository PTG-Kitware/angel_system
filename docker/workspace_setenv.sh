# This encapsulates environment setup that should be set for both build-time
# and run-time.

# Enable multicast flag on the loopback device (should always be present)
ifconfig lo multicast

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
