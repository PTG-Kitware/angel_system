#!/bin/bash
#
# All-in-one workspace build.
#
# This script is expecting to be run inside the container workspace root by a
# user in a TTY session. This will run all of the standard build steps using
# the same component scripts as in the Dockerfile build.
# Standard advice is to perform builds before sourcing install setup scripts.
#
# Additional parameters are passed on to the underlying ROS workspace
# `colcon build` invocation.
#
# For additional debugging of build products, try adding:
#   --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
#
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# We don't want to auto apt update every single time, so only do it once on
# first invocation of this script. The /tmp directory is not expected to be
# mounted to the host filesystem, so it should be "fresh" every instantiation
# of a container.
ANGEL_APT_UPDATED="/tmp/ANGEL_APT_UPDATED"
if ! [[ -f "${ANGEL_APT_UPDATED}" ]]
then
  apt-get -y update
  touch "$ANGEL_APT_UPDATED"
fi

# Always run this as file edits through normal development may change
# behavior, so we want to react to that.
"$SCRIPT_DIR"/workspace_build_rosdep_install.sh

"$SCRIPT_DIR"/workspace_build_pydep_install.sh

"$SCRIPT_DIR"/workspace_build_ros.sh "$@"

"$SCRIPT_DIR"/workspace_build_npm_install.sh
