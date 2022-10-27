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

# This script will be installed into the directory that houses the `ros` source
# directory, i.e. the location that `colcon build` should be run from.
cd "$SCRIPT_DIR"

# When running this manually, it should be in the use-case of manually
# re-building due to debugging, developing, or similar use-case. In that
# use-case, it is likely useful to attempt to requery for rosdep installs
# updates. During a container image build, we want to skip this as this will
# have already occurred / is cached in an earlier layer. The container build is
# NOT a tty shell, so that is our conditional hinge.
if tty -s
then
  apt-get -y update
  rosdep install -i --from-path "${ANGEL_WORKSPACE_DIR}" --rosdistro "${ROS_DISTRO}" -y
fi

# Activate the base ROS context and build our local workspace.
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.bash"
colcon build --continue-on-error --merge-install "$@"

# Build web resources for Demo/Engineering UI
pushd "${ANGEL_WORKSPACE_DIR}"/src/angel_utils/demo_ui
npm install
popd
