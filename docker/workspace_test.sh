#!/bin/bash
#
# Run CMake-based testing across packages in our workspace.
#
# This could be run after a workspace build to check that tests from component
# packages pass.
#
# NOTE: Regular package testing is under development and it is not yet expected
# that this will regularly pass.
#
set -e
source "$ANGEL_WORKSPACE_DIR"/install/setup.bash
colcon test --merge-install --packages-ignore detectron2 "$@"
