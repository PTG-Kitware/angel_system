#!/bin/bash
set -e
source "$ANGEL_WORKSPACE_DIR"/install/setup.bash
colcon test --merge-install --packages-ignore detectron2 "$@"
