#!/bin/bash

if [[ -z "$1" ]]
then
  echo "ERROR: Missing path to bag file as first positional argument."
  exit 1
fi
BAG_PATH="$1"
shift

ros2 run angel_utils bag_extractor.py --ros-args \
  -p bag_path:="$BAG_PATH" \
  -p pv_image_frame_id:=PVFramesBGR \
  "$@"
