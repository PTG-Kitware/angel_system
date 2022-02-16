#!/usr/bin/env bash
#
# Further setup docker environment for development.
# This script is expected to be sourced, not just run.
#

apt update
apt install -y \
  less \
  vim \
  python3-opencv \
  ros-foxy-rmw-cyclonedds-cpp \
  tmux
