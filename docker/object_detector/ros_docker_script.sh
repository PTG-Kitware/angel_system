#!/bin/bash
#
# Using second half of approach #3 described here:
#   http://wiki.ros.org/docker/Tutorials/GUI
#
set -e

# create a temporary xauth file for use with this container
tmp_xauth_file="$(mktemp "$PWD"/docker_xauth.XXXXXX)"
touch $tmp_xauth_file
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $tmp_xauth_file nmerge -
echo "Temp auth file: $tmp_xauth_file"

# Run container with appropriate passthroughs
docker_xauth_path="/tmp/.docker.xauth"
set +e
docker run --rm -it \
  -v /dev/shm:/dev/shm \
  -v "$PWD":"$PWD" \
  -w "$PWD" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$tmp_xauth_file":"$docker_xauth_path":rw \
  --env="DISPLAY" \
  --env="XAUTHORITY=${docker_xauth_path}" \
  --net=host \
  --gpus all \
  --privileged \
  "$@" \
  osrf/ros:foxy-desktop
docker_rc="$?"
set -e
echo "docker retcode: $docker_rc"

# Clean up temp auth file
echo "Cleaning up temp auth file"
rm "$tmp_xauth_file"

# Mirror docker exit code
exit "$docker_rc"
