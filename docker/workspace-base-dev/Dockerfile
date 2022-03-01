##############################################################################
# Base system packages and depedencies for development convenience.
#
# This should include packages convenient for development, but not specifically
# runtime requirements for the ROS2 system.
#

FROM osrf/ros:foxy-desktop AS base

SHELL ["/bin/bash", "-c"]

# System Package dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
 && apt-get -y install \
        bash-completion \
        less \
        vim \
        parallel \
        tmux \
    # Clean up apt resources.
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Activate some bash completion
RUN echo "source /etc/bash_completion" >> ~/.bashrc \
 && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

# Install opengl
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
