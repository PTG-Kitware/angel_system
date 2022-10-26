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
 && apt-get -y dist-upgrade \
 && apt-get -y install \
        bash-completion \
        gdb \
        htop \
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
