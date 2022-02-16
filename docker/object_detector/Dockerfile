##############################################################################
# Base system packages and depedencies
#
FROM osrf/ros:foxy-desktop AS ros
MAINTAINER josh.anderson@kitware.com

SHELL ["/bin/bash", "-c"]

# System Package dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update \
 && apt-get -y install \
        curl \
        less \
        sudo \
        vim \
        parallel \
        python3-dev \
        python3-pip \
        python3-tk \
        ros-foxy-rmw-cyclonedds-cpp \
        ros-foxy-demo-nodes-cpp \
        ros-foxy-demo-nodes-py \
 # Clean up apt resources.
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        smqtk-detection \
        opencv-python \
        torch \
        torchvision

# Export language options to use UTF-8, desired by Click
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Export language options to use UTF-8, desired by Click
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN mkdir /object_detector

# Copy in ros folder and build
COPY ros/ /object_detector/ros

RUN cd /object_detector/ros/python && \
    colcon build --merge-install

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/object_detector/ros/cyclonedds_profile.xml
