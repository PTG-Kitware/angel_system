##############################################################################
# Base system packages and depedencies
#
FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04 AS ptg-gpu
CMD nvidia-smi

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

# Add ``object_detector`` user
RUN useradd -mr -s /bin/bash object_detector \
 # sudo permission for modifying permissions at runtime (see entrypoint.sh).
 # TODO: Remove this eventually since its defeating the point of having a user.
 && echo "object_detector ALL=(ALL:ALL) NOPASSWD:ALL" >>/etc/sudoers

USER object_detector
WORKDIR /home/object_detector

COPY --chown=object_detector:object_detector \
     docker/object_detector/entrypoint.sh /home/object_detector/

# Copy in ros/python subscriber and build
COPY --chown=object_detector:object_detector \
     ros/python /home/object_detector/ros/python

RUN cd /home/object_detector/ros/python && \
    colcon build --merge-install

# Copy in ros profiles
COPY --chown=object_detector:object_detector \
     ros/cyclonedds_profile.xml /home/object_detector/ros/

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/home/object_detector/ros/cyclonedds_profile.xml

#ENTRYPOINT ["/home/object_detector/entrypoint.sh"]
