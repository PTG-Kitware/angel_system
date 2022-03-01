#
# Image housing ROS2 workspace dependencies and build.
#
# Assumptions:
# - ROS_DISTRO is defined with the name of a distribution installed under /opt/ros/
#
ARG PTG_REGISTRY
ARG BASE_IMAGE
ARG PTG_TAG
FROM ${PTG_REGISTRY}/${BASE_IMAGE}:${PTG_TAG} AS base

SHELL ["/bin/bash", "-c"]

# Export language options to use UTF-8
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

#
# Isolate package.xml files for rosdep dependency installation.
#
FROM base AS tmp_package_files
COPY ./ros /angel_workspace/src
RUN find /angel_workspace/src -type f \! -name "package.xml" -print | xargs rm

#
# Build our workspace
#
FROM base AS build

# Bring in just package files from source tree for rosdep to use.
COPY --from=tmp_package_files /angel_workspace /angel_workspace

RUN apt-get -y update \
 && apt-get -y install \
        gettext-base \
        python3-pip \
        python3-venv \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        # for python simpleaudio
        libasound2-dev \
 && rosdep install -i --from-path /angel_workspace --rosdistro ${ROS_DISTRO} -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Bring in poetry for standard python package installation.
# NODE: Installing "dev" dependencies here because why not. Stop if the desire
#       becomes to create a minimal image.
ENV PATH=/root/.local/bin:${PATH}
ENV POETRY_VERSION=1.1.13
RUN pip3 install --user pip==22.0.3 \
 && (curl -sSL 'https://install.python-poetry.org' | python3 -) \
 && poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml /angel_workspace/
RUN cd /angel_workspace \
    # TODO: Re-enable use of hashes once editable/git deps are removed. \
    #       - Remove `--without-hashes` option. \
    # * Poetry extraneously finds matplotlib requiring setuptools-scm.
    #   theoretically being "fixed" in matplotlib 3.6
 && poetry export --without-hashes --dev -f requirements.txt | \
      pip3 install --no-deps --user -r /dev/stdin \
 && pip uninstall -y setuptools-scm

# Load in any model files the workspace will require at runtime.
# TODO

COPY ./ros /angel_workspace/src
WORKDIR /angel_workspace

COPY docker/workspace_build.sh /angel_workspace/
RUN /angel_workspace/workspace_build.sh

COPY docker/cyclonedds_profile.xml.tmpl \
     docker/workspace_entrypoint.sh \
     docker/workspace_setenv.sh \
     /angel_workspace/

ENTRYPOINT ["/angel_workspace/workspace_entrypoint.sh"]
CMD ["bash"]
