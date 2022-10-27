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

# Basic system package installations
RUN apt-get -y update \
 && apt-get -y install \
        gettext-base \
        python3-pip \
        python3-venv \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-test-msgs \
        ros-${ROS_DISTRO}-pybind11-vendor \
        # for demo-ui
        ros-${ROS_DISTRO}-rosbridge-server \
        tmuxinator \
        # for python simpleaudio
        libasound2-dev \
        # for opengl use with trimesh
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libxext6 \
        libx11-6 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# For the demo-ui
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - 
RUN apt-get install -y nodejs && node -v && npm -v

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Bring in poetry for standard python package installation.
# NODE: Installing "dev" dependencies here because why not. Stop if the desire
#       becomes to create a minimal image.
ENV PATH=/root/.local/bin:${PATH}
ENV POETRY_VERSION=1.1.13
RUN pip3 install --user pip==22.0.3 \
 && (curl -sSL 'https://install.python-poetry.org' | python3 -) \
 && poetry config virtualenvs.create false

# Location to root our installation and workspace.
# This should be an absolute path.
ARG ANGEL_WORKSPACE_DIR
ENV ANGEL_WORKSPACE_DIR="${ANGEL_WORKSPACE_DIR}"

#
# Isolate package.xml files for rosdep dependency installation.
#
FROM base AS tmp_package_files
COPY ./ros "${ANGEL_WORKSPACE_DIR}"/src
RUN mkdir "${ANGEL_WORKSPACE_DIR}"/tmp \
 && (find "${ANGEL_WORKSPACE_DIR}"/src -type f -name 'package.xml' -print | xargs -n1 -I{} cp --parents {} "${ANGEL_WORKSPACE_DIR}"/tmp) \
 && rm -r "${ANGEL_WORKSPACE_DIR}"/src \
 && mv "${ANGEL_WORKSPACE_DIR}"/tmp/"${ANGEL_WORKSPACE_DIR}"/src "${ANGEL_WORKSPACE_DIR}"/ \
 && rm -r "${ANGEL_WORKSPACE_DIR}"/tmp

#
# Build our workspace
#
FROM base AS build

# Bring in just package files from source tree for rosdep to use.
COPY --from=tmp_package_files "${ANGEL_WORKSPACE_DIR}" "${ANGEL_WORKSPACE_DIR}"
RUN apt-get -y update \
 && rosdep install -i --from-path "${ANGEL_WORKSPACE_DIR}" --rosdistro ${ROS_DISTRO} -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install tracked python dependencies
COPY poetry.lock pyproject.toml "${ANGEL_WORKSPACE_DIR}"/
RUN cd "${ANGEL_WORKSPACE_DIR}" \
    # * Poetry extraneously finds matplotlib requiring setuptools-scm.
    #   theoretically being "fixed" in matplotlib 3.6
 && poetry export --dev -f requirements.txt | \
      pip3 install --no-deps --user -r /dev/stdin \
 && pip uninstall -y setuptools-scm

# Build the angel_system python package
COPY ./angel_system "${ANGEL_WORKSPACE_DIR}"/angel_system
RUN cd "${ANGEL_WORKSPACE_DIR}" \
 && pip install --no-deps --user -e .

# Load in any model files the workspace will require at runtime.
# TODO

COPY ./ros "${ANGEL_WORKSPACE_DIR}/src"
COPY docker/workspace_build.sh "${ANGEL_WORKSPACE_DIR}"/
RUN "${ANGEL_WORKSPACE_DIR}/workspace_build.sh"

RUN cd "${ANGEL_WORKSPACE_DIR}"/src/angel_utils/demo_ui/ \
    && npm install

# Bring in tmux run configurations
# - add symlink to tmuxinator's expected "config home"
# - add symlink to default the tmux config for manual use
COPY ./tmux "${ANGEL_WORKSPACE_DIR}"/tmux
RUN ln -s "${ANGEL_WORKSPACE_DIR}"/tmux "${HOME}/.tmuxinator" \
 && ln -s "${ANGEL_WORKSPACE_DIR}"/tmux/tmux.conf "${HOME}"/.tmux.conf

COPY docker/cyclonedds_profile.xml.tmpl \
     docker/workspace_entrypoint.sh \
     docker/workspace_setenv.sh \
     "${ANGEL_WORKSPACE_DIR}"/

WORKDIR "${ANGEL_WORKSPACE_DIR}"
ENTRYPOINT ["./workspace_entrypoint.sh"]
CMD ["bash"]
