#
# Image housing ROS2 workspace dependencies and build.
#
# Assumptions:
# - ROS_DISTRO is defined with the name of a distribution installed under /opt/ros/
#
ARG ROS_VERSION="foxy"
FROM osrf/ros:${ROS_VERSION}-desktop AS base

SHELL ["/bin/bash", "-c"]

# Export language options to use UTF-8
ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Basic system package installations
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -y update \
 && apt-get -y install \
        bash-completion \
        gdb \
        gettext-base \
        htop \
        iproute2 \
        less \
        vim \
        parallel \
        tmux \
        tree \
        wget \
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
        libx11-6

# Activate some bash completion
RUN echo "source /etc/bash_completion" >> ~/.bashrc \
 && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

## Acquire and install NVIDIA CUDA 12.5
## * This is required for the ZED SDK installation.
RUN --mount=type=cache,target=/root/nvidia,sharing=locked \
    wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run \
      --no-verbose -c -O /root/nvidia/cuda_12.5.0_555.42.02_linux.run \
  && chmod +x /root/nvidia/cuda_12.5.0_555.42.02_linux.run \
  && /root/nvidia/cuda_12.5.0_555.42.02_linux.run --silent --toolkit
ENV PATH=/usr/local/cuda-12.5/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:${LD_LIBRARY_PATH}

## Install ZED SDK and dependencies.
## We are skipping the python installation here as the pacakge version has been
## manually deduced and added to our poetry configuration to allow for its
## depedencies to be resolved along side everything else we use.
RUN --mount=type=cache,target=/root/zed,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt install zstd \
 && wget https://download.stereolabs.com/zedsdk/4.1/cu121/ubuntu20 \
      --no-verbose -c -O /root/zed/ZED_SDK_Ubuntu20_cuda12.1_v4.1.2.zstd.run \
 && chmod +x /root/zed/ZED_SDK_Ubuntu20_cuda12.1_v4.1.2.zstd.run \
 && /root/zed/ZED_SDK_Ubuntu20_cuda12.1_v4.1.2.zstd.run \
    --quiet \
    --target /root/zed/install \
    -- \
    --silent \
    --skip_python

# For the demo-ui
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    (curl -sL https://deb.nodesource.com/setup_20.x | bash -) \
 && apt-get install -y nodejs \
 && node -v \
 && npm -v

# Bring in poetry for standard python package installation.
# Specifying a separate virtual environment that will be natively "activated".
# NOTE: I have some misgivings about making "~/.local/" a "venv", but we have
#       not observed adverse effects yet...
ENV VIRTUAL_ENV=/root/.local
ENV PATH=/root/.local/bin:${PATH}
ENV POETRY_VERSION=1.7.1
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    python3 -m venv --system-site-packages "${VIRTUAL_ENV}"\
 && pip3 install pip==23.3.1 \
 && (curl -sSL 'https://install.python-poetry.org' | python3 -) \
 && poetry config virtualenvs.create false

# Location to root our installation and workspace.
# This should be an absolute path.
ARG ANGEL_WORKSPACE_DIR
ENV ANGEL_WORKSPACE_DIR="${ANGEL_WORKSPACE_DIR}"
WORKDIR "${ANGEL_WORKSPACE_DIR}"

#
# Isolate package.xml files for rosdep dependency installation.
#
FROM base AS tmp_package_files
COPY ./ros "${ANGEL_WORKSPACE_DIR}"/ros
RUN mkdir "${ANGEL_WORKSPACE_DIR}"/tmp \
 && (find "${ANGEL_WORKSPACE_DIR}"/ros -type f -name 'package.xml' -print | xargs -n1 -I{} cp --parents {} "${ANGEL_WORKSPACE_DIR}"/tmp) \
 && rm -r "${ANGEL_WORKSPACE_DIR}"/ros \
 && mv "${ANGEL_WORKSPACE_DIR}"/tmp/"${ANGEL_WORKSPACE_DIR}"/ros "${ANGEL_WORKSPACE_DIR}"/ \
 && rm -r "${ANGEL_WORKSPACE_DIR}"/tmp

#
# Build our workspace
#
FROM base AS build

# Bring in just package files from source tree for rosdep to use.
COPY --from=tmp_package_files "${ANGEL_WORKSPACE_DIR}" "${ANGEL_WORKSPACE_DIR}"
COPY docker/workspace_build_rosdep_install.sh "${ANGEL_WORKSPACE_DIR}/"
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -y update \
 && "${ANGEL_WORKSPACE_DIR}/workspace_build_rosdep_install.sh"

# Install tracked python dependencies
COPY poetry.lock pyproject.toml docker/workspace_build_pydep_install.sh "${ANGEL_WORKSPACE_DIR}"/
COPY ./python-tpl "${ANGEL_WORKSPACE_DIR}/python-tpl"
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    --mount=type=cache,target=/root/.cache/pypoetry,sharing=locked \
    "${ANGEL_WORKSPACE_DIR}/workspace_build_pydep_install.sh"

# Create editable-installation hooks for the angel_system python package.
# * The container is intended to be environment-only, but it is highly
#   convenient to pre-establish the editable-installed package's metadata and
#   .pth linkage within the container. This allows importing the *real*
#   angel_system package from anywhere within the container post-mounting.
#   It is burdensom to require users of the environment container to "install"
#   the angel_system python package on every container instantiation.
# * This is intentionally separated from the above dep installation so as to
#   not trigger re-installation of all dependencies if there is only a change
#   in the library and not with the libraries dependencies.
RUN mkdir "${ANGEL_WORKSPACE_DIR}"/angel_system \
 && touch "${ANGEL_WORKSPACE_DIR}"/angel_system/__init__.py \
 && pip install --no-deps -e .

# Bring in tmux run configurations
# - add symlink to tmuxinator's expected "config home"
# - add symlink to default the tmux config for manual use
COPY ./tmux "${ANGEL_WORKSPACE_DIR}"/tmux
RUN ln -s "${ANGEL_WORKSPACE_DIR}"/tmux "${HOME}/.tmuxinator" \
 && ln -s "${ANGEL_WORKSPACE_DIR}"/tmux/tmux.conf "${HOME}"/.tmux.conf

COPY docker/cyclonedds_profile.xml.tmpl \
     docker/fastrtps_profile.xml.tmpl \
     docker/workspace_entrypoint.sh \
     docker/workspace_setenv.sh \
     docker/workspace_build.sh \
     docker/workspace_test.sh \
     "${ANGEL_WORKSPACE_DIR}"/

ENTRYPOINT ["./workspace_entrypoint.sh"]
CMD ["bash"]
