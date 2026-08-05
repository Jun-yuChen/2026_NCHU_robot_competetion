FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# Basic dependencies
RUN apt-get update && apt-get install -y \
    sudo \
    locales \
    tzdata \
    curl \
    wget \
    git \
    vim \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    usbutils \
    udev \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    libssl-dev \
    libgtk-3-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    > /etc/apt/sources.list.d/ros2.list

# ROS 2 Humble + RViz + RealSense
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-ros-gz \
    ros-humble-control-msgs \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# PyTorch with CUDA 12.8
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Create non-root user (Moved UP so we can configure their environment)
ARG USERNAME=ros
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    groupadd -f -g 20 dialout && \
    usermod -aG dialout,sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Set up the ROS 2 workspace directory and permissions
WORKDIR /ros2_ws
RUN chown -R ${USERNAME}:${USERNAME} /ros2_ws

# Source ROS 2 automatically for BOTH the root and the new 'ros' user
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /root/.bashrc

RUN echo "source /opt/ros/humble/setup.bash" >> /home/${USERNAME}/.bashrc && \
    echo "source /ros2_ws/install/setup.bash 2>/dev/null || true" >> /home/${USERNAME}/.bashrc

# Switch to the non-root user
USER ${USERNAME}

# Default command to run at startup (Must be at the very bottom)
CMD ["bash"]
