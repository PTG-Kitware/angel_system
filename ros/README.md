ROS2 workspace root.

# Build workspace
Start a docker container with `ros_docker_script.sh` and then:
```bash
$ ./ros_docker_prereqs.sh
$ source ros_docker_setenv.sh
$ rosdep install -i --from-path . --rosdistro foxy -y
$ colcon build
```

# Running from a built workspace
Start a docker container with `ros_docker_script.sh` and then:
```bash
$ ./ros_docker_prereqs.sh
$ source ros_docker_setenv.sh
$ source install/setup.sh
$ ros2 run ...
```

# `rosdep` Lessons
References to the lists that rosdep uses to resolve names:
    /etc/ros/rosdep/sources.list.d/20-default.list
