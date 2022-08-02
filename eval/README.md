# Enter the ANGEL system workspace
`$ ./angel-workplace-shell.sh`

`$ source install/setup.sh`

`$ ./workplace_build.sh`

# Create an activity detection ROS bag from an annotated ROS bag
`$ tmuxinator start -p tmux/record_ros_bag_activity_only.yml`

In the "play_ros_bag" tab, run
`$ ros2 bag play <annotated_ros_bag>`

Once the bag has finished running, run CTL+C in the "ros_bag_record" tab

Move the new activity only ros bag to a location accessible outside the Docker workspace

# Extract the ROS bag
`$ ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:="<activity_only_ros_bag>" -p extract_images:="False" -p extract_audio:="False" -p extract_eye_gaze_data:="False" -p extract_head_pose_data:="False" -p extract_hand_pose_data:="False" -p extract_spatial_map_data:="False"`

# Run eval
`$ python evaluate.py <optional flags>`
