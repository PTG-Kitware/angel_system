Enter the ANGEL system workspace
################################
`$ ./angel-workplace-shell.sh`

`$ source install/setup.sh`

`$ ./workplace_build.sh`

Create an activity detection ROS bag from an annotated ROS bag
##############################################################
`$ tmuxinator start -p tmux/record_ros_bag_activity_only.yml`

In the "play_ros_bag" tab, run
`$ ros2 bag play <annotated_ros_bag>`

Once the bag has finished running, run CTL+C in the "ros_bag_record" tab

Move the new activity only ros bag to a location accessible outside the Docker workspace

Extract the ROS bags
####################

Extract images from the annotated ros bag
-----------------------------------------
Turning off all flags is optional, but they are not needed for eval.
`$ ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:="<annotated_ros_bag>" -p extract_audio:="False" -p extract_eye_gaze_data:="False" -p extract_head_pose_data:="False" -p extract_hand_pose_data:="False" -p extract_spatial_map_data:="False" -p extract_annotation_event_data:="False"`

Extract activity detections from the new ros bag
------------------------------------------------
`$ ros2 run angel_utils bag_extractor.py --ros-args -p bag_path:="<activity_only_ros_bag>" -p extract_images:="False" -p extract_audio:="False" -p extract_eye_gaze_data:="False" -p extract_head_pose_data:="False" -p extract_hand_pose_data:="False" -p extract_spatial_map_data:="False" -p extract_annotation_event_data:="False"`

Run eval
########
`$ ptg_eval_activity <optional flags>`

E.g.:
```bash
$ poetry run ptg_eval --labels activity_detector_annotation_labels.xlsx \
                      --activity_gt labels_test_v1.4.feather \
                      --extracted_activity_detections sample_dets.json \
                      --output_dir ./eval_output/
```
