==========
Evaluation
==========
This document assumes familiarity with :ref:`Activity Annotation and Ground
Truth`.

Enter the ANGEL system workspace
################################
``$ ./angel-workplace-shell.sh``

``$ source install/setup.sh``

``$ ./workplace_build.sh``

Create an activity detection ROS bag from an annotated ROS bag
##############################################################
``$ tmuxinator start -p tmux/record_ros_bag_activity_only.yml``

In the "play_ros_bag" tab, run
``$ ros2 bag play <annotated_ros_bag>``

Once the bag has finished running, run CTL+C in the "ros_bag_record" tab

Move the new activity only ros bag to a location accessible outside the Docker workspace

Extract the ROS bags
####################

Extract images from the annotated ros bag
-----------------------------------------
Turning off all flags is optional, but they are not needed for eval:

.. prompt:: bash

    ros2 run angel_utils bag_extractor.py --ros-args \
        -p bag_path:="<annotated_ros_bag>" \
        -p extract_audio:="False" \
        -p extract_images:="True" \
        -p extract_eye_gaze_data:="False" \
        -p extract_head_pose_data:="False" \
        -p extract_hand_pose_data:="False" \
        -p extract_spatial_map_data:="False" \
        -p extract_annotation_event_data:="False" \
        -p extract_activity_detection_data:="False" \
        -p extract_depth_images:="False" \
        -p extract_depth_head_pose_data:="False"

Extract activity detections from the new ros bag
------------------------------------------------
.. prompt:: bash

    ros2 run angel_utils bag_extractor.py --ros-args \
        -p bag_path:="<activity_only_ros_bag>" \
        -p extract_audio:="False" \
        -p extract_images:="False" \
        -p extract_eye_gaze_data:="False" \
        -p extract_head_pose_data:="False" \
        -p extract_hand_pose_data:="False" \
        -p extract_spatial_map_data:="False" \
        -p extract_annotation_event_data:="False" \
        -p extract_activity_detection_data:="True" \
        -p extract_depth_images:="False" \
        -p extract_depth_head_pose_data:="False"

Run eval
########
``$ ptg_eval_activity <optional flags>``

The evaluation tool should be run with a ``--time_window`` option that is
non-trivially smaller than predicted activity time windows.
This is due to evaluation needing to associate any ground-truth or predicted
activities to a time-window that is *fully contained* within that activity's
temporal bounds.

E.g.::

    $ poetry run ptg_eval_activity \
            --activity_gt labels_test_v1.4.csv \
            --extracted_activity_detections sample_dets.json \
            --time_window 0.1 \
            --uncertainty_pad 0.05 \
            --output_dir ./eval_output/
