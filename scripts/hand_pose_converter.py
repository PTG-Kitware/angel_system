import argparse
import json
import os

import numpy as np

from angel_system.utils.matrix_conversion import (
    convert_1d_4x4_to_2d_matrix,
    project_3d_pos_to_2d_image,
)


# NOTE: These values were extracted from the projection matrix provided by the
# Unity main camera with the Windows MR plugin (now deprecated).
# For some unknown reason, the values provided by the Open XR plugin are
# different and do not provide the correct results. In the future, we should
# figure out why they are different or extract the focal length values from the
# MediaFrameReader which provides the frames, instead of the Unity main camera.
FOCAL_LENGTH_X = 1.6304
FOCAL_LENGTH_Y = 2.5084

PROJECTION_MATRIX = [
    [FOCAL_LENGTH_X, 0.0, 0.0, 0.0],
    [0.0, FOCAL_LENGTH_Y, 0.0, 0.0],
    [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
    [0.0, 0.0, -1.0, 0.0]
]


class HandPoseConverter():
    """
    Converts hand poses in 3d world space coordinates to 2d image space coordinates.
    """

    def __init__(self, hand_pose_file, head_pose_file, hand_pose_out_file):
        self.hand_pose_file = hand_pose_file
        self.head_pose_file = head_pose_file
        self.hand_pose_out_file = hand_pose_out_file

        # Parse the bag
        self.convert_poses()

    def convert_poses(self) -> None:
        """
        Main function to convert from 3D to 2D coordinates. For each hand pose
        in the hand pose json file, we find the closest matching head pose,
        extract the world and projection matrices, and convert the 3D joint
        position to 2D camera space.

        At the end, an output json is saved with the new data.
        """
        # Open the hand poses json file
        with open(self.hand_pose_file) as f:
            hand_pose_data = json.load(f)

        # Open the head poses json file
        with open(self.head_pose_file) as f:
            head_pose_data = json.load(f)

        tolerance = (5 / 60.0) * 1e9 # ns
        hand_poses_3d = []

        hand_msg_index = 0
        for hand_pose in hand_pose_data:
            if hand_msg_index % 1000 == 0:
                print(f"Parsing message {hand_msg_index}")
            hand_msg_index += 1

            # Find the closest headset pose message for this hand msg
            hand_time_ns = hand_pose['time_sec'] * 1e9 + hand_pose['time_nanosec']
            min_diff = None
            min_idx = None
            for idx, h in enumerate(head_pose_data):
                head_time_ns = h['time_sec'] * 1e9 + h['time_nanosec']

                diff = abs(head_time_ns - hand_time_ns)
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            if min_diff < tolerance:
                matching_head_pose = head_pose_data[idx]

            if matching_head_pose is None:
                print("no match")
                hand_pose_3d_dict = {
                    "time_sec": hand_pose['time_sec'],
                    "time_nanosec": hand_pose['time_nanosec'],
                    "hand": hand_pose['hand'],
                    "joint_poses": None
                }
                hand_poses_3d.append(hand_pose_3d_dict)
                continue

            projection_matrix = PROJECTION_MATRIX
            camera_to_world_matrix = convert_1d_4x4_to_2d_matrix(
                matching_head_pose['world_matrix']
            )
            world_to_camera_matrix = np.linalg.inv(camera_to_world_matrix)

            # List of joints in this message
            joints_3d = []
            for j in hand_pose['joint_poses']:
                # Get the hand pose
                joint_position_3d = np.array(
                    [j['position'][0],
                     j['position'][1],
                     j['position'][2],
                     1]
                )
                coords = project_3d_pos_to_2d_image(
                    joint_position_3d,
                    world_to_camera_matrix,
                    projection_matrix,
                )

                joints_3d.append({'joint': j["joint"],  "position": coords})

            # Append this pose to our list of poses
            hand_pose_3d_dict = {
                "time_sec": hand_pose['time_sec'],
                "time_nanosec": hand_pose['time_nanosec'],
                "hand": hand_pose['hand'],
                "joint_poses": joints_3d
            }
            hand_poses_3d.append(hand_pose_3d_dict)

        # Write to the output file
        with open(self.hand_pose_out_file, mode="w", encoding="utf-8") as f:
            json.dump(hand_poses_3d, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir")
    args = parser.parse_args()

    root_dir = args.root_dir
    subs = next(os.walk(root_dir))[1]
    print(subs)

    for folder in subs:
        print(f"Processing folder: {folder}")

        hand_pose_file = root_dir + "/" + folder + "/" + "hand_pose_data.json"
        head_pose_file = root_dir + "/" + folder + "/" + "head_pose_data.json"
        hand_pose_out_file = folder + "_hand_pose_2d_data.json"

        try:
            converter = HandPoseConverter(hand_pose_file, head_pose_file, hand_pose_out_file)
        except FileNotFoundError:
            hand_pose_file = root_dir + "/" + folder + "/" + "hand_pose_data.txt"
            head_pose_file = root_dir + "/" + folder + "/" + "head_pose_data.txt"
            converter = HandPoseConverter(hand_pose_file, head_pose_file, hand_pose_out_file)
