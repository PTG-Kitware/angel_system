import argparse
import json
import os

import numpy as np


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
    [0.0, 0.0, -1.0, 0.0],
]


def project_3d_pos_to_2d_image(position, inverse_world_mat, projection_mat):
    """
    Projects the 3d position vector into 2d image space using the given world
    to camera and camera projection matrices.

    The image coordinates returned are in the range [-1, 1]. Values outside of
    this range are clipped.
    """
    # Convert from world space to camera space
    x = np.matmul(inverse_world_mat, position)
    # Convert from camera space to image space
    image = np.matmul(projection_mat, x)
    # print("image coords", image)

    # Normalize
    image_scaled = image / -image[3]  # perspective divide
    # print("image scaled coords", image_scaled)

    image_scaled_x = image_scaled[0]
    image_scaled_y = image_scaled[1]

    # Clipping - Limit to -1 and 1
    clipped = 0
    if image_scaled_x > 1:
        image_scaled_x = 1
        clipped = 1
    elif image_scaled_x < -1:
        image_scaled_x = -1
        clipped = 1
    if image_scaled_y > 1:
        image_scaled_y = 1
        clipped = 1
    elif image_scaled_y < -1:
        image_scaled_y = -1
        clipped = 1

    width = 1280
    height = 720

    half_width = width / 2
    half_height = height / 2

    w = image_scaled[3]

    # Convert to screen coordinates
    projected_point_x = width - ((image_scaled[0] * width) / (2 * w) + half_width)
    projected_point_y = (image_scaled[1] * height) / (2 * w) + half_height
    # print("pixel coords", projected_point_x, projected_point_y)

    return (
        [projected_point_x, projected_point_y, 1.0],
        [image_scaled_x, image_scaled_y, 1.0],
        clipped,
    )


def convert_1d_4x4_to_2d_matrix(matrix_1d):
    matrix_2d = [[], [], [], []]
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            matrix_2d[row].append(matrix_1d[idx])

    return matrix_2d


class HandPoseConverter:
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

        tolerance = (5 / 60.0) * 1e9  # ns
        hand_poses_3d = []

        hand_msg_index = 0
        for hand_pose in hand_pose_data:
            if hand_msg_index % 100 == 0:
                print(f"Parsing message {hand_msg_index}")
            hand_msg_index += 1

            # Find the closest headset pose message for this hand msg
            hand_time_ns = hand_pose["time_sec"] * 1e9 + hand_pose["time_nanosec"]
            min_diff = None
            min_idx = None
            for idx, h in enumerate(head_pose_data):
                head_time_ns = h["time_sec"] * 1e9 + h["time_nanosec"]

                diff = abs(head_time_ns - hand_time_ns)
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    min_idx = idx

            if min_diff < tolerance:
                matching_head_pose = head_pose_data[min_idx]

            if matching_head_pose is None:
                print("no match")
                hand_pose_3d_dict = {
                    "time_sec": hand_pose["time_sec"],
                    "time_nanosec": hand_pose["time_nanosec"],
                    "hand": hand_pose["hand"],
                    "joint_poses": None,
                }
                hand_poses_3d.append(hand_pose_3d_dict)
                continue

            projection_matrix = PROJECTION_MATRIX
            camera_to_world_matrix = convert_1d_4x4_to_2d_matrix(
                matching_head_pose["world_matrix"]
            )
            # print(camera_to_world_matrix)
            world_to_camera_matrix = np.linalg.inv(camera_to_world_matrix)

            # List of joints in this message
            joints_3d = []
            for j in hand_pose["joint_poses"]:
                # Get the hand pose
                joint_position_3d = np.array(
                    [j["position"][0], j["position"][1], j["position"][2], 1]
                )
                proj, coords, clipped = project_3d_pos_to_2d_image(
                    joint_position_3d,
                    world_to_camera_matrix,
                    projection_matrix,
                )

                joints_3d.append(
                    {
                        "joint": j["joint"],
                        "position": coords,
                        "projected": proj,
                        "clipped": clipped,
                    }
                )

            # Append this pose to our list of poses
            hand_pose_3d_dict = {
                "time_sec": hand_pose["time_sec"],
                "time_nanosec": hand_pose["time_nanosec"],
                "hand": hand_pose["hand"],
                "joint_poses": joints_3d,
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

        hand_pose_file = (
            root_dir + "/" + folder + "/_extracted/" + "hand_pose_data.json"
        )
        head_pose_file = (
            root_dir + "/" + folder + "/_extracted/" + "head_pose_data.json"
        )
        hand_pose_out_file = (
            root_dir + "/" + folder + "/_extracted/_hand_pose_2d_data.json"
        )

        try:
            converter = HandPoseConverter(
                hand_pose_file, head_pose_file, hand_pose_out_file
            )
        except FileNotFoundError:
            hand_pose_file = root_dir + "/" + folder + "/" + "hand_pose_data.txt"
            head_pose_file = root_dir + "/" + folder + "/" + "head_pose_data.txt"
            converter = HandPoseConverter(
                hand_pose_file, head_pose_file, hand_pose_out_file
            )
