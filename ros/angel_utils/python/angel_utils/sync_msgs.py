"""
Contains functions to synchronize ROS messages.
"""
from typing import Dict
from typing import List

import numpy as np

from builtin_interfaces.msg import Time
from angel_msgs.msg import HandJointPosesUpdate


def get_frame_synced_hand_poses(
    frame_stamps: List[Time],
    hand_poses: Dict[str, List[np.array]],
    hand_pose_stamps: Dict[str, List[Time]],
    slop_ns: float
):
    """
    Synchronize the given set of frame stamps with the given set of hand poses.
    Returns a set of hand poses with the closest hand pose for each frame.
    If no hand pose is available for the current frame, a zero filled vector
    is used in its place.
    """
    lhand_pose_set = []
    rhand_pose_set = []

    hands = hand_pose_stamps.keys()

    # TODO add a lock here?
    for f in frame_stamps:
        # Find the closest matching hand pose for this frame timestamp
        frm_nsec = f.sec * 10e9 + f.nanosec

        # Search through left hand pose stamps
        for h in hands:
            closest_hand_pose = None
            closest_hand_idx = None
            for idx, stamp in enumerate(hand_pose_stamps[h]):
                h_nsec = stamp.sec * 10e9 + stamp.nanosec

                diff = abs(frm_nsec - h_nsec)
                if closest_hand_pose is None or (diff < closest_hand_pose):
                    closest_hand_pose = diff
                    closest_hand_idx = idx

            #print(closest_hand_pose, closest_hand_idx)

            if closest_hand_pose is None:
                if h == "lhand":
                    lhand_pose_set.append(np.zeros((63,)))
                else:
                    rhand_pose_set.append(np.zeros((63,)))
            elif closest_hand_pose < slop_ns:
                if h == "lhand":
                    lhand_pose_set.append(hand_poses[h][closest_hand_idx])
                else:
                    rhand_pose_set.append(hand_poses[h][closest_hand_idx])
            else:
                # Not within tolerance, so use 0 filled vector
                if h == "lhand":
                    lhand_pose_set.append(np.zeros((63,)))
                else:
                    rhand_pose_set.append(np.zeros((63,)))

    return lhand_pose_set, rhand_pose_set
