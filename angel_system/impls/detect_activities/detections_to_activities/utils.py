from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def obj_det2d_set_to_feature(
    label_vec,
    xs,
    ys,
    ws,
    hs,
    label_confidences,
    descriptors,
    obj_obj_contact_state,
    obj_obj_contact_conf,
    obj_hand_contact_state,
    obj_hand_contact_conf,
    label_to_ind: Dict[str, int],
    version: int = 1,
):
    """Convert ObjectDetection2dSet fields into a feature vector.

    :param label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.

    :param version:
        Version of the feature conversion approach.
    """
    num_act = len(label_to_ind)
    num_dets = len(label_vec)

    if version == 1:
        """
        Feature vector that encodes the activation feature of each class

        [A[obj1] ... A[objN]]
        """
        feature_vec = np.zeros(num_act)

        for i in range(num_dets):
            ind = label_to_ind[label_vec[i]]
            feature_vec[ind] = np.maximum(feature_vec[ind], label_confidences[i])
    elif version == 2:
        """
        Feature vector that encodes the distance of each object from each hand
        along with the activation features

        [
            A[right hand], D[right hand, obj1]x, D[right hand, obj1]y, ... , D[right hand, objN]y,
            A[left hand], D[left hand, obj1]x, D[left hand, obj1]y, ... , D[left hand, objN]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            A[obj1] ... A[objN]
        ]
        """
        feature_vec = []
        act = np.zeros(num_act)
        bboxes = [[0, 0, 0, 0] for i in range(num_act)]

        for i in range(num_dets):
            label = label_vec[i]
            conf = label_confidences[i]
            bbox = [xs[i], ys[i], ws[i], hs[i]]  # xywh

            ind = label_to_ind[label_vec[i]]

            if conf > act[ind]:
                act[ind] = conf
                bboxes[ind] = bbox

        def dist_from_hand(hand_idx, hand_center):
            hand_dist = [(0, 0) for i in range(num_act)]
            for i in range(num_act):
                if i == hand_idx:
                    continue

                x, y, w, h = bboxes[i]
                obj_center = [x + (w / 2), y + (h / 2)]

                if hand_center != [0.0, 0.0]:
                    dist_x = hand_center[0] - obj_center[0]
                    dist_y = hand_center[1] - obj_center[1]
                else:
                    dist_x = 0
                    dist_y = 0

                hand_dist[i] = (dist_x, dist_y)

            return hand_dist

        def find_hand(hand_str):
            hand_idx = label_to_ind[hand_str]
            hand_bbox = bboxes[hand_idx]
            hand_conf = act[hand_idx]

            x, y, w, h = hand_bbox
            hand_center = [x + (w / 2), y + (h / 2)]

            # Compute distances to the right hand
            if hand_conf != 0:
                hand_dist = dist_from_hand(hand_idx, hand_center)
            else:
                hand_dist = [(0, 0) for i in range(num_act)]

            return hand_idx, hand_bbox, hand_conf, hand_center, hand_dist

        # Find the right hand
        (
            right_hand_idx,
            right_hand_bbox,
            right_hand_conf,
            right_hand_center,
            right_hand_dist,
        ) = find_hand("hand (right)")

        # Find the left hand
        (
            left_hand_idx,
            left_hand_bbox,
            left_hand_conf,
            left_hand_center,
            left_hand_dist,
        ) = find_hand("hand (left)")

        # Distance between hands
        if right_hand_center != [0.0, 0.0] and left_hand_center != [0.0, 0.0]:
            hands_dist_x = right_hand_center[0] - left_hand_center[0]
            hands_dist_y = right_hand_center[1] - left_hand_center[1]
        else:
            hands_dist_x = 0
            hands_dist_y = 0
        hands_dist = (hands_dist_x, hands_dist_y)

        # Remove hands from lists
        del right_hand_dist[right_hand_idx]
        del right_hand_dist[left_hand_idx - 1]

        del left_hand_dist[right_hand_idx]
        del left_hand_dist[left_hand_idx - 1]

        act = np.delete(act, [right_hand_idx, left_hand_idx])

        # Create feature vec
        feature_vec = []
        feature_vec.append(right_hand_conf)
        for rhd in right_hand_dist:
            feature_vec.append(rhd[0])
            feature_vec.append(rhd[1])
        feature_vec.append(left_hand_conf)
        for lhd in left_hand_dist:
            feature_vec.append(lhd[0])
            feature_vec.append(lhd[1])
        feature_vec.append(hands_dist[0])
        feature_vec.append(hands_dist[1])
        for a in act:
            feature_vec.append(a)

        feature_vec = np.array(feature_vec, dtype=np.float64)

    else:
        raise NotImplementedError(f"Unhandled version '{version}'")

    return feature_vec
