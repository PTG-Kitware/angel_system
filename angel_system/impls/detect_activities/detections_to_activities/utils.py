from typing import Dict

import numpy as np



def obj_det2d_set_to_feature(
    label_vec,
    left,
    right,
    top,
    bottom,
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
            A[right hand], D[right hand, obj1] ... D[right hand, objN],
            A[left hand], D[left hand, obj1] ... D[left hand, objN],
            D[right hand, left hand],
            A[obj1] ... A[objN]
        ]
        """
        feature_vec = []
        act = np.zeros(num_act)
        bboxes = [[0, 0, 0, 0] for i in range(num_act)]
        
        for i in range(num_dets):
            label = label_vec[i]
            conf = label_confidences[i]
            bbox = [top[i], left[i], bottom[i], right[i]]

            ind = label_to_ind[label_vec[i]]

            if conf > act[ind]:
                act[ind] = conf
                bboxes[ind] = bbox

        def dist_from_hand(hand_idx, hand_center):
            hand_dist = [(0, 0) for i in range(num_act)]
            for i in range(num_act):
                if i == hand_idx:
                    continue
                
                obj_bbox = bboxes[i]
                obj_center = [
                    (obj_bbox[3] - obj_bbox[1])/2,
                    (obj_bbox[0] - obj_bbox[2])/2
                ]

                dist_x = hand_center[0] - obj_center[0]
                dist_y = hand_center[1] - obj_center[1]

                hand_dist[i] = (dist_x, dist_y)

            return hand_dist

        # Find the right hand
        right_hand_idx = label_to_ind["hand (right)"]
        right_hand_bbox = bboxes[right_hand_idx]
        right_hand_conf = act[right_hand_idx]

        right_hand_center = [
            (right_hand_bbox[3] - right_hand_bbox[1])/2,
            (right_hand_bbox[0] - right_hand_bbox[2])/2
        ]

        # Compute distances to the right hand
        if right_hand_conf != 0:
            right_hand_dist = dist_from_hand(right_hand_idx, right_hand_center)
        else:
            right_hand_dist = [(0, 0) for i in range(num_act)]
        
        # Find the left hand
        left_hand_idx = label_to_ind["hand (left)"]
        left_hand_bbox = bboxes[left_hand_idx]
        left_hand_conf = act[left_hand_idx]

        left_hand_center = [
            (left_hand_bbox[3] - left_hand_bbox[1])/2,
            (left_hand_bbox[0] - left_hand_bbox[2])/2
        ]

        # Compute distances to the left hand
        if left_hand_conf != 0:
            left_hand_dist = dist_from_hand(left_hand_idx, left_hand_center)
        else:
            left_hand_dist = [(0, 0) for i in range(num_act)]
        
        # Distance between hands
        hands_dist_x = right_hand_center[0] - left_hand_center[0]
        hands_dist_y = right_hand_center[1] - left_hand_center[1]

        hands_dist = (hands_dist_x, hands_dist_y)

        # Remove hands from lists
        del right_hand_dist[right_hand_idx]
        del right_hand_dist[left_hand_idx-1]

        del left_hand_dist[right_hand_idx]
        del left_hand_dist[left_hand_idx-1]

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
