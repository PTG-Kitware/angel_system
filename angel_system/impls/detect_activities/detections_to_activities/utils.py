from typing import Dict
from typing import Tuple

import kwimage

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def tlbr_to_xywh(
    top: npt.ArrayLike,
    left: npt.ArrayLike,
    bottom: npt.ArrayLike,
    right: npt.ArrayLike,
) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Convert array-likes of vectorized TLBR (top-left-bottom-right) box
    coordinates into XYWH (x, y, width, height) format (similarly vectorized)

    :param top: Array-like of top box coordinate values.
    :param left: Array-like of left box coordinate values.
    :param bottom: Array-like of bottom box coordinate values.
    :param right: Array-like of right box coordinate values.

    :return:
    """
    assert (
        len(top) == len(left) == len(bottom) == len(right)
    ), "No all input array-likes were the same length."
    xs = np.asarray(left)
    ys = np.asarray(top)
    ws = np.asarray(right) - xs
    hs = np.asarray(bottom) - ys
    return xs, ys, ws, hs


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
    if version == 1:
        """
        Feature vector that encodes the activation feature of each class

        Len: 42

        [A[obj1] ... A[objN]]
        """
        feature_vec = obj_det2d_set_to_feature_by_method(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            label_to_ind,
            use_activation=True,
        )

    elif version == 2:
        """
        Feature vector that encodes the distance of each object from each hand,
        and the activation features

        Len: 204

        [
            A[right hand],
            D[right hand, obj1]x, D[right hand, obj1]y, ... , D[right hand, objN]y,
            A[left hand],
            D[left hand, obj1]x, D[left hand, obj1]y, ... , D[left hand, objN]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            A[obj1] ... A[objN]
        ]
        """
        feature_vec = obj_det2d_set_to_feature_by_method(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            label_to_ind,
            use_activation=True,
            use_hand_dist=True,
        )

    elif version == 3:
        """
        Feature vector that encodes the distance of each object to the center of the frame,
        the intersection of each object to the hands,
        and the activation features

        Len: 207

        [
            A[right hand],
            D[right hand, center]x, D[right hand, center]y,
            A[left hand],
            D[left hand, center]x, D[left hand, center]y,
            I[right hand, left hand]
            A[obj1],
            I[right hand, obj1],
            I[left hand, obj1]
            D[obj1, center]x, D[obj1, center]y
        ]
        """
        feature_vec = obj_det2d_set_to_feature_by_method(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            label_to_ind,
            use_activation=True,
            use_center_dist=True,
            use_intersection=True,
        )

    elif version == 5:
        """
        Feature vector that encodes the distance of each object from each hand,
        the intersection of each object to the hands,
        and the activation features

        Len: 1 + ((N-2)*2) + 1 + ((N-2)*2) + 2 + 1 + (3 * (N-2)), where N is the number of object classes

        [
            A[right hand],
            D[right hand, obj1]x, D[right hand, obj1]y, ... , D[right hand, objN]y,
            A[left hand],
            D[left hand, obj1]x, D[left hand, obj1]y, ... , D[left hand, objN]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            I[right hand, left hand]
            A[obj1] I[right hand, obj1] I[left hand, obj1], ... , I[left hand, objN]
        ]
        """
        feature_vec = obj_det2d_set_to_feature_by_method(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            label_to_ind,
            use_activation=True,
            use_hand_dist=True,
            use_intersection=True,
        )

    else:
        raise NotImplementedError(f"Unhandled version '{version}'")

    # print(f"feat {feature_vec}")
    # print(len(feature_vec))
    return feature_vec


def obj_det2d_set_to_feature_by_method(
    label_vec,
    xs,
    ys,
    ws,
    hs,
    label_confidences,
    label_to_ind: Dict[str, int],
    use_activation=False,
    use_hand_dist=False,
    use_center_dist=False,
    use_intersection=False,
):
    #########################
    # Default values
    #########################
    default_dist = (0, 0)  # (1280 * 2, 720 * 2)
    default_center_dist = (0, 0)  # (1280, 720)
    default_bbox = [0, 0, 0, 0]  # [0, 0, 1280, 720]
    default_center = ([[0]], [[0]])  # kwimage.Boxes([default_bbox], "xywh").center

    #########################
    # Data
    #########################
    num_act = len(label_to_ind)
    num_dets = len(label_vec)

    act = np.zeros(num_act)
    bboxes = [default_bbox for i in range(num_act)]

    for i in range(num_dets):
        label = label_vec[i]
        conf = label_confidences[i]
        bbox = [xs[i], ys[i], ws[i], hs[i]]  # xywh

        ind = label_to_ind[label_vec[i]]

        if conf > act[ind]:
            act[ind] = conf
            bboxes[ind] = bbox

    #########################
    # util functions
    #########################
    def find_hand(hand_str):
        hand_idx = label_to_ind[hand_str]
        hand_bbox = bboxes[hand_idx]
        hand_conf = act[hand_idx]

        hand_bbox = kwimage.Boxes([bboxes[hand_idx]], "xywh")

        return hand_idx, hand_bbox, hand_conf, hand_bbox.center

    def dist_to_center(center, obj_center):
        center_dist = [
            obj_center[0][0][0] - center[0][0][0],
            obj_center[1][0][0] - center[1][0][0],
        ]
        return center_dist

    def dist_from_hand(hand_idx, hand_center):
        hand_dist = [default_dist for i in range(num_act)]
        for i in range(num_act):
            obj_bbox = kwimage.Boxes([bboxes[i]], "xywh")
            obj_center = obj_bbox.center

            if obj_center != default_center:
                hand_dist[i] = dist_to_center(obj_center, hand_center)

        return hand_dist

    #########################
    # Hands
    #########################
    # Find the right hand
    (right_hand_idx, right_hand_bbox, right_hand_conf, right_hand_center) = find_hand(
        "hand (right)"
    )

    # Find the left hand
    (left_hand_idx, left_hand_bbox, left_hand_conf, left_hand_center) = find_hand(
        "hand (left)"
    )

    #########################
    # Distances
    #########################
    if use_hand_dist:
        # Compute distances to the right hand
        right_hand_dist = (
            dist_from_hand(right_hand_idx, right_hand_center)
            if right_hand_conf != 0
            else [default_dist for i in range(num_act)]
        )

        # Compute distances to the left hand
        left_hand_dist = (
            dist_from_hand(left_hand_idx, left_hand_center)
            if left_hand_conf != 0
            else [default_dist for i in range(num_act)]
        )

    else:
        right_hand_dist = left_hand_dist = None

    if use_center_dist:
        image_center = kwimage.Boxes([default_bbox], "xywh").center
        default_center_dist = [image_center[0][0][0] * 2, image_center[1][0][0] * 2]

        distances_to_center = []
        for i in range(num_act):
            obj_conf = act[i]

            obj_bbox = kwimage.Boxes([bboxes[i]], "xywh")
            obj_center = obj_bbox.center

            center_dist = (
                dist_to_center(image_center, obj_center)
                if obj_conf != 0
                else default_center_dist
            )

            distances_to_center.append(center_dist)

    #########################
    # Intersection
    #########################
    if use_intersection:

        def intersect(hand_bbox, bbox):
            if (
                list(hand_bbox.data[0]) == default_bbox
                or list(bbox.data[0]) == default_bbox
            ):
                # one or both of the boxes are missing
                return 0

            iarea = hand_bbox.isect_area(bbox)
            hand_area = hand_bbox.area

            v = iarea / hand_area

            return v[0][0]

        right_hand_intersection = []
        left_hand_intersection = []

        for i in range(num_act):
            obj_bbox = kwimage.Boxes([bboxes[i]], "xywh")

            i_right_obj = intersect(right_hand_bbox, obj_bbox)
            right_hand_intersection.append(i_right_obj)

            i_left_obj = intersect(left_hand_bbox, obj_bbox)
            left_hand_intersection.append(i_left_obj)
    else:
        right_hand_intersection = left_hand_intersection = None

    #########################
    # Feature vector
    #########################
    feature_vec = []
    # Add hand data
    for hand_conf, hand_idx, hand_dist, hand_intersection in [
        (right_hand_conf, right_hand_idx, right_hand_dist, right_hand_intersection),
        (left_hand_conf, left_hand_idx, left_hand_dist, left_hand_intersection),
    ]:
        if use_activation:
            feature_vec.append([hand_conf])
        if use_hand_dist:
            hd1 = [
                item
                for ii, tupl in enumerate(hand_dist)
                for item in tupl
                if ii not in [right_hand_idx, left_hand_idx]
            ]
            feature_vec.append(hd1)
        if use_center_dist:
            feature_vec.append(distances_to_center[hand_idx])

    if use_hand_dist:
        feature_vec.append(right_hand_dist[left_hand_idx])
    if use_intersection:
        feature_vec.append([right_hand_intersection[left_hand_idx]])

    # Add object data
    for i in range(num_act):
        if i in [right_hand_idx, left_hand_idx]:
            # We already have the hand data
            continue

        if use_activation:
            feature_vec.append([act[i]])
        if use_intersection:
            feature_vec.append([right_hand_intersection[i]])
            feature_vec.append([left_hand_intersection[i]])
        if use_center_dist:
            feature_vec.append(distances_to_center[i])

    feature_vec = [item for sublist in feature_vec for item in sublist]  # flatten
    feature_vec = np.array(feature_vec, dtype=np.float64)

    return feature_vec
