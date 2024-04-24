from typing import Dict
from typing import Tuple

import kwimage

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def convert_labels_to_left_right_hands(
    label_vec,
    xs, ys,
):
    """
    Update any hand labels in ``label_vec`` to be 
    hand (left) or hand (right) based on the bbox's position relative to
    the center of the image
    """
    hands_possible_labels = ['hand (right)', 'hand (left)', 'hand', 'hands']
    non_objects_labels = ['patient', 'user']
    image_center = 1280//2 # hardcoded width?
    hands_loc_dict = {}
    for i, label in enumerate(label_vec):
        if label not in hands_possible_labels:
            continue

        hand_center = (xs[i] + ws[i])//2
        # Roughly determine if the hand is left or right based on 
        # which side of the image it appears on (assumes ego centric view)
        if hand_center < image_center:
            # left hand
            if "hand (left)" not in hands_loc_dict.keys():
                label_vec[i] = "hand (left)"
                hands_loc_dict[label_vec[i]] = (hand_center, i)
            else:
                if hand_center > hands_loc_dict["hand (left)"][0]:
                    label_vec[i] = "hand (right)"
                    hands_loc_dict[label_vec[i]] = (hand_center, i)
                else:
                    prev_index = hands_loc_dict["hand (left)"][1]
                    label_vec[prev_index] = "hand (right)"
                    label_vec[i] = "hand (left)"
        else:
            # right hand
            if "hand (right)" not in hands_loc_dict.keys():
                label_vec[i] = "hand (right)"
                hands_loc_dict[label_vec[i]] = (hand_center, i)
            else:
                if hand_center < hands_loc_dict["hand (right)"][0]:
                    label_vec[i] = "hand (left)"
                    hands_loc_dict[label_vec[i]] = (hand_center, i)
                else:
                    prev_index = hands_loc_dict["hand (right)"][1]
                    label_vec[prev_index] = "hand (left)"
                    label_vec[i] = "hand (right)"
    
    # Remap the label ids to include left and right hand labels
    # And remove the non object labels
    if "hand" in label_to_ind.keys():
        label_to_ind_tmp = {}
        for key, value in label_to_ind.items():
            if key == "hand":
                label_to_ind_tmp["hand (left)"] = value
                label_to_ind_tmp["hand (right)"] = value + 1
            elif key in non_objects_labels:
                continue
            else:
                label_to_ind_tmp[key] = value + 1
        
        label_to_ind = label_to_ind_tmp
    
    return label_vec, label_to_ind

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
    label_to_ind: Dict[str, int],
    version: int = 1,
    top_n_objects=3,
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
    elif version == 6:
        """
        Feature vector that encodes the distance of each object from each hand,
        the intersection of each object to the hands,
        the distance from the center of the hands to each patient joint,
        and the distance from the center of each object to each patient joint,
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
            use_joint_hand_offset=True,
            use_joint_object_offset=True,
            top_n_objects=top_n_objects,
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
    pose_keypoints,
    label_to_ind: Dict[str, int],
    use_activation=False,
    use_hand_dist=False,
    use_center_dist=False,
    use_intersection=False,
    use_joint_hand_offset=False,
    use_joint_object_offset=False,
):
    #########################
    # Default values
    #########################
    default_dist = (0, 0)  # (1280 * 2, 720 * 2)
    default_center_dist = (0, 0)  # (1280, 720)
    default_bbox = [0, 0, 0, 0]  # [0, 0, 1280, 720]
    default_center = ([[0]], [[0]])  # kwimage.Boxes([default_bbox], "xywh").center
    zero_joint_offset = [0 for i in range(22)]

    #########################
    # Data
    #########################
    # Number of object detection classes
    num_det_classes = len(label_to_ind)

    # Maximum confidence observe per-class across input object detections.
    # If a class has not been observed, it is set to 0 confidence.
    det_class_max_conf = np.zeros(num_det_classes)
    # The bounding box of the maximally confident detection
    det_class_bbox = np.zeros((num_det_classes, 4), dtype=np.float64)
    det_class_bbox[:] = default_bbox

    # Binary mask indicate which detection classes are present on this frame.
    det_class_mask = np.zeros(num_det_classes, dtype=np.bool_)

    # Record the most confident detection for each object class as recorded in
    # `label_to_ind` (confidence & bbox)
    for i, label in enumerate(label_vec):
        if label in label_to_ind:
            conf = label_confidences[i]

            ind = label_to_ind[label]
            det_class_mask[ind] = True
            
            if conf > det_class_max_conf[ind]:
                det_class_max_conf[ind] = conf
                det_class_bbox[ind] = [xs[i], ys[i], ws[i], hs[i]]  # xywh

    det_class_kwboxes = kwimage.Boxes(det_class_bbox, "xywh")

    #########################
    # util functions
    #########################
    def find_hand(hand_str):
        hand_idx = label_to_ind[hand_str]
        hand_conf = det_class_max_conf[hand_idx]
        hand_bbox = kwimage.Boxes([det_class_bbox[hand_idx]], "xywh")

        return hand_idx, hand_bbox, hand_conf, hand_bbox.center

    def dist_to_center(center1, center2):
        center_dist = [
            center2[0][0][0] - center1[0][0][0],
            center2[1][0][0] - center1[1][0][0],
        ]
        return center_dist

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

    RIGHT_IDX = 0
    LEFT_IDX = 1
    right_left_hand_kwboxes = det_class_kwboxes[[right_hand_idx, left_hand_idx]]
    # Mask detailing hand presence in the scene.
    hand_mask = det_class_mask[[right_hand_idx, left_hand_idx]]
    # 2-D mask object class gate per hand
    hand_by_object_mask = np.dot(hand_mask[:, None], det_class_mask[None, :])

    #########################
    # Distances
    #########################
    if use_hand_dist:
        # Compute distances to the right and left hands. Distance to the hand
        # is defined by `hand.center - object.center`.
        # `kwcoco.Boxes.center` returns a tuple of two arrays, each shaped
        # [n_boxes, 1].
        obj_centers_x, obj_centers_y = det_class_kwboxes.center  # [n_dets, 1]
        hand_centers_x, hand_centers_y = right_left_hand_kwboxes.center  # [2, 1]
        # Hand distances from objects. Shape: [2, n_dets]
        hand_dist_x = np.subtract(
            hand_centers_x,
            obj_centers_x.T,
            where=hand_by_object_mask,
            # required, otherwise indices may be left uninitialized.
            out=np.zeros(shape=hand_by_object_mask.shape),
        )
        hand_dist_y = np.subtract(
            hand_centers_y,
            obj_centers_y.T,
            where=hand_by_object_mask,
            # required, otherwise indices may be left uninitialized.
            out=np.zeros(shape=hand_by_object_mask.shape),
        )
        # Collate into arrays of (x, y) coordinates.
        right_hand_dist = np.stack(
            [hand_dist_x[RIGHT_IDX], hand_dist_y[RIGHT_IDX]], axis=1
        )
        left_hand_dist = np.stack(
            [hand_dist_x[LEFT_IDX], hand_dist_y[LEFT_IDX]], axis=1
        )

    else:
        right_hand_dist = left_hand_dist = None

    if use_center_dist:
        image_center = kwimage.Boxes([default_bbox], "xywh").center
        default_center_dist = [image_center[0][0][0] * 2, image_center[1][0][0] * 2]

        distances_to_center = []
        for i in range(num_det_classes):
            obj_conf = det_class_max_conf[i]

            obj_bbox = kwimage.Boxes([det_class_bbox[i]], "xywh")
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
        # Computing hand-object intersection.
        # Intersection here is defined as the percentage of the hand box
        # intersected by the representative object bounding-box.
        # If a hand or object is not present in the scene, then their
        # respective intersection area is 0.
        hand_obj_intersection_vol = right_left_hand_kwboxes.isect_area(
            det_class_kwboxes
        )
        right_left_hand_area = right_left_hand_kwboxes.area
        # Handling avoiding div-by-zero using the `where` parameter.
        hand_obj_intersection = np.divide(
            hand_obj_intersection_vol,
            right_left_hand_area,
            where=right_left_hand_area != 0,
            # Specifying out otherwise there may be uninitialized values in
            # indices where `right_left_hand_area == 0`.
            out=np.zeros_like(hand_obj_intersection_vol),
        )
        right_hand_intersection = hand_obj_intersection[0]
        left_hand_intersection = hand_obj_intersection[1]

    else:
        right_hand_intersection = left_hand_intersection = None

    #########################
    # Joints
    #########################
    def calc_joint_offset(bbox_center):
        hand_point = np.array(bbox_center) # TODO: check x,y order
        
        offset_vector = []
        if pose_keypoints == zero_joint_offset:
            return zero_joint_offset
        
        for joint in pose_keypoints:
            jx, jy = joint['xy']
            joint_point = np.array((jx, jy))
            dist = np.linalg.norm(joint_point - hand_point)
            offset_vector.append(dist)
            
        return offset_vector

    # HAND - JOINTS
    if use_joint_hand_offset:
        joint_left_hand_offset = calc_joint_offset(left_hand_center)
        joint_right_hand_offset = calc_joint_offset(right_hand_center)

    # OBJECTS - JOINTS
    if use_joint_object_offset:
        joint_object_offset = []
        for i in range(num_det_classes):
            obj_bbox = kwimage.Boxes([det_class_bbox[i]], "xywh")
            obj_center = obj_bbox.center

            offset_vector = calc_joint_offset(obj_center)
            joint_object_offset.append(offset_vector)

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

    # Add distance and intersection between hands.
    if use_hand_dist:
        feature_vec.append(right_hand_dist[left_hand_idx])
    if use_intersection:
        feature_vec.append([right_hand_intersection[left_hand_idx]])

    # Add object data
    for i in range(num_det_classes):
        if i in [right_hand_idx, left_hand_idx]:
            # We already have the hand data
            continue

        if use_activation:
            feature_vec.append([det_class_max_conf[i]])
        if use_intersection:
            feature_vec.append([right_hand_intersection[i]])
            feature_vec.append([left_hand_intersection[i]])
        if use_center_dist:
            feature_vec.append(distances_to_center[i])
    
    # Add joint data
    if use_joint_hand_offset:
        feature_vec.append(joint_left_hand_offset)
        feature_vec.append(joint_right_hand_offset)
    if use_joint_object_offset:
        for i in range(num_det_classes):
            if i in [right_hand_idx, left_hand_idx]:
                # We already have the hand data
                continue
            feature_vec.append(joint_object_offset[i])

    feature_vec = [item for sublist in feature_vec for item in sublist]  # flatten
    feature_vec = np.array(feature_vec, dtype=np.float64)

    return feature_vec
