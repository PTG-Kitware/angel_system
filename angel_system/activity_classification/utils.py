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
            top_n_objects=top_n_objects
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
    use_joint_hand_offset=False,
    use_joint_object_offset=False,
    top_n_objects=3,
):
    """
    `label_vec`, `xs`, `ys`, `ws`, hs` are to all be parallel in association
    and describe the object detections to create an embedding from.

    :param label_vec: Object label of the most confident class for each
        detection.
    :param xs: Upper-left X coordinate for each detection.
    :param ys: Upper-left Y coordinate for each detection.
    :param ws: Pixel width for each detection.
    :param hs: Pixel height for each detection.
    :param label_confidences: Confidence value of the most confident class for
        each detection.
    :param label_to_ind: Mapping of detection class indices
    :param use_activation:
    :param use_hand_dist:
    :param use_center_dist:
    :param use_intersection:

    :return: Feature vector embedding of the input detections.
    """
    #########################
    # Default values
    #########################
    default_dist = (0, 0)  # (1280 * 2, 720 * 2)
    default_center_dist = (0, 0)  # (1280, 720)
    default_bbox = [0, 0, 0, 0]  # [0, 0, 1280, 720]
    default_center = ([[0]], [[0]])  # kwimage.Boxes([default_bbox], "xywh").center
    width, height = 1280, 720
    image_center = width//2

    #########################
    # Data
    #########################
    # Number of object detection classes
    num_det_classes = len(label_to_ind) + 1 # accomedate 2 hands instead of 1, accomedate top 3 objects
    
    # print(f"label_to_ind: {label_to_ind}")
    
    # modify label_to_ind dict to accomedate 2 hands instead of 1
    if "hand" in label_to_ind.keys() or "hands (right)" in label_to_ind.keys() or "hands (left)" in label_to_ind.keys():
        hands_label_exists = True
        label_to_ind_tmp = {}
        for key, value in label_to_ind.items():
            if key == "hand":
                label_to_ind_tmp["hands (left)"] = value
                label_to_ind_tmp["hands (right)"] = value + 1
            else:
                label_to_ind_tmp[key] = value + 1
        
        label_to_ind = label_to_ind_tmp
    else:
        hands_label_exists = False
    
    # print(f'label_to_ind: {label_to_ind}, label_to_ind_tmp: {label_to_ind_tmp}')
    # print(f'label_confidences: {label_confidences}, label_to_ind_tmp: {label_to_ind_tmp}')
    # exit()
    # print(f"num_det_classes: {num_det_classes}")
    # top_n_objects = 3
    # Maximum confidence observe per-class across input object detections.
    # If a class has not been observed, it is set to 0 confidence.
    det_class_max_conf = np.zeros((num_det_classes, top_n_objects))
    # The bounding box of the maximally confident detection
    det_class_bbox = np.zeros((top_n_objects, num_det_classes, 4), dtype=np.float64)
    det_class_bbox[:] = default_bbox

    # Binary mask indicate which detection classes are present on this frame.
    det_class_mask = np.zeros(num_det_classes, dtype=np.bool_)

    # Record the most confident detection for each object class as recorded in
    # `label_to_ind` (confidence & bbox)
    # if hands_label_exists:
    #     hands_loc_dict = {}
    #     for i, label in enumerate(label_vec):
    #         # if label == "hands":
    #         #     hand_center = xs[i] + ws[i]//2
    #         #     if hand_center < image_center:
    #         #         if "hands (left)" not in hands_loc_dict.keys():
    #         #             label_vec[i] = "hands (left)"
    #         #             hands_loc_dict[label_vec[i]] = (hand_center, i)
    #         #         else:
    #         #             if hand_center > hands_loc_dict["hands (left)"][0]:
    #         #                 label_vec[i] = "hands (right)"
    #         #                 hands_loc_dict[label_vec[i]] = (hand_center, i)
    #         #     else:
    #         #         if "hands (right)" not in hands_loc_dict.keys():
    #         #             label_vec[i] = "hands (right)"
    #         #             hands_loc_dict[label_vec[i]] = (hand_center, i)
    #         #         else:
    #         #             if hand_center < hands_loc_dict["hands (right)"][0]:
    #         #                 label_vec[i] = "hands (right)"
    #         #                 hands_loc_dict[label_vec[i]] = (hand_center, i)
    #         #         # label_vec[i] = "hands (right)"
    #         if label == "tourniquet_label":
    #             label_vec[i] = "tourniquet_tourniquet"
    #         elif label == "tourniquet_windlass":
    #             label_vec[i] = "tourniquet_tourniquet"
    
    # print(f"label vec: {label_vec}")
    for i, label in enumerate(label_vec):
        if label in label_to_ind:
            conf = label_confidences[i]
            ind = label_to_ind[label]
            det_class_mask[ind] = True
            conf_list = det_class_max_conf[ind, :]
            # print(f"conf_list: {conf_list}")
            # print(f"label: {label}, conf: {conf}")
            if conf > det_class_max_conf[ind].min():
                # conf_list = np.append(conf_list, conf)
                # conf_list = np.sort(conf_list)[::-1][:-1]
                # first_zero = np.where(conf_list==0)#[0][0]
                # if len(first_zero)==0:
                first_zero = np.where(conf_list==conf_list.min())#[0][0]
                # print(first_zero)
                first_zero = first_zero[0][0]
                # print(f"first_zero: {first_zero}")
                conf_list[first_zero] = conf
                # print(f"label: {label}, conf_list: {conf_list}")
                # print(f"conf_list: {conf_list}")
                obj_index = np.where(conf_list==conf)

                det_class_max_conf[ind] = conf_list
                # det_class_max_conf[ind] = conf
                det_class_bbox[obj_index, ind] = [xs[i], ys[i], ws[i], hs[i]]  # xywh

    det_class_kwboxes = kwimage.Boxes(det_class_bbox, "xywh")
    # print(f"det_class_kwboxes: {det_class_kwboxes.shape}")
    # print(f'label_vec: {label_vec}')
    #########################
    # util functions
    #########################
    def find_hand(hand_str):
        # hand_str = "hands"
        hand_idx = label_to_ind[hand_str]
        # print(f"hand_index: {hand_idx}")
        hand_conf = det_class_max_conf[hand_idx][0]
        hand_bbox = kwimage.Boxes([det_class_bbox[0, hand_idx]], "xywh")

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
        "hands (right)"
    )

    # Find the left hand
    (left_hand_idx, left_hand_bbox, left_hand_conf, left_hand_center) = find_hand(
        "hands (left)"
    )

    RIGHT_IDX = 0
    LEFT_IDX = 1
    right_left_hand_kwboxes = det_class_kwboxes[0,[right_hand_idx, left_hand_idx]]
    # Mask detailing hand presence in the scene.
    hand_mask = det_class_mask[[right_hand_idx, left_hand_idx]]
    # 2-D mask object class gate per hand
    hand_by_object_mask = np.dot(hand_mask[:, None], det_class_mask[None, :])

    # print(f"hand_by_object_mask: {(top_n_objects, *hand_by_object_mask.shape)}")
    #########################
    # Distances
    #########################
    if use_hand_dist:
        # Compute distances to the right and left hands. Distance to the hand
        # is defined by `hand.center - object.center`.
        # `kwcoco.Boxes.center` returns a tuple of two arrays, each shaped
        # [n_boxes, 1].
        
        all_obj_centers_x, all_obj_centers_y = det_class_kwboxes.center  # [n_dets, 1]
        hand_centers_x, hand_centers_y = right_left_hand_kwboxes.center  # [2, 1]
        # Hand distances from objects. Shape: [2, n_dets]
        right_hand_dist_n = np.zeros((top_n_objects, hand_by_object_mask.shape[1], hand_by_object_mask.shape[0]))
        left_hand_dist_n = np.zeros((top_n_objects, hand_by_object_mask.shape[1], hand_by_object_mask.shape[0]))
        # print(f"left_hand_dist_n: {left_hand_dist_n.shape}")
        for object_index in range(top_n_objects):
            obj_centers_x =  all_obj_centers_x[object_index]
            obj_centers_y = all_obj_centers_y[object_index]  # [n_dets, 1]
            
            # print(f"obj_centers_x: {obj_centers_x.shape}")
            # print(f"hand_centers_x: {hand_centers_x.shape}")
            # print(f"obj_centers_y: {obj_centers_y.shape}")
            # print(f"hand_centers_y: {hand_centers_y.shape}")
            # print(f"hand_by_object_mask: {hand_by_object_mask.shape}")
            
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
            
            right_hand_dist_n[object_index] = right_hand_dist
            left_hand_dist_n[object_index] = left_hand_dist
            # print(f"right_hand_dist: {right_hand_dist.shape}")
            # print(f"left_hand_dist: {left_hand_dist.shape}")

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
        right_hand_intersection_n = np.zeros((top_n_objects, hand_by_object_mask.shape[1]))
        left_hand_intersection_n = np.zeros((top_n_objects, hand_by_object_mask.shape[1]))
        for object_index in range(top_n_objects):
            
            hand_obj_intersection_vol = right_left_hand_kwboxes.isect_area(
                det_class_kwboxes[object_index]
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
            
            right_hand_intersection_n[object_index] = right_hand_intersection
            left_hand_intersection_n[object_index] = left_hand_intersection
            
            # print(f"right_hand_intersection: {right_hand_intersection.shape}")

    else:
        right_hand_intersection = left_hand_intersection = None

    #########################
    # Feature vector
    #########################
    feature_vec = []
    # Add hand data
    for object_index in range(top_n_objects):
        
        right_hand_dist = right_hand_dist_n[object_index]
        left_hand_dist = left_hand_dist_n[object_index]
        
        right_hand_intersection = right_hand_intersection_n[object_index]
        left_hand_intersection = left_hand_intersection_n[object_index]
        
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
                feature_vec.append([det_class_max_conf[i][object_index]])
            if use_intersection:
                feature_vec.append([right_hand_intersection[i]])
                feature_vec.append([left_hand_intersection[i]])
            if use_center_dist:
                feature_vec.append(distances_to_center[i])

    feature_vec = [item for sublist in feature_vec for item in sublist]  # flatten
    # feature_vec = np.array(feature_vec, dtype=np.float64)
    # print(f"feature vector: {len(feature_vec)}")

    # print(f"feature vector: {feature_vec.shape}")
    
    return feature_vec
