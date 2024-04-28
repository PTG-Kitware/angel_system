import os

from typing import Dict
from typing import Tuple

import kwimage
import random

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from PIL import Image
from pathlib import Path


#########################
# Default values
#########################
default_dist = (0, 0)  # (1280 * 2, 720 * 2)
default_center_dist = (0, 0)  # (1280, 720)
default_bbox = [0, 0, 0, 0]  # [0, 0, 1280, 720]
default_center = ([[0]], [[0]])  # kwimage.Boxes([default_bbox], "xywh").center
zero_joint_offset = [0 for i in range(22)]

colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(colors)

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

def feature_version_to_options(feature_version):
    options = {}

    """
    Feature vector that encodes the activation feature of each class

    Len: num_obj_classes

    [A[obj1] ... A[objN]]
    """
    options[1] = {      
        "use_activation": True
    }

    """
    Feature vector that encodes the distance of each object from each hand,
    and the activation features

    Len: 1 + (num_obj_classes-2)*2 + 1 + (num_obj_classes-2)*2 + 2 + num_obj_classes-2

    [
        A[right hand],
        D[right hand, obj1]x, D[right hand, obj1]y, ... , D[right hand, objN]y,
        A[left hand],
        D[left hand, obj1]x, D[left hand, obj1]y, ... , D[left hand, objN]y,
        D[right hand, left hand]x, D[right hand, left hand]y,
        A[obj1] ... A[objN]
    ]
    """
    options[2] = {
        "use_activation": True,
        "use_hand_dist": True,
    }

    """
    Feature vector that encodes the distance of each object to the center of the frame,
    the intersection of each object to the hands,
    and the activation features

    Len: 1 + 2 + 1 + 2 + 1 + (1+1+1+2)*(num_obj_classes-2)

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
        ...
    ]
    """
    options[3] = {
        "use_activation": True,
        "use_center_dist": True,
        "use_intersection": True
    }

    """
    Feature vector that encodes the distance of each object from each hand,
    the intersection of each object to the hands,
    and the activation features

    Len: 1 + ((num_obj_classes-2)*2) + 1 + ((num_obj_classes-2)*2) + 2 + 1 + (3 * (num_obj_classes-2))

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
    options[5] = {
        "use_activation": True,
        "use_hand_dist": True,
        "use_intersection": True
    }

    """
    Feature vector that encodes the distance of each object from each hand,
    the intersection of each object to the hands,
    the distance from the center of the hands to each patient joint,
    and the distance from the center of each object to each patient joint,
    and the activation features

    Len: 1 + ((num_obj_classes-2)*2) + 1 + ((num_obj_classes-2)*2)
    + 2 + 1 + (1 + 1 + 1)*(num_obj_classes-2)
    + 22*2 + 22*2
    + ((22)*(num_obj_classes-2))*2
    [
        A[right hand],
        D[right hand, obj1]x, D[right hand, obj1]y, ... , D[right hand, objN]y,
        A[left hand],
        D[left hand, obj1]x, D[left hand, obj1]y, ... , D[left hand, objN]y,
        D[right hand, left hand]x, D[right hand, left hand]y,
        I[right hand, left hand],
        A[obj1] I[right hand, obj1] I[left hand, obj1], ... , I[left hand, objN],
        D[right hand, joint1]x, ... , D[right hand, joint 22]y,
        D[left hand, joint1]x, ... , D[left hand, joint 22]y,
        D[obj1, joint1]x, ... , D[obj1, joint22]y,
        ..., 
        D[objN, joint1]x, ... , D[objN, joint22]y
    ]
    """
    options[6] = {
        "use_activation": True,
        "use_hand_dist": True,
        "use_intersection": True,
        "use_joint_hand_offset": True,
        "use_joint_object_offset": True
    }

    return options[feature_version]

def obj_det2d_set_to_feature(
    label_vec,
    xs,
    ys,
    ws,
    hs,
    label_confidences,
    pose_keypoints,
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
    opts = feature_version_to_options(version)
    feature_vec = obj_det2d_set_to_feature_by_method(
        label_vec,
        xs,
        ys,
        ws,
        hs,
        label_confidences,
        pose_keypoints,
        label_to_ind,
        **opts,
        top_n_objects=top_n_objects
    )

    # print(f"feat {feature_vec}")
    # print(len(feature_vec))
    return feature_vec

def plot_feature_vec(
        image_fn,
        right_hand_center, left_hand_center, feature_vec,
        obj_label_to_ind, joint_names=["nose", "mouth", "throat", "chest", "stomach", "left_upper_arm", "right_upper_arm", "left_lower_arm", "right_lower_arm", "left_wrist", "right_wrist", "left_hand", "right_hand", "left_upper_leg", "right_upper_leg", "left_knee", "right_knee", "left_lower_leg", "right_lower_leg", "left_foot", "right_foot", "back"],
        use_activation=False,
        use_hand_dist=False,
        use_center_dist=False,
        use_intersection=False,
        use_joint_hand_offset=False,
        use_joint_object_offset=False,
        colors=['yellow', 'red', 'green', 'lightblue', 'blue', 'purple', 'orange'],
        output_dir="feature_visualization"
    ):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rh_dists = []
    lh_dists = []
    obj_confs = []
    obj_im_center_dists = []
    rh_joint_dists = []
    lh_joint_dists = []
    obj_joint_dists = []      

    non_object_labels = ["hand (left)", "hand (right)", "user", "patient"]
    labels = sorted(obj_label_to_ind)
    for non_obj_label in non_object_labels:
        labels.remove(non_obj_label)

    # Right hand
    if use_activation:
        ind = 0
        right_hand_conf = feature_vec[ind]
    if use_hand_dist:
        for obj_label in labels:
            ind += 1
            obj_rh_dist_x = feature_vec[ind]
            ind += 1
            obj_rh_dist_y = feature_vec[ind]

            rh_dists.append([obj_rh_dist_x, obj_rh_dist_y])
    if use_center_dist:
        ind += 1
        rh_im_center_dist_x = feature_vec[ind]
        ind += 1
        rh_im_center_dist_y = feature_vec[ind]

    # Left hand
    if use_activation:
        ind +=1
        left_hand_conf = feature_vec[ind]

    if use_hand_dist:
        # Left hand distances
        for obj_label in labels:
            ind += 1
            obj_lh_dist_x = feature_vec[ind]
            ind += 1
            obj_lh_dist_y = feature_vec[ind]

            lh_dists.append([obj_lh_dist_x, obj_lh_dist_y])
    if use_center_dist:
        ind += 1
        lh_im_center_dist_x = feature_vec[ind]
        ind += 1
        lh_im_center_dist_y = feature_vec[ind]
    
    # Right - left hand
    if use_hand_dist:
        # Right - left hand distance
        ind += 1
        lh_rh_dist_x = feature_vec[ind]
        ind += 1
        lh_rh_dist_y = feature_vec[ind]
    if use_intersection:
        ind += 1
        lh_rh_intersect = feature_vec[ind]

    # Objects
    for obj_label in labels:
        if use_activation:
            # Object confidence
            ind += 1
            obj_conf = feature_vec[ind]

            obj_confs.append(obj_conf)

        if use_intersection:
            # obj - right hand intersection
            ind += 1
            obj_rh_intersect = feature_vec[ind]
            # obj - left hand intersection
            ind += 1
            obj_lh_intersect = feature_vec[ind]

        if use_center_dist:
            ind += 1
            obj_im_center_dist_x = feature_vec[ind]
            ind += 1
            obj_im_center_dist_y = feature_vec[ind]

            obj_im_center_dists.append([obj_im_center_dist_x, obj_im_center_dist_y])
    
    # Joints
    if use_joint_hand_offset:
        # right hand - joints distances
        for i in range(22):
            ind += 1
            rh_jointi_dist_x = feature_vec[ind]
            ind += 1
            rh_jointi_dist_y = feature_vec[ind]

            rh_joint_dists.append([rh_jointi_dist_x, rh_jointi_dist_y])
    
        # left hand - joints distances
        for i in range(22):
            ind += 1
            lh_jointi_dist_x = feature_vec[ind]
            ind += 1
            lh_jointi_dist_y = feature_vec[ind]

            lh_joint_dists.append([lh_jointi_dist_x, lh_jointi_dist_y])
        
    if use_joint_object_offset:
        # obj - joints distances
        for obj_label in labels:
            joints_dists = []
            for i in range(22):
                ind += 1
                obj_jointi_dist_x = feature_vec[ind]
                ind += 1
                obj_jointi_dist_y = feature_vec[ind]

                joints_dists.append([obj_jointi_dist_x, obj_jointi_dist_y])

            obj_joint_dists.append(joints_dists)

    # Draw
    fig, ((rh_dist_ax, lh_dist_ax), (im_center_dist_ax, empty_ax), (rh_joint_dist_ax, lh_joint_dist_ax)) = plt.subplots(3, 2, figsize=(15, 15))
    axes = [rh_dist_ax, lh_dist_ax, im_center_dist_ax, empty_ax, rh_joint_dist_ax, lh_joint_dist_ax]
    flags = [use_hand_dist, use_hand_dist, use_center_dist, False, use_joint_hand_offset, use_joint_hand_offset]

    rh_dist_ax.set_title('Objects from distance to right hand')
    lh_dist_ax.set_title('Objects from distance to left hand')
    im_center_dist_ax.set_title('Objects from distance to image center')
    rh_joint_dist_ax.set_title('Joints from distance to right hand')
    lh_joint_dist_ax.set_title('Joints from distance to left hand')

    image = Image.open(image_fn)
    image = np.array(image)

    # Default values for each plot
    for ax, flag in zip(axes, flags):
        if not flag:
            continue

        ax.imshow(image)

        ax.plot(right_hand_center[0], right_hand_center[1], color=colors[0], marker='o')
        ax.annotate(f"hand (right): {round(right_hand_conf, 2)}", right_hand_center, color="black", annotation_clip=False)

        ax.plot(left_hand_center[0], left_hand_center[1], color=colors[1], marker='o')
        ax.annotate(f"hand (left): {round(left_hand_conf, 2)}", left_hand_center, color="black", annotation_clip=False)

    def draw_points_by_distance(ax, distances, pt, color, labels, confs):
        # Make sure the reference point exists
        if pt == [default_center[0][0][0], default_center[1][0][0]]:
            return

        for i, dist in enumerate(distances):
            # Make sure the object point exists
            if dist == list(default_dist):
                continue

            obj_pt = [pt[0] - dist[0], pt[1] - dist[1]] # pt - obj_pt = dist
            
            ax.plot([pt[0], obj_pt[0]], [pt[1], obj_pt[1]], color=color, marker='o')
            ax.annotate(f"{labels[i]}: {round(confs[i], 2)}", obj_pt, color="black", annotation_clip=False)
    
    if use_hand_dist:
        rh_dist_color = colors[2]
        draw_points_by_distance(rh_dist_ax, rh_dists, right_hand_center, rh_dist_color, labels, obj_confs)

        lh_dist_color = colors[3]
        draw_points_by_distance(lh_dist_ax, lh_dists, left_hand_center, lh_dist_color, labels, obj_confs)

    if use_center_dist:
        obj_im_center_dist_color = colors[4]
        image_center = [1280//2, 720//2]
        ax.plot(image_center, color=colors[1], marker='o')
        ax.annotate("image_center", image_center, color="black", annotation_clip=False)
        draw_points_by_distance(im_center_dist_ax, obj_im_center_dists, image_center, obj_im_center_dist_color, labels, obj_confs)
        
    if use_joint_hand_offset:
        rh_joint_color = colors[6]
        draw_points_by_distance(rh_joint_dist_ax, rh_joint_dists, right_hand_center, rh_joint_color, joint_names, [1]*len(joint_names))
        lh_joint_color = colors[5]
        draw_points_by_distance(lh_joint_dist_ax, lh_joint_dists, left_hand_center, lh_joint_color, joint_names, [1]*len(joint_names))
    
    plt.savefig(f"{output_dir}/{os.path.basename(image_fn)}")
    plt.close(fig)

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
    top_n_objects=1 # TODO
):
    

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
            center1[0][0][0] - center2[0][0][0],
            center1[1][0][0] - center2[1][0][0],
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
        offset_vector = []
        if pose_keypoints == zero_joint_offset:
            for joint in pose_keypoints:
                offset_vector.append(default_dist)
            return offset_vector
        
        for joint in pose_keypoints:
            jx, jy = joint['xy']
            joint_point = [jx, jy]
            dist = [bbox_center[0][0][0] - joint_point[0], bbox_center[1][0][0] - joint_point[1]]
            #dist = np.linalg.norm(joint_point - hand_point)
            offset_vector.append(dist)
            
        return offset_vector

    # HAND - JOINTS
    if use_joint_hand_offset:
        joint_right_hand_offset = calc_joint_offset(right_hand_center)
        joint_left_hand_offset = calc_joint_offset(left_hand_center)

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
        for rh_offset in joint_right_hand_offset:
            feature_vec.append(rh_offset)
        for lh_offset in joint_left_hand_offset:
            feature_vec.append(lh_offset)
        
    if use_joint_object_offset:
        for i in range(num_det_classes):
            if i in [right_hand_idx, left_hand_idx]:
                # We already have the hand data
                continue
            for offset in joint_object_offset[i]:
                feature_vec.append(offset)

    feature_vec = [item for sublist in feature_vec for item in sublist]  # flatten
    feature_vec = np.array(feature_vec, dtype=np.float64)

    return feature_vec
