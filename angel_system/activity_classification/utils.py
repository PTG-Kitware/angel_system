import os

from typing import Dict, Tuple, List

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
default_center_list = [default_center[0][0][0], default_center[1][0][0]]
zero_joint_offset = [0 for i in range(22)]

random_colors = list(mcolors.CSS4_COLORS.keys())
random.shuffle(random_colors)


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
        List of x values, List of y values, List of width values, List of height values
    """
    assert (
        len(top) == len(left) == len(bottom) == len(right)
    ), "No all input array-likes were the same length."
    xs = np.asarray(left)
    ys = np.asarray(top)
    ws = np.asarray(right) - xs
    hs = np.asarray(bottom) - ys
    return xs, ys, ws, hs


def feature_version_to_options(feature_version: int) -> Dict[str, bool]:
    """Convert the feature version number to a dict of
    boolean flags indicating which data values should be added to the feature vector

    :param feature_version: Version of the feature conversion approach.

    :return:
        Dictionary of flag names and boolean values that match the input parameters
        to the functions that create/utilize the feature vector
    """
    options = {}

    """
    Feature vector that encodes the activation feature of each class

    Len: top_k_objects * num_obj_classes

    [
        for k_obj in top_k_object:
            A[obj1] ... A[objN]
    ]
    """
    options[1] = {"use_activation": True}

    """
    Feature vector that encodes the distance of each object from each hand,
    and the activation features

    Len:
    top_k_objects * (
        1 + (num_obj_classes-2)*2 + 1 + (num_obj_classes-2)*2 + 2 + (num_obj_classes-2)
    )

    [
        for k_obj in top_k_object:
            A[right hand],
            D[right hand, obj1_k]x, D[right hand, obj1_k]y, ... , D[right hand, objN_k]y,
            A[left hand],
            D[left hand, obj1_k]x, D[left hand, obj1_k]y, ... , D[left hand, objN_k]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            A[obj1_k] ... A[objN_k]
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

    Len:
    top_k_objects * (
        1 + 2 + 1 + 2 + 1 + (1 + 1 + 1 + 2) * (num_obj_classes-2)
    )

    [
        for k_obj in top_k_object:
            A[right hand],
            D[right hand, center]x, D[right hand, center]y,
            A[left hand],
            D[left hand, center]x, D[left hand, center]y,
            I[right hand, left hand],
            A[obj1_k] I[right hand, obj1_k] I[left hand, obj1_k], D[obj1_k, center]x, D[obj1_k, center]y ... , D[objN_k, center]y
    ]
    """
    options[3] = {
        "use_activation": True,
        "use_center_dist": True,
        "use_intersection": True,
    }

    """
    Feature vector that encodes the distance of each object from each hand,
    the intersection of each object to the hands,
    and the activation features

    Len: 
    top_k_objects * (
        1 + 2 * (num_obj_classes-2) + 1 + 2 * (num_obj_classes-2) + 2 + 1 + (1 + 1 + 1) * (num_obj_classes-2)
    )
    
    [
        for k_obj in top_k_object:
            A[right hand],
            D[right hand, obj1_k]x, D[right hand, obj1_k]y, ... , D[right hand, objN_k]y,
            A[left hand],
            D[left hand, obj1_k]x, D[left hand, obj1_k]y, ... , D[left hand, objN_k]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            I[right hand, left hand],
            A[obj1_k] I[right hand, obj1_k] I[left hand, obj1_k], ... , I[left hand, objN_k]
    ]
    """
    options[5] = {
        "use_activation": True,
        "use_hand_dist": True,
        "use_intersection": True,
    }

    """
    Feature vector that encodes the distance of each object from each hand,
    the intersection of each object to the hands,
    the distance from the center of the hands to each patient joint,
    and the distance from the center of each object to each patient joint,
    and the activation features

    Len: 
    top_k_objects * (
        (1 + (num_obj_classes-2)*2) * 2  + 2 + 1
        + (num_obj_classes-2) * (1+1+1)
    )
    + 22*2 + 22*2
    + top_k_objects * ((22*2)*(num_obj_classes-2))

    
    [
        for k_obj in top_k_object:
            A[right hand],
            D[right hand, obj1_k]x, D[right hand, obj1_k]y, ... , D[right hand, objN_k]y,
            A[left hand],
            D[left hand, obj1_k]x, D[left hand, obj1_k]y, ... , D[left hand, objN_k]y,
            D[right hand, left hand]x, D[right hand, left hand]y,
            I[right hand, left hand],
            A[obj1_k] I[right hand, obj1_k] I[left hand, obj1_k], ... , I[left hand, objN_k],
        D[left hand, joint1]x, ... , D[left hand, joint 22]y,
        D[right hand, joint1]x, ... , D[right hand, joint 22]y,
        for k_obj in top_k_object:
            D[obj1_k, joint1]x, ... , D[obj1_k, joint22]y,
            ..., 
            D[objN_k, joint1]x, ... , D[objN_k, joint22]y
    ]
    """
    options[6] = {
        "use_activation": True,
        "use_hand_dist": True,
        "use_intersection": True,
        "use_joint_hand_offset": True,
        "use_joint_object_offset": True,
    }

    return options[feature_version]


def obj_det2d_set_to_feature(
    label_vec: List[str],
    xs: List[float],
    ys: List[float],
    ws: List[float],
    hs: List[float],
    label_confidences: List[float],
    pose_keypoints: List[Dict],
    obj_label_to_ind: Dict[str, int],
    version: int = 1,
    top_k_objects: int = 1,
):
    """Convert ObjectDetection2dSet fields into a feature vector.

    :param label_vec: List of object labels for each detection (length: # detections)
    :param xs: List of x values for each detection (length: # detections)
    :param ys: List of y values for each detection (length: # detections)
    :param ws: List of width values for each detection (length: # detections)
    :param hs: List of height values for each detection (length: # detections)
    :param label_confidences: List of confidence values for each detection (length: # detections)
    :param pose_keypoints:
        List of joints, represented by a dictionary contining the x and y corrdinates of the points and the category id and string
    :param obj_label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.
    :param version:
        Version of the feature conversion approach.
    :param top_k_objects: Number top confidence objects to use per label, defaults to 1

    :return: resulting feature data
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
        obj_label_to_ind,
        top_k_objects=top_k_objects,
        **opts,
    )

    # print(f"feat {feature_vec}")
    # print(len(feature_vec))
    return feature_vec


def plot_feature_vec(
    image_fn: str,
    right_hand_center: list,
    left_hand_center: list,
    feature_vec: np.array,
    obj_label_to_ind: Dict[str, int],
    output_dir: str,
    top_k_objects: int = 1,
    use_activation: bool = False,
    use_hand_dist: bool = False,
    use_center_dist: bool = False,
    use_intersection: bool = False,
    use_joint_hand_offset: bool = False,
    use_joint_object_offset: bool = False,
    joint_names: List[str] = [
        "nose",
        "mouth",
        "throat",
        "chest",
        "stomach",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "left_upper_leg",
        "right_upper_leg",
        "left_knee",
        "right_knee",
        "left_lower_leg",
        "right_lower_leg",
        "left_foot",
        "right_foot",
        "back",
    ],
    colors: List[str] = [
        "yellow",
        "red",
        "green",
        "lightblue",
        "blue",
        "purple",
        "orange",
    ],
):
    """Plot the object and joint points based on the hand bbox centers and the distance values
    in the feature vector

    :param image_fn: Path to the image to draw on
    :param right_hand_center: List of the x and y coordinates of the right hand box center
    :param left_hand_center: List of the x and y coordinates of the left hand box center
    :param feature_vec: Numpy array of values determined by the provided flags
    :param obj_label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.
    :param output_dir: Path to a folder to save the generated images to
    :param top_k_objects: Number top confidence objects to use per label, defaults to 1
    :param use_activation: If True, add the confidence values of the detections to the feature vector, defaults to False
    :param use_hand_dist: If True, add the distance of the detection centers to both hand centers to the feature vector, defaults to False
    :param use_intersection: If True, add the intersection of the detection boxes with the hand boxes to the feature vector, defaults to False
    :param use_joint_hand_offset: If True, add the distance of the hand centers to the patient joints to the feature vector, defaults to False
    :param use_joint_object_offset: If True, add the distance of the object centers to the patient joints to the feature vector, defaults to False
    :param joint_names: List of the joint names
    :param colors: List of colors to use when plotting points
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rh_joint_dists = []
    lh_joint_dists = []
    rh_dists_k = [[] for i in range(top_k_objects)]
    lh_dists_k = [[] for i in range(top_k_objects)]
    obj_confs_k = [[] for i in range(top_k_objects)]
    obj_im_center_dists_k = [[] for i in range(top_k_objects)]
    obj_joint_dists_k = [[] for i in range(top_k_objects)]

    non_object_labels = ["hand (left)", "hand (right)", "user", "patient"]
    labels = sorted(obj_label_to_ind)
    for non_obj_label in non_object_labels:
        labels.remove(non_obj_label)

    ind = -1
    for object_k_index in range(top_k_objects):
        # RIGHT HAND
        if use_activation:
            ind += 1
            right_hand_conf = feature_vec[ind]

        if use_hand_dist:
            for obj_label in labels:
                ind += 1
                obj_rh_dist_x = feature_vec[ind]
                ind += 1
                obj_rh_dist_y = feature_vec[ind]

                rh_dists_k[object_k_index].append([obj_rh_dist_x, obj_rh_dist_y])

        if use_center_dist:
            ind += 1
            rh_im_center_dist_x = feature_vec[ind]
            ind += 1
            rh_im_center_dist_y = feature_vec[ind]

        # LEFT HAND
        if use_activation:
            ind += 1
            left_hand_conf = feature_vec[ind]

        if use_hand_dist:
            # Left hand distances
            for obj_label in labels:
                ind += 1
                obj_lh_dist_x = feature_vec[ind]
                ind += 1
                obj_lh_dist_y = feature_vec[ind]

                lh_dists_k[object_k_index].append([obj_lh_dist_x, obj_lh_dist_y])

        if use_center_dist:
            ind += 1
            lh_im_center_dist_x = feature_vec[ind]
            ind += 1
            lh_im_center_dist_y = feature_vec[ind]

        # Right - left hand
        if use_hand_dist:
            # Right - left hand distance
            ind += 1
            rh_lh_dist_x = feature_vec[ind]
            ind += 1
            rh_lh_dist_y = feature_vec[ind]
        if use_intersection:
            ind += 1
            lh_rh_intersect = feature_vec[ind]

        # OBJECTS
        for obj_label in labels:
            if use_activation:
                # Object confidence
                ind += 1
                obj_conf = feature_vec[ind]

                obj_confs_k[object_k_index].append(obj_conf)

            if use_intersection:
                # obj - right hand intersection
                ind += 1
                obj_rh_intersect = feature_vec[ind]
                # obj - left hand intersection
                ind += 1
                obj_lh_intersect = feature_vec[ind]

            if use_center_dist:
                # image center - obj distances
                ind += 1
                obj_im_center_dist_x = feature_vec[ind]
                ind += 1
                obj_im_center_dist_y = feature_vec[ind]

                obj_im_center_dists_k[object_k_index].append(
                    [obj_im_center_dist_x, obj_im_center_dist_y]
                )

    # HANDS-JOINTS
    if use_joint_hand_offset:
        # left hand - joints distances
        for i in range(22):
            ind += 1
            lh_jointi_dist_x = feature_vec[ind]
            ind += 1
            lh_jointi_dist_y = feature_vec[ind]

            lh_joint_dists.append([lh_jointi_dist_x, lh_jointi_dist_y])

        # right hand - joints distances
        for i in range(22):
            ind += 1
            rh_jointi_dist_x = feature_vec[ind]
            ind += 1
            rh_jointi_dist_y = feature_vec[ind]

            rh_joint_dists.append([rh_jointi_dist_x, rh_jointi_dist_y])

    # OBJS-JOINTS
    if use_joint_object_offset:
        for object_k_index in range(top_k_objects):
            # obj - joints distances
            for obj_label in labels:
                joints_dists = []
                for i in range(22):
                    ind += 1
                    obj_jointi_dist_x = feature_vec[ind]
                    ind += 1
                    obj_jointi_dist_y = feature_vec[ind]

                    joints_dists.append([obj_jointi_dist_x, obj_jointi_dist_y])

                obj_joint_dists_k[object_k_index].append(joints_dists)

    # Draw
    fig, (
        (lh_dist_ax, rh_dist_ax),
        (im_center_dist_ax, obj_joint_dist_ax),
        (lh_joint_dist_ax, rh_joint_dist_ax),
    ) = plt.subplots(3, 2, figsize=(15, 15))
    axes = [
        rh_dist_ax,
        lh_dist_ax,
        im_center_dist_ax,
        obj_joint_dist_ax,
        rh_joint_dist_ax,
        lh_joint_dist_ax,
    ]
    flags = [
        use_hand_dist,
        use_hand_dist,
        use_center_dist,
        use_joint_object_offset,
        use_joint_hand_offset,
        use_joint_hand_offset,
    ]

    rh_dist_ax.set_title("Objects from distance to right hand")
    lh_dist_ax.set_title("Objects from distance to left hand")
    im_center_dist_ax.set_title("Objects from distance to image center")
    obj_joint_dist_ax.set_title("Joints from distance to objects*")
    rh_joint_dist_ax.set_title("Joints from distance to right hand")
    lh_joint_dist_ax.set_title("Joints from distance to left hand")

    rh_dist_color = colors[2]
    lh_dist_color = colors[3]
    obj_im_center_dist_color = colors[4]
    lh_joint_color = colors[5]
    rh_joint_color = colors[6]

    image = Image.open(image_fn)
    image = np.array(image)

    # Default values for each plot
    for ax, flag in zip(axes, flags):
        if not flag:
            continue

        ax.imshow(image)

        ax.plot(right_hand_center[0], right_hand_center[1], color=colors[0], marker="o")
        ax.annotate(
            f"hand (right): {round(right_hand_conf, 2)}",
            right_hand_center,
            color="black",
            annotation_clip=False,
        )

        ax.plot(left_hand_center[0], left_hand_center[1], color=colors[1], marker="o")
        ax.annotate(
            f"hand (left): {round(left_hand_conf, 2)}",
            left_hand_center,
            color="black",
            annotation_clip=False,
        )

    def draw_points_by_distance(ax, distances, pt, color, labels, confs):
        # Make sure the reference point exists
        if pt == default_center_list:
            return

        for i, dist in enumerate(distances):
            # Make sure the object point exists
            if dist == list(default_dist):
                continue

            obj_pt = [pt[0] - dist[0], pt[1] - dist[1]]  # pt - obj_pt = dist

            ax.plot([pt[0], obj_pt[0]], [pt[1], obj_pt[1]], color=color, marker="o")
            ax.annotate(
                f"{labels[i]}: {round(confs[i], 2)}",
                obj_pt,
                color="black",
                annotation_clip=False,
            )

    if use_joint_hand_offset:
        draw_points_by_distance(
            rh_joint_dist_ax,
            rh_joint_dists,
            right_hand_center,
            rh_joint_color,
            joint_names,
            [1] * len(joint_names),
        )
        draw_points_by_distance(
            lh_joint_dist_ax,
            lh_joint_dists,
            left_hand_center,
            lh_joint_color,
            joint_names,
            [1] * len(joint_names),
        )

    if use_hand_dist:
        rh_dist_ax.plot(
            [right_hand_center[0], right_hand_center[0] - rh_lh_dist_x],
            [right_hand_center[1], right_hand_center[1] - rh_lh_dist_y],
            color=random_colors[0],
            marker="o",
        )

    for object_k_index in range(top_k_objects):
        if use_hand_dist:
            draw_points_by_distance(
                rh_dist_ax,
                rh_dists_k[object_k_index],
                right_hand_center,
                rh_dist_color,
                labels,
                obj_confs_k[object_k_index],
            )
            draw_points_by_distance(
                lh_dist_ax,
                lh_dists_k[object_k_index],
                left_hand_center,
                lh_dist_color,
                labels,
                obj_confs_k[object_k_index],
            )

        if use_center_dist:
            image_center = [1280 // 2, 720 // 2]
            im_center_dist_ax.plot(image_center, color=colors[1], marker="o")
            im_center_dist_ax.annotate(
                "image_center", image_center, color="black", annotation_clip=False
            )
            draw_points_by_distance(
                im_center_dist_ax,
                obj_im_center_dists_k[object_k_index],
                image_center,
                obj_im_center_dist_color,
                labels,
                obj_confs_k[object_k_index],
            )

        if use_joint_object_offset:

            obj_pts = []
            if use_hand_dist:
                if right_hand_center != default_center_list:
                    obj_pts = [
                        (
                            [
                                right_hand_center[0] - rh_dist[0],
                                right_hand_center[1] - rh_dist[1],
                            ]
                            if rh_dist != list(default_dist)
                            else default_center_list
                        )
                        for rh_dist in rh_dists_k[object_k_index]
                    ]
                elif left_hand_center != default_center_list:
                    obj_pts = [
                        (
                            [
                                left_hand_center[0] - lh_dist[0],
                                left_hand_center[1] - lh_dist[1],
                            ]
                            if lh_dist != list(default_dist)
                            else default_center_list
                        )
                        for lh_dist in lh_dists_k[object_k_index]
                    ]
            elif use_center_dist:
                obj_pts = [
                    (
                        [
                            image_center[0] - im_center_dist[0],
                            image_center[1] - im_center_dist[1],
                        ]
                        if im_center_dist != list(default_dist)
                        else default_center_list
                    )
                    for im_center_dist in obj_im_center_dists_k[object_k_index]
                ]

            if not obj_pts:
                continue

            for i, obj_pt in enumerate(obj_pts):
                if obj_pt == default_center_list:
                    continue

                obj_joint_color = random_colors[(object_k_index * len(obj_pt)) + i]
                obj_joint_dist_ax.plot(
                    obj_pt[0], obj_pt[1], color=obj_joint_color, marker="o"
                )
                obj_joint_dist_ax.annotate(
                    f"{labels[i]}: {round(obj_confs_k[object_k_index][i], 2)}",
                    obj_pt,
                    color="black",
                    annotation_clip=False,
                )
                draw_points_by_distance(
                    obj_joint_dist_ax,
                    obj_joint_dists_k[object_k_index][i],
                    obj_pt,
                    obj_joint_color,
                    joint_names,
                    [1] * len(joint_names),
                )

    Path(f"{output_dir}/full_feature_vec").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{output_dir}/full_feature_vec/{os.path.basename(image_fn)}")

    def copy_ax_to_new_fig(ax, subfolder):
        ax.remove()

        fig2 = plt.figure(figsize=(15, 15))
        ax.figure = fig2
        fig2.axes.append(ax)
        fig2.add_axes(ax)

        dummy = fig2.add_subplot(111)
        ax.set_position(dummy.get_position())
        dummy.remove()

        Path(f"{output_dir}/{subfolder}").mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/{subfolder}/{os.path.basename(image_fn)}")

        plt.close(fig2)

    # Save each subplot as its own image
    for ax, subfolder, flag in zip(
        [
            lh_dist_ax,
            rh_dist_ax,
            im_center_dist_ax,
            obj_joint_dist_ax,
            lh_joint_dist_ax,
            rh_joint_dist_ax,
        ],
        [
            "left_hand_obj_dist",
            "right_hand_obj_dist",
            "image_center_obj_dist",
            "obj_joints_dist",
            "left_hand_joints_dist",
            "right_hand_joints_dist",
        ],
        flags,
    ):
        if not flag:
            continue
        copy_ax_to_new_fig(ax, subfolder)

    plt.close(fig)


def obj_det2d_set_to_feature_by_method(
    label_vec: List[str],
    xs: List[float],
    ys: List[float],
    ws: List[float],
    hs: List[float],
    label_confidences: List[float],
    pose_keypoints: List[Dict],
    obj_label_to_ind: Dict[str, int],
    top_k_objects: int = 1,
    use_activation: bool = False,
    use_hand_dist: bool = False,
    use_center_dist: bool = False,
    use_intersection: bool = False,
    use_joint_hand_offset: bool = False,
    use_joint_object_offset: bool = False,
):
    """
    :param label_vec: List of object labels for each detection (length: # detections)
    :param xs: List of x values for each detection (length: # detections)
    :param ys: List of y values for each detection (length: # detections)
    :param ws: List of width values for each detection (length: # detections)
    :param hs: List of height values for each detection (length: # detections)
    :param label_confidences: List of confidence values for each detection (length: # detections)
    :param pose_keypoints:
        List of joints, represented by a dictionary contining the x and y corrdinates of the points and the category id and string
    :param obj_label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.
    :param top_k_objects: Number top confidence objects to use per label, defaults to 1
    :param use_activation: If True, add the confidence values of the detections to the feature vector, defaults to False
    :param use_hand_dist: If True, add the distance of the detection centers to both hand centers to the feature vector, defaults to False
    :param use_intersection: If True, add the intersection of the detection boxes with the hand boxes to the feature vector, defaults to False
    :param use_joint_hand_offset: If True, add the distance of the hand centers to the patient joints to the feature vector, defaults to False
    :param use_joint_object_offset: If True, add the distance of the object centers to the patient joints to the feature vector, defaults to False

    :return:
        resulting feature data
    """
    #########################
    # Data
    #########################
    # Number of object detection classes
    num_det_classes = len(obj_label_to_ind)

    # Maximum confidence observe per-class across input object detections.
    # If a class has not been observed, it is set to 0 confidence.
    det_class_max_conf = np.zeros((num_det_classes, top_k_objects))
    # The bounding box of the maximally confident detection
    det_class_bbox = np.zeros((top_k_objects, num_det_classes, 4), dtype=np.float64)
    det_class_bbox[:] = default_bbox

    # Binary mask indicate which detection classes are present on this frame.
    det_class_mask = np.zeros((top_k_objects, num_det_classes), dtype=np.bool_)

    # Record the most confident detection for each object class as recorded in
    # `obj_label_to_ind` (confidence & bbox)
    for i, label in enumerate(label_vec):
        assert label in obj_label_to_ind, f"Label {label} is unknown"

        conf = label_confidences[i]
        ind = obj_label_to_ind[label]

        conf_list = det_class_max_conf[ind, :]
        if conf > det_class_max_conf[ind].min():
            # Replace the lowest confidence object with our new higher confidence object
            min_conf_ind = np.where(conf_list == conf_list.min())[0][0]

            conf_list[min_conf_ind] = conf
            det_class_bbox[min_conf_ind, ind] = [xs[i], ys[i], ws[i], hs[i]]
            det_class_mask[min_conf_ind, ind] = True

            # Sort the confidences to determine the top_k order
            sorted_index = np.argsort(conf_list)[::-1]
            sorted_conf_list = np.array([conf_list[k] for k in sorted_index])

            # Reorder the values to match the confidence top_k order
            det_class_max_conf[ind] = sorted_conf_list

            bboxes = det_class_bbox.copy()
            mask = det_class_mask.copy()
            for idx, sorted_ind in enumerate(sorted_index):
                det_class_bbox[idx, ind] = bboxes[sorted_ind, ind]
                det_class_mask[idx, ind] = mask[sorted_ind, ind]

    det_class_kwboxes = kwimage.Boxes(det_class_bbox, "xywh")

    #########################
    # util functions
    #########################
    def find_hand(hand_str):
        hand_idx = obj_label_to_ind[hand_str]
        hand_conf = det_class_max_conf[hand_idx][0]
        hand_bbox = kwimage.Boxes([det_class_bbox[0, hand_idx]], "xywh")

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

    right_left_hand_kwboxes = det_class_kwboxes[0, [right_hand_idx, left_hand_idx]]

    # Mask detailing hand presence in the scene.
    RIGHT_IDX = 0
    LEFT_IDX = 1
    hand_mask = [det_class_mask[0][right_hand_idx], det_class_mask[0][left_hand_idx]]
    # Mask detailing hand and object presence in the scene.
    hand_by_object_mask_k = np.zeros(
        (top_k_objects, 2, num_det_classes), dtype=np.bool_
    )

    for object_k_index in range(top_k_objects):
        x = np.array(
            [
                [
                    hand_mask[RIGHT_IDX] and det_class
                    for det_class in det_class_mask[object_k_index]
                ],
                [
                    hand_mask[LEFT_IDX] and det_class
                    for det_class in det_class_mask[object_k_index]
                ],
            ]
        )
        hand_by_object_mask_k[object_k_index] = x

    #########################
    # Hand distances
    #########################
    if use_hand_dist:
        # Compute distances to the right and left hands. Distance to the hand
        # is defined by `hand.center - object.center`.
        # `kwcoco.Boxes.center` returns a tuple of two arrays, each shaped
        # [n_boxes, 1].
        all_obj_centers_x, all_obj_centers_y = det_class_kwboxes.center  # [n_dets, 1]
        hand_centers_x, hand_centers_y = right_left_hand_kwboxes.center  # [2, 1]

        # Hand distances from objects. Shape: [top_k, n_dets, 2]
        right_hand_dist_k = np.zeros((top_k_objects, num_det_classes, 2))
        left_hand_dist_k = np.zeros((top_k_objects, num_det_classes, 2))
        for object_k_index in range(top_k_objects):
            obj_centers_x = all_obj_centers_x[object_k_index]
            obj_centers_y = all_obj_centers_y[object_k_index]

            hand_dist_x = np.subtract(
                hand_centers_x,
                obj_centers_x.T,
                where=hand_by_object_mask_k[object_k_index],
                # required, otherwise indices may be left uninitialized.
                out=np.zeros(shape=(2, num_det_classes)),
            )
            hand_dist_y = np.subtract(
                hand_centers_y,
                obj_centers_y.T,
                where=hand_by_object_mask_k[object_k_index],
                # required, otherwise indices may be left uninitialized.
                out=np.zeros(shape=(2, num_det_classes)),
            )

            # Collate into arrays of (x, y) coordinates.
            right_hand_dist = np.stack(
                [hand_dist_x[RIGHT_IDX], hand_dist_y[RIGHT_IDX]], axis=1
            )
            # for dist in right_hand_dist:
            #    if not hand_by_object_mask_k[object_k_index][RIGHT_IDX]
            left_hand_dist = np.stack(
                [hand_dist_x[LEFT_IDX], hand_dist_y[LEFT_IDX]], axis=1
            )

            right_hand_dist_k[object_k_index] = right_hand_dist
            left_hand_dist_k[object_k_index] = left_hand_dist

    else:
        right_hand_dist_k = left_hand_dist_k = None

    #########################
    # Image center
    # distances
    #########################
    if use_center_dist:
        image_center = kwimage.Boxes(
            [0, 0, 1280, 720], "xywh"
        ).center  # Hard coded image size
        default_center_dist = [image_center[0][0][0] * 2, image_center[1][0][0] * 2]

        # Object distances from image center. Shape: [top_k, n_dets, 2]
        image_center_obj_dist_k = np.zeros((top_k_objects, num_det_classes, 2))
        for object_k_index in range(top_k_objects):
            obj_centers_x = all_obj_centers_x[object_k_index]
            obj_centers_y = all_obj_centers_y[object_k_index]

            for obj_ind in range(num_det_classes):
                obj_conf = det_class_max_conf[obj_ind]

                obj_bbox = kwimage.Boxes(
                    [det_class_bbox[object_k_index][obj_ind]], "xywh"
                )
                obj_center = obj_bbox.center

                center_dist = (
                    dist_to_center(image_center, obj_center)
                    if obj_conf != 0
                    else default_center_dist
                )

                image_center_obj_dist_k[object_k_index][obj_ind] = center_dist
    else:
        image_center_obj_dist_k = None

    #########################
    # Intersection
    #########################
    if use_intersection:
        # Computing hand-object intersection.
        # Intersection here is defined as the percentage of the hand box
        # intersected by the representative object bounding-box.
        # If a hand or object is not present in the scene, then their
        # respective intersection area is 0.
        # Shape: [top_k, n_dets]
        right_hand_intersection_k = np.zeros((top_k_objects, num_det_classes))
        left_hand_intersection_k = np.zeros((top_k_objects, num_det_classes))
        for object_k_index in range(top_k_objects):
            obj_bboxes = det_class_kwboxes[object_k_index]

            hand_obj_intersection_vol = right_left_hand_kwboxes.isect_area(obj_bboxes)
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

            right_hand_intersection_k[object_k_index] = right_hand_intersection
            left_hand_intersection_k[object_k_index] = left_hand_intersection
    else:
        right_hand_intersection_k = left_hand_intersection_k = None

    #########################
    # Joints
    #########################
    def calc_joint_offset(bbox_center_x, bbox_center_y):
        offset_vector = []
        if pose_keypoints == zero_joint_offset or (
            bbox_center_x == default_center_list[0]
            and bbox_center_y == default_center_list[1]
        ):
            # If we don't have the joints or the object, return default values
            for joint in pose_keypoints:
                offset_vector.append(default_dist)
            return offset_vector

        for joint in pose_keypoints:
            jx, jy = joint["xy"]
            joint_point = [jx, jy]

            dist = [bbox_center_x - joint_point[0], bbox_center_y - joint_point[1]]
            offset_vector.append(dist)

        return offset_vector

    # HAND - JOINTS
    if use_joint_hand_offset:
        joint_right_hand_offset = calc_joint_offset(
            right_hand_center[0][0][0], right_hand_center[1][0][0]
        )
        joint_left_hand_offset = calc_joint_offset(
            left_hand_center[0][0][0], left_hand_center[1][0][0]
        )

    # OBJECTS - JOINTS
    if use_joint_object_offset:
        # Object distances from patient joints. Shape: [top_k, n_dets, 22, 2]
        obj_joints_dist_k = np.zeros((top_k_objects, num_det_classes, 22, 2))
        for object_k_index in range(top_k_objects):
            obj_centers_x = all_obj_centers_x[object_k_index]
            obj_centers_y = all_obj_centers_y[object_k_index]

            joint_object_offset = []
            for obj_ind in range(num_det_classes):
                offset_vector = calc_joint_offset(
                    obj_centers_x[obj_ind], obj_centers_y[obj_ind]
                )
                joint_object_offset.append(offset_vector)

            obj_joints_dist_k[object_k_index] = joint_object_offset

    #########################
    # Feature vector
    #########################
    feature_vec = []

    for object_k_index in range(top_k_objects):
        # HANDS
        for hand_conf, hand_idx, hand_dist in [
            (right_hand_conf, right_hand_idx, right_hand_dist_k[object_k_index]),
            (left_hand_conf, left_hand_idx, left_hand_dist_k[object_k_index]),
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
                feature_vec.append(image_center_obj_dist_k[0][hand_idx])

        # RIGHT-LEFT HAND
        if use_hand_dist:
            feature_vec.append(right_hand_dist_k[0][left_hand_idx])
        if use_intersection:
            feature_vec.append([right_hand_intersection_k[0][left_hand_idx]])

        # OBJECTS
        for obj_ind in range(num_det_classes):
            if obj_ind in [right_hand_idx, left_hand_idx]:
                # We already have the hand data
                continue

            if use_activation:
                feature_vec.append([det_class_max_conf[obj_ind][object_k_index]])
            if use_intersection:
                feature_vec.append([right_hand_intersection_k[object_k_index][obj_ind]])
                feature_vec.append([left_hand_intersection_k[object_k_index][obj_ind]])
            if use_center_dist:
                feature_vec.append(image_center_obj_dist_k[object_k_index][obj_ind])

    # HANDS-JOINTS
    if use_joint_hand_offset:
        for lh_offset in joint_left_hand_offset:
            feature_vec.append(lh_offset)

        for rh_offset in joint_right_hand_offset:
            feature_vec.append(rh_offset)

    # OBJ-JOINTS
    if use_joint_object_offset:
        for object_k_index in range(top_k_objects):
            for obj_ind in range(num_det_classes):
                if obj_ind in [right_hand_idx, left_hand_idx]:
                    # We already have the hand data
                    continue
                for offset in obj_joints_dist_k[object_k_index][obj_ind]:
                    feature_vec.append(offset)

    feature_vec = [item for sublist in feature_vec for item in sublist]  # flatten
    feature_vec = np.array(feature_vec, dtype=np.float64)

    return feature_vec
