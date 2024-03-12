import os
import yaml
import pickle
import kwcoco
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from typing import Union, Tuple, List
from pathlib import Path, PosixPath
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import average_precision_score

from angel_system.data.common.load_data import sanitize_str

from angel_system.activity_classification.utils import (
    obj_det2d_set_to_feature,
)


def data_loader(
    dset: Union[str, PosixPath, kwcoco.CocoDataset], act_labels: dict
) -> Tuple[dict, dict, dict, dict, dict, dict, dict]:
    """Parse the data in ``dset``

    :param dset: kwcoco dataset
    :param act_labels: The activity labels

    :return:
        - act_map: Activity label string to id dict
        - inv_act_map: Activity id to label string dict
        - image_activity_gt: Image id to activity label string dict
        - image_id_to_dataset: Image id to id in ``dset`` dict
        - label_to_ind: Object detection labels to ids dict
        - act_id_to_str: Object detection ids to labels dict
        - ann_by_image: Image id to annotation dict
    """
    print("Loading data....")
    # Description to ID map.
    # print(f"act labels: {act_labels}")
    # exit()
    act_map = {}
    inv_act_map = {}
    for step in act_labels["labels"]:
        act_map[sanitize_str(step["label"])] = step["id"]
        inv_act_map[step["id"]] = step["label"]

    if 0 not in act_map.values():
        act_map["background"] = 0
        inv_act_map[0] = "background"

    # print(f"act_map: {act_map}")
    
    # Load object detections
    if type(dset) == str or type(dset) == PosixPath:
        dset_fn = dset
        dset = kwcoco.CocoDataset(dset_fn)
        print(f"Loaded dset from file: {dset_fn}")
        dset.fpath = dset_fn

    image_activity_gt = {}
    image_id_to_dataset = {}
    for img_id in dset.imgs:
        im = dset.imgs[img_id]
        
        # print(f"image: {im}")
        
        gid = im["id"]
        image_id_to_dataset[gid] = os.path.split(im["file_name"])[0]

        activity_gt = im["activity_gt"]
        if activity_gt is None:
            activity_gt = "background"  # continue

        image_activity_gt[gid] = act_map[sanitize_str(activity_gt)]

    dsets = sorted(list(set(image_id_to_dataset.values())))
    image_id_to_dataset = {
        i: dsets.index(image_id_to_dataset[i]) for i in image_id_to_dataset
    }

    min_cat = min([dset.cats[i]["id"] for i in dset.cats])
    num_act = len(dset.cats)
    label_to_ind = {
        dset.cats[i]["name"]: dset.cats[i]["id"] - min_cat for i in dset.cats
    }
    print(
        f"Object label mapping:\n\t"
        f"{json.dumps([o['name'] for o in dset.categories().objs])}"
    )
    act_id_to_str = {dset.cats[i]["id"]: dset.cats[i]["name"] for i in dset.cats}

    ann_by_image = {}
    for gid, anns in dset.index.gid_to_aids.items():
        ann_by_image[gid] = []
        for ann_id in anns:
            ann = dset.anns[ann_id]
            ann_by_image[gid].append(ann)
    
    # print(f"ann by image: 17069: {ann_by_image[17069]}")

    return (
        act_map,
        inv_act_map,
        image_activity_gt,
        image_id_to_dataset,
        label_to_ind,
        act_id_to_str,
        ann_by_image,
    )


def compute_feats(
    act_map: dict,
    image_activity_gt: dict,
    image_id_to_dataset: dict,
    label_to_ind: dict,
    act_id_to_str: dict,
    ann_by_image: dict,
    feat_version=1,
    objects_joints: bool =False,
    hands_joints: bool =False,
    aug_trans_range = None,
    aug_rot_range = None,
    top_n_objects=3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute features from object detections

    :param act_map: Activity label string to id
    :param image_activity_gt: Image id to activity label string dict
    :param image_id_to_dataset: Image id to id in ``dset`` dict
    :param label_to_ind: Object detection labels to ids dict
    :param act_id_to_str: Object detection ids to labels dict
    :param ann_by_image: Image id to annotation dict

    :return: resulting feature data and its labels
    """
    print("Computing features...")
    X = []
    Y = []
    dataset_id = []
    last_dset = 0

    hands_possible_labels = ['hand (right)', 'hand (left)', 'hand', 'hands']
    non_objects_labels = ['patient', 'user']
    # hands_inds = [label_to_ind[label] for label in hands_possible_labels if label in label_to_ind.keys()]
    # non_object_inds = [label_to_ind[label] for label in non_objects_labels if label in label_to_ind.keys()]
    hands_inds = [key for key, label in act_id_to_str.items() if label in hands_possible_labels]
    non_object_inds = [key for key, label in act_id_to_str.items() if label in non_objects_labels]
    object_inds = list(set(list(label_to_ind.values())) - set(hands_inds) - set(non_object_inds))
    # print(f"label_to_ind: {label_to_ind}")
    # print(f"act_id_to_str: {act_id_to_str}")
    # print(f"hands_inds: {hands_inds}")
    # print(f"object_inds: {object_inds}")
    # exit()
    
    for image_id in sorted(list(ann_by_image.keys())):
        label_vec = []
        xs = []
        ys = []
        ws = []
        hs = []
        label_confidences = []
        obj_obj_contact_state = []
        obj_obj_contact_conf = []
        obj_hand_contact_state = []
        obj_hand_contact_conf = []
        
        if objects_joints or hands_joints:
            joint_left_hand_offset = []
            joint_right_hand_offset = []
            joint_object_offset = []
        
        num_hands, num_objects = 0, 0
        
        for ann in ann_by_image[image_id]:
            if "keypoints" in ann.keys():
                pose_keypoints = ann['keypoints']
            
            elif "confidence" in ann.keys():
                label_vec.append(act_id_to_str[ann["category_id"]])
                x, y = ann["bbox"][0], ann["bbox"][1]
                w, h = ann["bbox"][2], ann["bbox"][3]
                
                if aug_trans_range != None and aug_rot_range != None:
                    
                    # print(f"performing augmentation")
                    random_translation_x = np.random.uniform(aug_trans_range[0], aug_trans_range[1])
                    random_translation_y = np.random.uniform(aug_trans_range[0], aug_trans_range[1])
                    random_rotation = np.random.uniform(aug_rot_range[0], aug_rot_range[1])
                    
                    # print(f"random_translation_x: {random_translation_x}, random_translation_y: {random_translation_y}")
                    
                    object_center_x, object_center_y = x + w//2, y + h//2
                    
                    rotation_matrix = np.array([[np.cos(random_rotation), -np.sin(random_rotation), random_translation_x], 
                                                [np.sin(random_rotation), np.cos(random_rotation), random_translation_y],
                                                [0, 0, 1]])
                    
                    # print(f"before xy: {x}, {y}")
                    
                    # x += random_translation_x
                    # y += random_translation_y
                    
                    xy = np.array([x, y, 1])
                    xy_center = np.array([object_center_x, object_center_y, 1])
                    
                    rot_xy = (xy-xy_center) @ rotation_matrix.T + xy_center
                    
                    # print(f"rot_xy: {rot_xy}")
                    # rot_xy = np.linalg.
                    
                    x = rot_xy[0]
                    y = rot_xy[1]
                    
                    # print(f"after xy: {x}, {y}")
                
                xs.append(x)
                ys.append(y)
                ws.append(w)
                hs.append(h)
                label_confidences.append(ann["confidence"])
                
                # print(f"ann: {ann}")
                
                if ann["category_id"] in hands_inds:
                    num_hands += 1
                elif ann['category_id'] in object_inds:
                    num_objects += 1
                try:
                    obj_obj_contact_state.append(ann["obj-obj_contact_state"])
                    obj_obj_contact_conf.append(ann["obj-obj_contact_conf"])
                    obj_hand_contact_state.append(ann["obj-hand_contact_state"])
                    obj_hand_contact_conf.append(ann["obj-hand_contact_conf"])
                except KeyError:
                    pass
                
        # print(f"pose keyponts: {pose_keypoints}")
        # print(f"label_vec: {label_vec}")
        image_center = 1280//2
        if num_hands > 0:
            hands_loc_dict = {}
            for i, label in enumerate(label_vec):
                if label == "hand":
                    hand_center = xs[i] + ws[i]//2
                    if hand_center < image_center:
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
        
        if "hand" in label_to_ind.keys():
            # hands_label_exists = True
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
        # else:
        #     hands_label_exists = False
        # print(f"num_hands: {num_hands}")
        # print(f"label_vec: {label_vec}")
        zero_offset = [0 for i in range(22)]
        if (num_hands > 0 or num_objects > 0) and (hands_joints or objects_joints):
            joint_object_offset = []
            for i, label in enumerate(label_vec):
                
                if hands_joints and num_hands > 0:
                    
                    if label == "hand (right)" or label == "hand (left)":
                        bx, by, bw, bh = xs[i], ys[i], ws[i], hs[i]
                        hcx, hcy = bx+(bw//2), by+(bh//2)
                        hand_point = np.array((hcx, hcy))
                        
                        offset_vector = []
                        if 'pose_keypoints' in locals():
                            for joint in pose_keypoints:
                                jx, jy = joint['xy']
                                joint_point = np.array((jx, jy))
                                dist = np.linalg.norm(joint_point - hand_point)
                                # print(f"joint_points: {joint_point}, hand_point: {hand_point}, hand_label: {label}, distance: {dist}")
                                offset_vector.append(dist)
                        else:
                            offset_vector = zero_offset
                            
                        # print(f"offset vector: {offset_vector}")
                        if label == "hand (left)":
                        # #     # hcx, hcy = bx+bw//2, by+by//w
                            joint_left_hand_offset = offset_vector
                        elif label == "hand (right)":
                            joint_right_hand_offset = offset_vector
                            
                    else:
                        if objects_joints and num_objects > 0:
                            bx, by, bw, bh = xs[i], ys[i], ws[i], hs[i]
                            ocx, ocy = bx+(bw//2), by+(bh//2)
                            object_point = np.array((ocx, ocy))
                            offset_vector = []
                            if 'pose_keypoints' in locals():
                                for joint in pose_keypoints:
                                    jx, jy = joint['xy']
                                    joint_point = np.array((jx, jy))
                                    # print(f"joint_points: {joint_point.dtype}, object_point: {object_point.dtype}")
                                    dist = np.linalg.norm(joint_point - object_point)
                                    offset_vector.append(dist)
                            else:
                                offset_vector = zero_offset
                                
                            joint_object_offset.append(offset_vector)
                        # object_offset_wrt = ann_id

                
            
        # print(f"joint_left_hand_offset: {joint_left_hand_offset}")
        # print(f"joint_right_hand_offset: {joint_right_hand_offset}")
        # print(f"joint_object_offset: {joint_object_offset}")
        # print(f"label_confidences: {label_confidences}")
        # print(f"label_vec: {label_vec}")

        feature_vec = obj_det2d_set_to_feature(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            None,
            obj_obj_contact_state,
            obj_obj_contact_conf,
            obj_hand_contact_state,
            obj_hand_contact_conf,
            label_to_ind,
            version=feat_version,
            top_n_objects=top_n_objects,
        )
        
        # exit()
        if objects_joints or hands_joints:
            zero_offset = [0 for i in range(22)]
            offset_vector = []
            if hands_joints:
                
                # print(f"joint_left_hand_offset: {len(joint_left_hand_offset)}")
                # print(f"joint_right_hand_offset: {len(joint_right_hand_offset)}")
                
                if len(joint_left_hand_offset) >= 1:
                    # print(f"joint_left_hand_offset[0]: {joint_left_hand_offset}")
                    offset_vector.extend(joint_left_hand_offset)
                else:
                    offset_vector.extend(zero_offset)
                
                if len(joint_right_hand_offset) >= 1:
                    offset_vector.extend(joint_right_hand_offset)
                else:
                    offset_vector.extend(zero_offset)
            if objects_joints:
                
                # print(f"joint_object_offset: {len(joint_object_offset)}")
                # print(f"joint_object_offset: {joint_object_offset}")
                
                for i in range(top_n_objects):
                    if len(joint_object_offset) > i:
                        offset_vector.extend(joint_object_offset[i])
                    else:
                        # print(f"offset_vector: {offset_vector}")
                        # print(f"zero_offset: {zero_offset}")
                        offset_vector.extend(zero_offset)

            
            feature_vec.extend(offset_vector)
            
        # print(f"offset_vector: {len(offset_vector)}")
        # print(f"feature_vec: {len(feature_vec)}")
        # print(f"feature_vec: {feature_vec}")
        # exit()
        
            
        feature_vec = np.array(feature_vec, dtype=np.float64)
        
        # print(f"feature vector: {feature_vec.shape}")
        
        X.append(feature_vec.ravel())
        
            
        try:
            dataset_id.append(image_id_to_dataset[image_id])
            last_dset = dataset_id[-1]
        except:
            dataset_id.append(last_dset)

        try:
            Y.append(image_activity_gt[image_id])
        except:
            Y.append(0)

    X = np.array(X)
    Y = np.array(Y)
    dataset_id = np.array(dataset_id)

    return X, Y


def plot_dataset_counts(
    X: np.ndarray, y: np.ndarray, output_dir: Union[str, PosixPath], split: str
):
    """Plot the number of ground truth steps in the data

    :param X: feature vector from object detections per image,
        shape is (# frames x # object detection labels)
    :param y: the target activity ids,
        shape is (# frames,)
    :param output_dir: Directory to save the plot to
    :param split: Which test set ``X`` and ``y`` come from. This is
        used to create a subfolder under ``output_dir``
    """
    plt.imshow(np.cov(X.T))
    plt.colorbar()

    y_ = y.tolist()
    x_ = list(range(max(y) + 1))
    counts = [y_.count(i) for i in x_]
    plt.close("all")
    fig = plt.figure(num=None, figsize=(14, 10), dpi=80)
    plt.rc("font", **{"size": 22})
    plt.rc("axes", linewidth=4)
    plt.bar(x_, counts)
    plt.xticks(x_)
    plt.ylabel("Counts", fontsize=34)
    plt.xlabel("Ground Truth Steps", fontsize=34)
    plt.tight_layout()

    plot_dir = f"{output_dir}/{split}"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(f"{plot_dir}/gt_counts.png")


def train(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train the random forest classifier

    :param X: feature vector from object detections per image,
        shape is (# frames x # object detection labels)
    :param y: the target activity ids,
        shape is (# frames,)

    :return: The trained random forest classifier
    """
    print("x", type(X), X.shape)
    print("y", type(y), y.shape)
    print("Train...")
    # Train model on test set.
    clf = RandomForestClassifier(
        max_depth=10,
        random_state=0,
        n_estimators=1000,
        max_features=0.1,
        bootstrap=True,
        verbose=True,
    )
    clf.fit(X, y)

    return clf


def validate(
    clf: RandomForestClassifier, X_final_test: np.ndarray, y_final_test: np.ndarray
) -> float:
    """Calculate the average precision of ``clf``

    :param clf: model
    :param X_final_test: feature vector from object detections per image,
        shape is (# frames x # object detection labels)
    :param y_final_test: the target activity ids,
        shape is (# frames,)

    :return: Average precision score
    """
    y_score = clf.predict_proba(X_final_test)  # num data pts x num classes

    # Convert column vector to indicator vector
    lb = preprocessing.LabelBinarizer()
    y_true = lb.fit(range(y_score.shape[1])).transform(y_final_test)

    s = average_precision_score(y_true, y_score)
    print(f"Average precision: {s}")

    return s


def save(
    output_dir: Union[str, PosixPath],
    act_str_list: List[str],
    label_to_ind: dict,
    clf: RandomForestClassifier,
):
    """Save the model to a pickle file

    :param output_dir: Path to save the model to
    :param act_str_list: List of activity label strings
    :param label_to_ind: Object detection labels to ids dict
    :param clf: model
    """
    output_fn = f"{output_dir}/activity_weights.pkl"
    with open(output_fn, "wb") as of:
        pickle.dump([label_to_ind, 1, clf, act_str_list], of)
    print(f"Saved weights to {output_fn}")


def train_activity_classifier(args: argparse.Namespace):
    """Load the data and train an activity classifier

    :param args: Input arguments
    """
    # Load labels
    with open(args.act_label_yaml, "r") as stream:
        print(f"Loading activity labels from: {args.act_label_yaml}")
        act_labels = yaml.safe_load(stream)

    # Load train dataset
    (
        act_map,
        inv_act_map,
        image_activity_gt,
        image_id_to_dataset,
        label_to_ind,
        act_id_to_str,
        ann_by_image,
    ) = data_loader(args.train_fn, act_labels)
    X, y = compute_feats(
        act_map,
        image_activity_gt,
        image_id_to_dataset,
        label_to_ind,
        act_id_to_str,
        ann_by_image,
    )
    plot_dataset_counts(X, y, args.output_dir, "train")

    # Load validation dataset
    (
        val_act_map,
        val_inv_act_map,
        val_image_activity_gt,
        val_image_id_to_dataset,
        val_label_to_ind,
        val_act_id_to_str,
        val_ann_by_image,
    ) = data_loader(args.val_fn, act_labels)
    X_final_test, y_final_test = compute_feats(
        val_act_map,
        val_image_activity_gt,
        val_image_id_to_dataset,
        val_label_to_ind,
        val_act_id_to_str,
        val_ann_by_image,
    )
    plot_dataset_counts(X_final_test, y_final_test, args.output_dir, "val")

    # Train
    clf = train(X, y)
    ap = validate(clf, X_final_test, y_final_test)

    # Save
    act_str_list = [inv_act_map[key] for key in sorted(list(set(y)))]
    save(args.output_dir, act_str_list, label_to_ind, clf)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        help="Object detections in kwcoco format for the train set",
        dest="train_fn",
        type=Path,
    )
    parser.add_argument(
        "--val",
        help="Object detections in kwcoco format for the validation set",
        dest="val_fn",
        type=Path,
    )
    parser.add_argument(
        "--act-label-yaml", help="", type=Path, dest="act_label_yaml", default=""
    )
    parser.add_argument(
        "--output-dir", help="", type=Path, dest="output_dir", default=""
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_activity_classifier(args)


if __name__ == "__main__":
    main()
