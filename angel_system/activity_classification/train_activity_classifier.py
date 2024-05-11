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
        - obj_label_to_ind: Object detection labels to ids dict
        - obj_ind_to_label: Object detection ids to labels dict
        - ann_by_image: Image id to annotation dict
    """
    print("Loading data....")

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
    obj_label_to_ind = {
        dset.cats[i]["name"]: dset.cats[i]["id"] - min_cat for i in dset.cats
    }
    print(f"Object label mapping:\n\t", obj_label_to_ind)
    obj_ind_to_label = {dset.cats[i]["id"]: dset.cats[i]["name"] for i in dset.cats}

    ann_by_image = {}
    for gid, anns in dset.index.gid_to_aids.items():
        ann_by_image[gid] = []
        for ann_id in anns:
            ann = dset.anns[ann_id]
            ann_by_image[gid].append(ann)

    return (
        act_map,
        inv_act_map,
        image_activity_gt,
        image_id_to_dataset,
        obj_label_to_ind,
        obj_ind_to_label,
        ann_by_image,
    )


def compute_feats(
    act_map: dict,
    image_activity_gt: dict,
    image_id_to_dataset: dict,
    obj_label_to_ind: dict,
    obj_ind_to_label: dict,
    ann_by_image: dict,
    feat_version=1,
    top_k_objects=1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute features from object detections

    :param act_map: Activity label string to id
    :param image_activity_gt: Image id to activity label string dict
    :param image_id_to_dataset: Image id to id in ``dset`` dict
    :param obj_label_to_ind: Object detection labels to ids dict
    :param obj_ind_to_label: Object detection ids to labels dict
    :param ann_by_image: Image id to annotation dict
    :param feat_version:
        Version of the feature conversion approach.
    :param top_k_objects: Number top confidence objects to use per label, defaults to 1

    :return: resulting feature data and its labels
    """
    print("Computing features...")
    X = []
    Y = []
    dataset_id = []
    last_dset = 0
    zero_joint_offset = [0 for i in range(22)]

    for image_id in sorted(list(ann_by_image.keys())):
        label_vec = []
        xs = []
        ys = []
        ws = []
        hs = []
        label_confidences = []
        pose_keypoints = []

        # Reorganize detections into lists
        if len(ann_by_image[image_id]) == 0:
            continue
        pose_keypoints = zero_joint_offset
        for ann in ann_by_image[image_id]:
            cat = obj_ind_to_label[ann["category_id"]]

            # Ignore the patient and user bboxes, use the pose from the patient
            if cat in ["patient", "user"]:
                if cat == "patient":
                    pose_keypoints = ann["keypoints"]
                continue

            label_vec.append(cat)

            x, y, w, h = ann["bbox"]
            xs.append(x)
            ys.append(y)
            ws.append(w)
            hs.append(h)

            label_confidences.append(ann["confidence"])

        # Ignore the patient and user labels in the feature vector
        only_obj_label_to_ind = {
            k: i
            for i, (k, v) in enumerate(obj_label_to_ind.items())
            if k not in ["patient", "user"]
        }

        # Compute feature vector
        feature_vec = obj_det2d_set_to_feature(
            label_vec,
            xs,
            ys,
            ws,
            hs,
            label_confidences,
            pose_keypoints,
            only_obj_label_to_ind,
            version=feat_version,
            top_k_objects=top_k_objects,
        )

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
    obj_label_to_ind: dict,
    clf: RandomForestClassifier,
):
    """Save the model to a pickle file

    :param output_dir: Path to save the model to
    :param act_str_list: List of activity label strings
    :param obj_label_to_ind: Object detection labels to ids dict
    :param clf: model
    """
    output_fn = f"{output_dir}/activity_weights.pkl"
    with open(output_fn, "wb") as of:
        pickle.dump([obj_label_to_ind, 1, clf, act_str_list], of)
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
        obj_label_to_ind,
        obj_ind_to_label,
        ann_by_image,
    ) = data_loader(args.train_fn, act_labels)
    X, y = compute_feats(
        act_map,
        image_activity_gt,
        image_id_to_dataset,
        obj_label_to_ind,
        obj_ind_to_label,
        ann_by_image,
    )
    plot_dataset_counts(X, y, args.output_dir, "train")

    # Load validation dataset
    (
        val_act_map,
        val_inv_act_map,
        val_image_activity_gt,
        val_image_id_to_dataset,
        val_obj_label_to_ind,
        val_obj_ind_to_label,
        val_ann_by_image,
    ) = data_loader(args.val_fn, act_labels)
    X_final_test, y_final_test = compute_feats(
        val_act_map,
        val_image_activity_gt,
        val_image_id_to_dataset,
        val_obj_label_to_ind,
        val_obj_ind_to_label,
        val_ann_by_image,
    )
    plot_dataset_counts(X_final_test, y_final_test, args.output_dir, "val")

    # Train
    clf = train(X, y)
    ap = validate(clf, X_final_test, y_final_test)

    # Save
    act_str_list = [inv_act_map[key] for key in sorted(list(set(y)))]
    save(args.output_dir, act_str_list, obj_label_to_ind, clf)


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
