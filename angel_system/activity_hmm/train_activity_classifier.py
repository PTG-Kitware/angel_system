import os
import yaml
import pickle
import kwcoco
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import average_precision_score

from support_functions import sanitize_str

from angel_system.impls.detect_activities.detections_to_activities.utils import (
    obj_det2d_set_to_feature,
)


def data_loader(fn, act_labels):
    print("Loading data....")
    # Description to ID map.
    act_map = {}
    inv_act_map = {}
    for step in act_labels["labels"]:
        act_map[sanitize_str(step["label"])] = step["id"]
        inv_act_map[step["id"]] = step["label"]

    if 0 not in act_map.values():
        act_map["background"] = 0
        inv_act_map[0] = "Background"

    # Load object detections
    dset = kwcoco.CocoDataset(fn)
    print(f"Loaded dset from file: {fn}")

    image_activity_gt = {}
    image_id_to_dataset = {}
    for img_id in dset.imgs:
        im = dset.imgs[img_id]
        gid = im["id"]
        image_id_to_dataset[gid] = os.path.split(im["file_name"])[0]

        activity_gt = im["activity_gt"]
        if activity_gt is None:
            continue

        image_activity_gt[gid] = act_map[sanitize_str(activity_gt)]

    dsets = sorted(list(set(image_id_to_dataset.values())))
    image_id_to_dataset = {
        i: dsets.index(image_id_to_dataset[i]) for i in image_id_to_dataset
    }

    min_cat = min([dset.cats[i]["id"] for i in dset.cats])
    num_act = len(dset.cats)
    label_to_ind = {dset.cats[i]["name"]: dset.cats[i]["id"] - min_cat for i in dset.cats}
    act_id_to_str = {dset.cats[i]["id"]: dset.cats[i]["name"] for i in dset.cats}

    ann_by_image = {}
    for i in dset.anns:
        ann = dset.anns[i]
        if ann["image_id"] not in ann_by_image:
            ann_by_image[ann["image_id"]] = [ann]
        else:
            ann_by_image[ann["image_id"]].append(ann)

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
    act_map,
    image_activity_gt,
    image_id_to_dataset,
    label_to_ind,
    act_id_to_str,
    ann_by_image,
):
    print("Computing features...")
    X = []
    y = []
    dataset_id = []
    last_dset = 0
    for image_id in sorted(list(ann_by_image.keys())):
        label_vec = []
        left = []
        right = []
        top = []
        bottom = []
        label_confidences = []
        obj_obj_contact_state = []
        obj_obj_contact_conf = []
        obj_hand_contact_state = []
        obj_hand_contact_conf = []

        for ann in ann_by_image[image_id]:
            label_vec.append(act_id_to_str[ann["category_id"]])
            left.append(ann["bbox"][0])
            right.append(ann["bbox"][1])
            top.append(ann["bbox"][2])
            bottom.append(ann["bbox"][3])
            label_confidences.append(ann["confidence"])

            try:
                obj_obj_contact_state.append(ann["obj-obj_contact_state"])
                obj_obj_contact_conf.append(ann["obj-obj_contact_conf"])
                obj_hand_contact_state.append(ann["obj-hand_contact_state"])
                obj_hand_contact_conf.append(ann["obj-hand_contact_conf"])
            except KeyError:
                pass

        feature_vec = obj_det2d_set_to_feature(
            label_vec,
            left,
            right,
            top,
            bottom,
            label_confidences,
            None,
            obj_obj_contact_state,
            obj_obj_contact_conf,
            obj_hand_contact_state,
            obj_hand_contact_conf,
            label_to_ind,
            version=1,
        )

        X.append(feature_vec.ravel())
        try:
            dataset_id.append(image_id_to_dataset[image_id])
            last_dset = dataset_id[-1]
        except:
            dataset_id.append(last_dset)

        try:
            y.append(image_activity_gt[image_id])
        except:
            y.append(0)

    X = np.array(X)
    y = np.array(y)
    dataset_id = np.array(dataset_id)

    return X, y


def plot_dataset_counts(X, y, output_dir, split):
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


def train(X, y):
    print("Train...")
    # Train model on test set.
    clf = RandomForestClassifier(
        max_depth=10,
        random_state=0,
        n_estimators=1000,
        max_features=0.1,
        bootstrap=True,
        verbose=True
    )
    clf.fit(X, y)

    return clf

def validate(clf, X_final_test, y_final_test):
    y_score = clf.predict_proba(X_final_test)

    lb = preprocessing.LabelBinarizer()
    y_true = lb.fit(range(y_score.shape[1])).transform(y_final_test)
    
    s = average_precision_score(y_true, y_score)
    print(f"Average precision: {s}")

    return s

def save(output_dir, act_str_list, label_to_ind, clf):
    output_fn = f"{output_dir}/activity_weights.pkl"
    with open(output_fn, "wb") as of:
        pickle.dump([label_to_ind, 1, clf, act_str_list], of)
    print(f"Saved weights to {output_fn}")

def train_activity_classifier(args):
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
    """
    parser.add_argument(
        "--pred-fnames",
        help="Object detections in kwcoco format",
        dest="pred_fnames",
        type=Path,
        nargs="+",
    )
    """
    parser.add_argument(
        "--train",
        help="Object detections in kwcoco format for the train set",
        dest="train_fn",
        type=Path
    )
    parser.add_argument(
        "--val",
        help="Object detections in kwcoco format for the validation set",
        dest="val_fn",
        type=Path
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

    """
    ###############
    # filepaths
    ###############
    pred_fnames = [
        "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_test.mscoco.json",
        "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_train_activity.mscoco.json",
        "/angel_workspace/ros_bags/m2_all_data_cleaned_fixed_with_steps/m2_all_data_cleaned_fixed_with_steps_results_val.mscoco.json",
    ]
    act_label_yaml = "/angel_workspace/config/activity_labels/medical_tourniquet.v2.yaml"
    output_fn = "/angel_workspace/model_files/recipe_m2_apply_tourniquet_v0.052.pkl"
    """


if __name__ == "__main__":
    main()
