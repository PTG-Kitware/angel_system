import os
import yaml
import pickle
import kwcoco
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from support_functions import sanitize_str

from angel_system.impls.detect_activities.detections_to_activities.utils import (
    obj_det2d_set_to_feature,
)


def data_loader(pred_fnames, act_label_yaml):
    print("Loading data....")
    # Load labels
    with open(act_label_yaml, "r") as stream:
        print(f"Loading activity labels from: {act_label_yaml}")
        act_labels = yaml.safe_load(stream)

    # Description to ID map.
    act_map = {}
    inv_act_map = {}
    for step in act_labels["labels"]:
        act_map[sanitize_str(step["full_str"])] = step["id"]
        inv_act_map[step["id"]] = step["full_str"]

    if 0 not in act_map.values():
        act_map["background"] = 0
        inv_act_map[0] = "Background"

    # Load object detections
    dat = None
    for pred_fname in pred_fnames:
        print(f"Loading dataset: {pred_fname}")
        dat_ = kwcoco.CocoDataset(pred_fname)

        if dat is not None:
            dat = dat.union(dat_)
        else:
            dat = dat_

    image_activity_gt = {}
    image_id_to_dataset = {}
    for img_id in dat.imgs:
        im = dat.imgs[img_id]
        image_id_to_dataset[im["id"]] = os.path.split(im["file_name"])[0]

        if im["activity_gt"] is None:
            continue

        image_activity_gt[im["id"]] = act_map[sanitize_str(im["activity_gt"])]

    dsets = sorted(list(set(image_id_to_dataset.values())))
    image_id_to_dataset = {
        i: dsets.index(image_id_to_dataset[i]) for i in image_id_to_dataset
    }

    min_cat = min([dat.cats[i]["id"] for i in dat.cats])
    num_act = len(dat.cats)
    label_to_ind = {dat.cats[i]["name"]: dat.cats[i]["id"] - min_cat for i in dat.cats}
    act_id_to_str = {dat.cats[i]["id"]: dat.cats[i]["name"] for i in dat.cats}

    ann_by_image = {}
    for i in dat.anns:
        ann = dat.anns[i]
        if ann["image_id"] not in ann_by_image:
            ann_by_image[ann["image_id"]] = [ann]
        else:
            ann_by_image[ann["image_id"]].append(ann)

    return act_map, inv_act_map, image_activity_gt, image_id_to_dataset, label_to_ind, act_id_to_str, ann_by_image

def compute_feats(act_map, image_activity_gt, image_id_to_dataset, label_to_ind, act_id_to_str, ann_by_image):
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

    ###############
    # Dataset
    ###############
    # Carve out final test set.
    val_fract = 0.2
    final_test_dataset_ids = sorted(list(set(dataset_id)))
    i = final_test_dataset_ids[int(np.round((len(final_test_dataset_ids) * val_fract)))]
    ind = dataset_id <= i
    X_final_test = X[ind]
    y_final_test = y[ind]
    dataset_id_final_test = dataset_id[ind]
    X = X[~ind]
    y = y[~ind]
    dataset_id = dataset_id[~ind]

    ref = set(range(len(act_map)))
    # Make sure every dataset has at least one example of each step.
    for i in sorted(list(set(dataset_id))):
        # ind = dataset_id == i
        # for j in ref.difference(set(y[ind])):
        for j in ref:
            y = np.hstack([y, j])
            X = np.vstack([X, np.zeros(X[0].shape)])
            dataset_id = np.hstack([dataset_id, i])

    return X, y

def plot_dataset_counts(X, y, output_dir):
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
    plt.savefig(f"{output_dir}/gt_counts.png")

def train(X, y):
    print("Train...")
    # Train model on test set.
    clf = RandomForestClassifier(
        max_depth=10,
        random_state=0,
        n_estimators=1000,
        max_features=0.1,
        bootstrap=True,
    )
    clf.fit(X, y)

    return clf

def save(output_dir, act_str_list, label_to_ind, clf):
    output_fn = f"{output_dir}/activity_weights.pkl"
    with open(output_fn, "wb") as of:
        pickle.dump([label_to_ind, 1, clf, act_str_list], of)
    print(f"Saved weights to {output_fn}")

def train_activity_classifier(args):
    act_map, inv_act_map, image_activity_gt, image_id_to_dataset, label_to_ind, act_id_to_str, ann_by_image = data_loader(args.pred_fnames, args.act_label_yaml)
    X, y = compute_feats(act_map, image_activity_gt, image_id_to_dataset, label_to_ind, act_id_to_str, ann_by_image)
    plot_dataset_counts(X, y, args.output_dir)
    
    clf = train(X, y)

    act_str_list = [inv_act_map[key] for key in sorted(list(set(y)))]
    save(args.output_dir, act_str_list, label_to_ind, clf)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-fnames",
        help="Object detections in kwcoco format",
        dest="pred_fnames",
        type=Path,
        nargs='+'
    )
    parser.add_argument(
        "--act-label-yaml",
        help="",
        type=Path,
        dest="act_label_yaml",
        default=""
    )
    parser.add_argument(
        "--output-dir",
        help="",
        type=Path,
        dest="output_dir",
        default=""
    )
    args = parser.parse_args()

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
