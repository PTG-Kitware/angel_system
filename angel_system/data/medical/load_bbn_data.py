"""
Load object detections and adds hand-object and object-object labels
based on the ground truth annotations.

This should be run on videos not used during training. 
"""
import os
import re
import glob
import cv2
import kwcoco
import kwimage
import shutil

import pandas as pd
import numpy as np

from angel_system.data.common.load_data import activities_from_dive_csv
from angel_system.data.medical.data_paths import KNOWN_BAD_VIDEOS

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def dive_to_activity_file(videos_dir):
    """DIVE CSV to BBN TXT frame-level annotation file format"""
    for dive_csv in glob.glob(f"{videos_dir}/*/*.csv"):
        print(dive_csv)
        video_dir = os.path.dirname(dive_csv)
        video = video_dir.split("/")[-1]

        activities = activities_from_dive_csv(dive_csv)

        with open(f"{video_dir}/{video}.skill_labels_by_frame.txt", "w") as f:
            for activity in activities:
                f.write(
                    f"{activity.start_frame}\t{activity.end_frame}\t{activity.class_label}\n"
                )


def bbn_activity_data_loader(
    videos_dir,
    video,
    step_map=None,
    lab_data=False,
):
    """Create a dictionary of start and end times for each activity in
    the BBN TXT frame-level annotation files
    """
    # Load ground truth activity
    if lab_data:
        skill_fn = glob.glob(f"{videos_dir}/{video}/*_skills_frame.txt")
        skill_fn = skill_fn[0]
    else:
        skill_fn = f"{videos_dir}/{video}/{video}.skill_labels_by_frame.txt"

    if os.path.isfile(
        f"{videos_dir}/{video}/THIS_DATA_SET_WAS_EXCLUDED"
    ) or not os.path.exists(skill_fn):
        print(f"{video} has no data")
        return {}

    skill_f = open(skill_fn)
    lines = skill_f.readlines()

    gt_activity = {}
    for line in lines:
        # start time, end time, class
        data = line.split("\t")

        class_label = data[2].lower().strip().strip(".").strip()

        try:
            start = int(data[0])
            end = int(data[1])
        except:
            continue

        if class_label not in gt_activity.keys():
            gt_activity[class_label] = []
        gt_activity[class_label].append({"start": start, "end": end})

    skill_f.close()

    print(f"Loaded ground truth from {skill_fn}")

    return gt_activity


def bbn_yolomodel_dataloader(
    root_dir, skill, valid_classes="all", split="train", filter_repeated_objs=False
):
    """
    Load the YoloModel data
    """
    data_dir = f"{root_dir}/{skill}/YoloModel"

    # Load class names
    classes_fn = f"{data_dir}/object_names.txt"
    with open(classes_fn, "r") as f:
        cats = [l.strip() for l in f.readlines()]
    if valid_classes == "all":
        valid_classes = cats

    # Load bboxes
    data = {}
    bboxes_dir = f"{data_dir}/LabeledObjects/{split}"

    for ann_fn in glob.glob(f"{bboxes_dir}/*.txt"):
        try:
            image_fn = ann_fn[:-3] + "png"
            assert os.path.exists(image_fn)
        except AssertionError:
            image_fn = ann_fn[:-3] + "jpg"
            assert os.path.exists(image_fn)

        image_name = image_fn[len(data_dir) + 1 :]

        image = cv2.imread(image_fn)
        im_h, im_w, c = image.shape

        used_classes = []
        data[image_name] = [{"im_size": {"height": im_h, "width": im_w}}]

        with open(ann_fn, "r") as f:
            dets = f.readlines()

        for det in dets:
            d = det.split(" ")
            cat = cats[int(d[0])]

            if cat not in valid_classes:
                continue

            if cat != "hand":
                if filter_repeated_objs and cat in used_classes:
                    print(f"{image_name} has repeated {cat} objects, ignoring")
                    # Ignore this image
                    del data[image_name]
                    break

            used_classes.append(cat)

            # center
            x = float(d[1]) * im_w
            y = float(d[2]) * im_h
            w = float(d[3]) * im_w
            h = float(d[4]) * im_h

            x = x - (0.5 * w)
            y = y - (0.5 * h)

            # tl_x, tl_y, br_x, br_y
            bbox = [x, y, (x + w), (y + h)]
            # bbox = [x-(0.5*w), y-(0.5*h), x+w, y+h]

            data[image_name].append(
                {
                    "area": w * h,
                    "cat": cat,
                    "segmentation": [],
                    "bbox": bbox,
                    "obj-obj_contact_state": 0,
                    "obj-hand_contact_state": 0,
                }
            )

    return valid_classes, data


def save_as_kwcoco(classes, data, save_fn="bbn-data.mscoco.json"):
    """
    Save the bboxes in the json file
    format used by the detector training
    """
    dset = kwcoco.CocoDataset()

    for class_ in classes:
        dset.add_category(name=class_)

    for im, bboxes in data.items():
        dset.add_image(
            file_name=im,
            width=bboxes[0]["im_size"]["width"],
            height=bboxes[0]["im_size"]["height"],
        )
        img = dset.index.file_name_to_img[im]

        for bbox in bboxes[1:]:
            cat = dset.index.name_to_cat[bbox["cat"]]

            xywh = (
                kwimage.Boxes([bbox["bbox"]], "tlbr").toformat("xywh").data[0].tolist()
            )

            ann = {
                "area": bbox["area"],
                "image_id": img["id"],
                "category_id": cat["id"],
                "segmentation": bbox["segmentation"],
                "bbox": xywh,
                "obj-obj_contact_state": bbox["obj-obj_contact_state"],
                "obj-hand_contact_state": bbox["obj-hand_contact_state"],
            }
            dset.add_annotation(**ann)

    dset.fpath = save_fn
    dset.dump(dset.fpath, newlines=True)


def activity_label_fixes(activity_label, target):
    if activity_label == "put_tourniquet_around":
        label = "place-tourniquet"
        label_id = 1
    if activity_label == "pulls_tight":
        label = "pull-tight"
        label_id = 2
    if activity_label == "secures" and target == "velcro_strap":
        label = "apply-strap-to-strap-body"
        label_id = 3
    if activity_label == "twist" and target == "windlass":
        label = "turn-windless"
        label_id = 4
    if (
        activity_label == "locks_into_windlass_keeper"
        or activity_label == "lock_into_windlass_keeper"
    ):
        label = "lock-windless"
        label_id = 5
    if (
        activity_label == "wraps_remaining_strap_around"
        or activity_label == "wrap_remaining_strap_around"
    ):
        label = "pull-remaining-strap"
        label_id = 6
    if activity_label == "secures" and target == "windlass":
        label = "secure-strap"
        label_id = 7
    if activity_label == "writes_on" and target == "tourniquet_label":
        label = "mark-time"
        label_id = 8

    return label, label_id


def bbn_activity_txt_to_csv(root_dir, output_dir):
    """
    Generate DIVE csv format activity annotations from BBN's text annotations

    :param root_dir: Path to a folder containing video folders
        Expected setup:
            root_dir/
                {VIDEO_NAME}/
                    {action labels by frame}.txt
                    images/
                        {filename}.png
                        ...

    """
    print(f"{root_dir}/*/*_action_labels_by_frame.txt")

    action_fns = glob.glob(
        f"{root_dir}/*/*.action_labels_by_frame.txt"
    )
    if not action_fns:
        # Lab videos
        action_fns = glob.glob(
            f"{root_dir}/*/*_skills_frame.txt"
        )
    if not action_fns:
        warnings.warn(f"No text annotations found in {root_dir}")
        return

    for action_txt_fn in action_fns:
        track_id = 0
        video_dir = os.path.dirname(action_txt_fn)
        video_name = os.path.basename(video_dir)
        if video_name in KNOWN_BAD_VIDEOS:
            continue

        action_f = open(action_txt_fn)
        lines = action_f.readlines()

        # Create output csv
        csv_fn = f"{output_dir}/{video_name}_activity_labels_v2.csv"
        csv_f = open(csv_fn, "w")
        csv_f.write(
            "# 1: Detection or Track-id,2: Video or Image Identifier,3: Unique Frame Identifier,4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),8: Detection or Length Confidence,9: Target Length (0 or -1 if invalid),10-11+: Repeated Species,Confidence Pairs or Attributes\n"
        )
        csv_f.write('# metadata,fps: 1,"exported_by: ""dive:typescript"""\n')

        for line in lines:
            data = line.split("\t")

            # Find frame filenames
            start_frame = int(data[0])
            end_frame = int(data[1])

            start_frame_fn = os.path.basename(
                glob.glob(f"{video_dir}/images/frame_{start_frame}_*.png")[0]
            )
            end_frame_fn = os.path.basename(
                glob.glob(f"{video_dir}/images/frame_{end_frame}_*.png")[0]
            )

            # Determine activity
            activity_str = data[2].strip().split(" ")
            hand = activity_str[0]
            activity = activity_str[1]
            target = activity_str[2] if len(activity_str) > 2 else None

            # convert activity_str info to our activity labels
            # this is hacky: fix later
            label = None
            label, label_id = activity_label_fixes(activity_label, target)

            if label is not None:
                line1 = f"{track_id},{start_frame_fn},{start_frame},1,1,2,2,1,-1,{label_id},1"
                csv_f.write(f"{line1}\n")
                line2 = (
                    f"{track_id},{end_frame_fn},{end_frame},1,1,2,2,1,-1,{label_id},1"
                )
                csv_f.write(f"{line2}\n")

                track_id += 1
        action_f.close()
        csv_f.close()


def find_bad_images(imgs, good_imgs, output_dir):
    good_image_fns = [os.path.basename(f) for f in glob.glob(f"{good_imgs}/*")]
    img_fns = [os.path.basename(f) for f in glob.glob(f"{imgs}/*")]

    print(len(img_fns))
    print(len(good_image_fns))

    bad_img_fns = [f for f in img_fns if f not in good_image_fns]

    print(len(bad_img_fns))

    for fn in bad_img_fns:
        shutil.copy(f"{imgs}/{fn}", output_dir)
