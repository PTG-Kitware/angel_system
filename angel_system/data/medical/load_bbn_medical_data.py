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
import pandas as pd
import numpy as np

from angel_system.data.common.load_data import activities_from_dive_csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

root_dir = "/data/PTG/medical/bbn_data/Release_v0.5/v0.52"


def dive_to_activity_file(videos_dir):
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
    add_inter_steps=False,
    add_before_finished_task=False,
):
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

    # Add in more time frames if applicable
    steps = list(step_map.keys()) if step_map else {}
    if add_inter_steps:
        print("Adding in-between steps")
        for i, step in enumerate(steps[:-1]):
            sub_step_str = step_map[step][0][0].lower().strip().strip(".").strip()
            next_sub_step_str = (
                step_map[steps[i + 1]][0][0].lower().strip().strip(".").strip()
            )
            if (
                sub_step_str in gt_activity.keys()
                and next_sub_step_str in gt_activity.keys()
            ):
                start = gt_activity[sub_step_str][0]["end"]
                end = gt_activity[next_sub_step_str][0]["start"]

                gt_activity[f"In between {step} and {steps[i+1]}".lower()] = [
                    {"start": start, "end": end}
                ]

    if add_before_finished_task:
        print("Adding before and finished")
        # before task
        sub_step_str = step_map["step 1"][0][0].lower().strip().strip(".").strip()
        if sub_step_str in gt_activity.keys():
            end = gt_activity[sub_step_str][0]["start"]  # when the first step starts
            gt_activity["not started"] = [{"start": 0, "end": end}]

        # after task
        sub_step_str = step_map["step 8"][0][0].lower().strip().strip(".").strip()
        if sub_step_str in gt_activity.keys():
            start = gt_activity[sub_step_str][0]["end"]  # when the last step ends
            end = len(glob.glob(f"{videos_dir}/{video}/_extracted/images/*.png")) - 1
            gt_activity["finished"] = [{"start": start, "end": end}]

    print(f"Loaded ground truth from {skill_fn}")

    return gt_activity


def bbn_medical_data_loader(
    skill, valid_classes="all", split="train", filter_repeated_objs=False
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
        print(ann_fn)

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


def data_loader(split, task_name):
    # Load gt bboxes for task
    task_classes, task_bboxes = bbn_medical_data_loader(task_name, split=split)

    # Combine task and person annotations
    # gt_bboxes = {**person_bboxes, **task_bboxes}
    # all_classes = person_classes + task_classes

    return task_classes, task_bboxes


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

    # print_class_freq(dset)

def bbn_activity_txt_to_csv():
    """
    Generate DIVE csv format activity annotations from BBN's text annotations
    """
    task = "M2_Tourniquet"
    print(f"{root_dir}/{task}/Data/*/*_action_labels_by_frame.txt")

    for action_txt_fn in glob.glob(f"{root_dir}/{task}/Data/*/*.action_labels_by_frame.txt"):
        track_id = 0
        video_dir = os.path.dirname(action_txt_fn)
        video_name = os.path.basename(video_dir)

        action_f = open(action_txt_fn)
        lines = action_f.readlines()

        # Create output csv
        task_dir = "m2_labels"
        csv_fn = f"{activity_dir}/{task_dir}/{video_name}_activity_labels_v2.csv"
        csv_f = open(csv_fn, "w")
        csv_f.write("# 1: Detection or Track-id,2: Video or Image Identifier,3: Unique Frame Identifier,4-7: Img-bbox(TL_x,TL_y,BR_x,BR_y),8: Detection or Length Confidence,9: Target Length (0 or -1 if invalid),10-11+: Repeated Species,Confidence Pairs or Attributes\n")
        csv_f.write('# metadata,fps: 1,"exported_by: ""dive:typescript"""\n')
        
        for line in lines:
            data = line.split("\t")

            # Find frame filenames
            start_frame = int(data[0])
            end_frame = int(data[1])

            start_frame_fn = os.path.basename(glob.glob(f"{video_dir}/images/frame_{start_frame}_*.png")[0])
            end_frame_fn = os.path.basename(glob.glob(f"{video_dir}/images/frame_{end_frame}_*.png")[0])

            # Determine activity
            activity_str = data[2].strip().split(" ")
            hand = activity_str[0]
            activity = activity_str[1]
            target = activity_str[2] if len(activity_str) > 2 else None

            # convert activity_str info to our activity labels
            # this is hacky: fix later
            label = None
            if activity == "put_tourniquet_around":
                label = "place-tourniquet"
                label_id = 1
            if activity == "pulls_tight":
                label = "pull-tight"
                label_id = 2
            if activity == "secures" and target == "velcro_strap":
                label = "apply-strap-to-strap-body"
                label_id = 3
            if activity == "twist" and target == "windlass":
                label = "turn-windless"
                label_id = 4
            if activity == "locks_into_windlass_keeper" or activity == "lock_into_windlass_keeper":
                label = "lock-windless"
                label_id = 5
            if activity == "wraps_remaining_strap_around" or activity == "wrap_remaining_strap_around":
                label = "pull-remaining-strap"
                label_id = 6
            if activity == "secures" and target == "windlass":
                label = "secure-strap"
                label_id = 7
            if activity == "writes_on" and target == "tourniquet_label":
                label = "mark-time"
                label_id = 8

            if label is not None:
                line1 = f"{track_id},{start_frame_fn},{start_frame},1,1,2,2,1,-1,{label_id},1"
                csv_f.write(f"{line1}\n")
                line2 = f"{track_id},{end_frame_fn},{end_frame},1,1,2,2,1,-1,{label_id},1"
                csv_f.write(f"{line2}\n")

                track_id += 1
        action_f.close()
        csv_f.close()

def main():
    # Should be M1 folder, M2 folder, etc
    subfolders = os.listdir(root_dir)
    for task_name in subfolders:
        for split in ["train", "test"]:
            classes, gt_bboxes = data_loader(split, task_name)

            out = f"{root_dir}/{task_name}/YoloModel/{task_name}_YoloModel_LO_{split}.mscoco.json"
            save_as_kwcoco(classes, gt_bboxes, save_fn=out)


if __name__ == "__main__":
    bbn_activity_txt_to_csv()
