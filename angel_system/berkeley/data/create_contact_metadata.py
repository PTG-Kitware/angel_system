"""
Load object detections and adds hand-object and object-object labels
based on the ground truth annotations.

This should be run on videos not used during training. 
"""
import os
import glob
import numpy as np
import pandas as pd
import pickle
import math
import pprint
import random

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from angel_system.data.common.kwcoco_utils import preds_to_kwcoco
from data.update_dets_utils import (
    load_hl_hand_bboxes,
    replace_compound_label,
    find_closest_hands,
    update_step_map,
    add_hl_hand_bbox
)
from data.run_obj_detector import (
    coffee_main,
    tourniquet_main,
    run_obj_detector
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def update_preds(
    activity_data_loader,
    preds,
    using_contact,
    original_step_map,
    step_map,
    data_root,
    data_dirs,
    experiment_flags,
):
    """
    Add the contact information back into the detections
    based on when the activities occurred and the contacts
    associated with the activities
    """
    activity_only_preds = {}
    for video_name in preds.keys():
        activity_only_preds[video_name] = {}

        # TODO: Be able to call activity_data_loader 
        # without task specific imports
        """
        gt_activity = activity_data_loader(
            videos_dir,
            video_name,
            original_step_map,
            lab_data,
            experiment_flags["using_inter_steps"],
            experiment_flags["using_before_finished_task"],
        )
        """
        gt_activity = activity_data_loader(
            video_name
        )

        if experiment_flags["add_background"]:
            step_map['background'] = [['background', []]] # Add background to loop

        for step, substeps in step_map.items():
            # print(step)

            for sub_step in substeps:
                sub_step_str = sub_step[0].lower().strip().strip(".").strip()
                # print(sub_step_str)
                objects = sub_step[1]

                matching_gts = (
                    gt_activity[sub_step_str]
                    if sub_step_str in gt_activity.keys()
                    else {}
                )
                # print('matching gt', matching_gts)

                matching_preds = {}
                for matching_gt in matching_gts:
                    matching_pred = {
                        ts: preds[video_name][ts]
                        for ts in preds[video_name].keys()
                        if matching_gt["start"] <= ts <= matching_gt["end"]
                    }
                    matching_preds.update(matching_pred)
                # print('matching preds', matching_preds)

                for frame in matching_preds.keys():
                    detected_classes = list(preds[video_name][frame].keys())
                    preds[video_name][frame]["meta"]["activity_gt"] = (
                        f"{sub_step_str} ({step})" if not inter_step else sub_step_str
                    )

                    if not experiment_flags["no_contact"]:
                        # Keep original detections
                        continue

                    if sub_step_str == "background":
                        # Remove any detections from background frames
                        for class_ in detected_classes:
                            if class_ != "meta":
                                del preds[video_name][frame][class_]

                        activity_only_preds[video_name][frame] = preds[video_name][
                            frame
                        ]
                        print("background frame ")
                        continue

                    found = []
                    for object_pair in objects:
                        # Determine if we found the objects relevant to the activity
                        found_items = 0

                        obj_hand_contact_state = (
                            True
                            if "hand" in object_pair[0].lower()
                            or "hand" in object_pair[1].lower()
                            else False
                        )
                        obj_obj_contact_state = not obj_hand_contact_state

                        # Update contact metadata
                        for obj in object_pair:
                            if obj == "hand":
                                hand_labels = find_closest_hands(
                                    object_pair,
                                    detected_classes,
                                    preds[video_name][frame],
                                )

                                if hand_labels is not None:
                                    found_items += 1
                                    if using_contact:
                                        for hand_label in hand_labels:
                                            for i in range(
                                                len(
                                                    preds[video_name][frame][hand_label]
                                                )
                                            ):
                                                preds[video_name][frame][hand_label][i][
                                                    "obj_hand_contact_state"
                                                ] = obj_hand_contact_state

                            elif obj in detected_classes:
                                found_items += 1
                                if using_contact:
                                    for i in range(len(preds[video_name][frame][obj])):
                                        preds[video_name][frame][obj][i][
                                            "obj_hand_contact_state"
                                        ] = obj_hand_contact_state
                                        preds[video_name][frame][obj][i][
                                            "obj_obj_contact_state"
                                        ] = obj_obj_contact_state

                            elif "+" in obj:
                                # We might be missing part of a compound label
                                # Let's try to fix that
                                (
                                    preds[video_name][frame],
                                    replaced,
                                ) = replace_compound_label(
                                    preds[video_name][frame],
                                    obj,
                                    detected_classes,
                                    using_contact,
                                    obj_hand_contact_state,
                                    obj_obj_contact_state,
                                )
                                detected_classes = list(preds[video_name][frame].keys())
                                if replaced is not None:
                                    found_items += 1

                        if found_items == 2:
                            # Only add frame if it has at least one full object pair
                            found.append(True)
                        else:
                            found.append(False)

                    detected_classes = list(preds[video_name][frame].keys())

                    if experiment_flags["filter_all_obj_frames"]:
                        if all(found):
                            print("Got all objects needed")
                            activity_only_preds[video_name][frame] = preds[video_name][
                                frame
                            ]
                    else:
                        activity_only_preds[video_name][frame] = preds[video_name][
                            frame
                        ]

    preds = activity_only_preds if experiment_flags["filter_activity_frames"] else preds
    print("Updated contact metadata in predictions")
    return preds

def main():
    experiment_name = "coffee_base"
    stage = "stage2"

    print("Experiment: ", experiment_name)
    print("Stage: ", stage)

    # Various experiment flags
    experiment_flags = {
        "no_contact": False if stage == "results" else True,
        "filter_activity_frames": False if stage == "results" else True,
        "filter_all_obj_frames": False if stage == "results" else True,
        "add_background": False if stage == "results" else True,
    }
    print(f"experiment flags: {experiment_flags}")

    (
        demo,
        training_split,
        data_root,
        data_dirs,
        activity_data_loader,
        metadata,
        step_map,
    ) = coffee_main(
        stage,
    )

    if stage == "stage2":
        splits = ['val', 'train_contact', 'test', 'train_activity']
    else:
        splits = ["test", "train_activity", "val"]

    if not os.path.exists("temp"):
        os.mkdir("temp")

    for split in splits:
        print(f"{split}: {len(training_split[split])} videos")

        # Raw detector output
        preds_no_contact, using_contact = run_obj_detector(
            demo,
            stage,
            data_root,
            data_dirs,
            training_split[split],
            no_contact=experiment_flags["no_contact"],
            ,
        )
        using_contact = True
        print(f"Using contact: {using_contact}")

        if add_hl_hands:
            preds_no_contact = add_hl_hand_bbox(preds_no_contact)

        fn = f"temp/{experiment_name}_{split}_preds_no_contact.pickle"
        print("temp file: ", fn)
        with open(fn, "wb") as fh:
            #preds_no_contact = pickle.load(fh)
            pickle.dump(preds_no_contact, fh)

        if stage != results:
            # Update contact info based on gt
            preds_with_contact = update_preds(
                activity_data_loader,
                preds_no_contact,
                using_contact,
                metadata["sub_steps"],
                step_map,
                data_root,
                data_dirs,
                experiment_flags,
            )
        else:
            preds_with_contact = preds_no_contact

        dset = preds_to_kwcoco(
            metadata,
            preds_with_contact,
            "",
            save_fn=f"{experiment_name}_{stage}_{split}.mscoco.json",
            # assuming detector already has the right labels so these aren't needed here
        )

    # TODO: train on save_fn + save model


if __name__ == "__main__":
    main()
