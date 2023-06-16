import os
import cv2
import kwcoco
import textwrap

import numpy as np
import ubelt as ub
import pandas as pd

from pathlib import Path


# Save
def preds_to_kwcoco(
    metadata,
    preds,
    save_dir,
    save_fn="result-with-contact.mscoco.json",
    using_step_labels=False,
    using_inter_steps=False,
    using_before_finished_task=False,
):
    """
    Save the predicitions in the json file
    format used by the detector training
    """
    import kwimage

    dset = kwcoco.CocoDataset()

    for class_ in metadata["thing_classes"]:
        if not using_step_labels and not using_inter_steps:
            # add original classes from model
            dset.add_category(name=class_)

        if using_step_labels:
            for i in range(1, 9):
                dset.add_category(name=f"{class_} (step {i})")

                if using_inter_steps:
                    if i != 8:
                        dset.add_category(name=f"{class_} (step {i+0.5})")
        if using_before_finished_task:
            dset.add_category(name=f"{class_} (before)")
            dset.add_category(name=f"{class_} (finished)")

    for video_name, predictions in preds.items():
        dset.add_video(name=video_name)
        vid = dset.index.name_to_video[video_name]["id"]

        for time_stamp in sorted(predictions.keys()):
            dets = predictions[time_stamp]
            fn = dets["meta"]["file_name"]

            activity_gt = (
                dets["meta"]["activity_gt"]
                if "activity_gt" in dets["meta"].keys()
                else None
            )

            dset.add_image(
                file_name=fn,
                video_id=vid,
                frame_index=dets["meta"]["frame_idx"],
                width=dets["meta"]["im_size"]["width"],
                height=dets["meta"]["im_size"]["height"],
                activity_gt=activity_gt,
            )
            img = dset.index.file_name_to_img[fn]

            del dets["meta"]

            for class_, det in dets.items():
                for i in range(len(det)):
                    cat = dset.index.name_to_cat[class_]

                    xywh = (
                        kwimage.Boxes([det[i]["bbox"]], "tlbr")
                        .toformat("xywh")
                        .data[0]
                        .tolist()
                    )

                    ann = {
                        "area": xywh[2] * xywh[3],
                        "image_id": img["id"],
                        "category_id": cat["id"],
                        "segmentation": [],
                        "bbox": xywh,
                        "confidence": det[i]["confidence_score"],
                    }

                    if "obj_obj_contact_state" in det[i].keys():
                        ann["obj-obj_contact_state"] = det[i]["obj_obj_contact_state"]
                        ann["obj-obj_contact_conf"] = det[i]["obj_obj_contact_conf"]
                    if "obj_hand_contact_state" in det[i].keys():
                        ann["obj-hand_contact_state"] = det[i]["obj_hand_contact_state"]
                        ann["obj-hand_contact_conf"] = det[i]["obj_hand_contact_conf"]

                    dset.add_annotation(**ann)

    dset.fpath = f"{save_dir}/{save_fn}" if save_dir != "" else save_fn
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")

    return dset


def print_class_freq(dset):
    freq_per_class = dset.category_annotation_frequency()
    stats = []

    for cat in dset.cats.values():
        freq = freq_per_class[cat["name"]]
        class_ = {
            "id": cat["id"],
            "name": cat["name"],
            #'instances_count': freq,
            #'def': '',
            #'synonyms': [],
            #'image_count': freq,
            #'frequency': '',
            #'synset': ''
        }

        stats.append(class_)

    print(f"MC50_CATEGORIES = {stats}")


def visualize_kwcoco(dset=None, save_dir=""):
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.patches as patches

    red_patch = patches.Patch(color="r", label="obj")
    green_patch = patches.Patch(color="g", label="obj-obj contact")
    blue_patch = patches.Patch(color="b", label="obj-hand contact")

    empty_ims = 0
    if type(dset) == str:
        print(f"Loaded dset from file: {dset}")
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im["video_id"]
        # if img_video_id == 3:
        #    continue

        fn = im["file_name"].split("/")[-1]
        gt = im["activity_gt"]  # if hasattr(im, 'activity_gt') else ''
        if not gt:
            gt = ""

        fig, ax = plt.subplots()
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = Image.open(im["file_name"])
        image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)

        ax.imshow(image)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann["confidence"]

            using_contact = (
                False  # True if 'obj-obj_contact_state' in ann.keys() else False
            )

            x, y, w, h = ann["bbox"]  # xywh
            cat = dset.cats[ann["category_id"]]["name"]
            if "tourniquet_tourniquet" in cat:
                tourniquet_im = image[int(y) : int(y + h), int(x) : int(x + w), ::-1]

                m2_fn = fn[:-4] + "_tourniquet_chip.png"
                m2_out = f"{save_dir}/video_{img_video_id}/images/chipped"
                Path(m2_out).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{m2_out}/{m2_fn}", tourniquet_im)

            label = f"{cat}: {round(conf, 2)}"

            color = "r"
            if using_contact and ann["obj-obj_contact_state"]:
                color = "g"
            if using_contact and ann["obj-hand_contact_state"]:
                color = "b"
            rect = patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor=color, facecolor="none"
            )

            ax.add_patch(rect)
            ax.annotate(label, (x, y), color="black")

        if using_contact:
            plt.legend(handles=[red_patch, green_patch, blue_patch], loc="lower left")

        video_dir = f"{save_dir}/video_{img_video_id}/images/"
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it
    plt.close("all")


def filter_kwcoco(dset, split):
    experiment_name = "m2_all_data_cleaned_fixed_with_steps"
    stage = "stage2"

    print("Experiment: ", experiment_name)
    print("Stage: ", stage)

    if type(dset) == str:
        print(f"Loaded dset from file: {dset}")
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    # Remove in-between categories
    remove_cats = []
    for cat_id in dset.cats:
        cat_name = dset.cats[cat_id]["name"]
        if ".5)" in cat_name or "(before)" in cat_name or "(finished)" in cat_name:
            remove_cats.append(cat_id)

    print(f"removing cat ids: {remove_cats}")
    dset.remove_categories(remove_cats)

    # Remove images with these
    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_images = []
    remove_anns = []
    for gid in sorted(gids):
        im = dset.imgs[gid]

        fn = im["file_name"].split("/")[-1]
        gt = im["activity_gt"]

        if gt == "not started" or "in between" in gt or gt == "finished":
            remove_images.append(gid)

        """
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann['confidence']
            if conf < 0.4:
                remove_anns.append(aid)
        """

    # print(f'removing {len(remove_anns)} annotations')
    # dset.remove_annotations(remove_anns)

    print(f"removing {len(remove_images)} images (and associated annotations)")
    dset.remove_images(remove_images)

    # Save to a new dataset to adjust ids
    new_dset = kwcoco.CocoDataset()
    new_cats = [
        {"id": 1, "name": "tourniquet_tourniquet (step 1)"},
        {"id": 2, "name": "tourniquet_tourniquet (step 2)"},
        {"id": 3, "name": "tourniquet_tourniquet (step 3)"},
        {"id": 4, "name": "tourniquet_tourniquet (step 4)"},
        {"id": 5, "name": "tourniquet_tourniquet (step 5)"},
        {"id": 6, "name": "tourniquet_tourniquet (step 6)"},
        {"id": 7, "name": "tourniquet_tourniquet (step 7)"},
        {"id": 8, "name": "tourniquet_tourniquet (step 8)"},
        {"id": 9, "name": "tourniquet_label (step 1)"},
        {"id": 10, "name": "tourniquet_label (step 2)"},
        {"id": 11, "name": "tourniquet_label (step 3)"},
        {"id": 12, "name": "tourniquet_label (step 4)"},
        {"id": 13, "name": "tourniquet_label (step 5)"},
        {"id": 14, "name": "tourniquet_label (step 6)"},
        {"id": 15, "name": "tourniquet_label (step 7)"},
        {"id": 16, "name": "tourniquet_label (step 8)"},
        {"id": 17, "name": "tourniquet_windlass (step 1)"},
        {"id": 18, "name": "tourniquet_windlass (step 2)"},
        {"id": 19, "name": "tourniquet_windlass (step 3)"},
        {"id": 20, "name": "tourniquet_windlass (step 4)"},
        {"id": 21, "name": "tourniquet_windlass (step 5)"},
        {"id": 22, "name": "tourniquet_windlass (step 6)"},
        {"id": 23, "name": "tourniquet_windlass (step 7)"},
        {"id": 24, "name": "tourniquet_windlass (step 8)"},
        {"id": 25, "name": "tourniquet_pen (step 1)"},
        {"id": 26, "name": "tourniquet_pen (step 2)"},
        {"id": 27, "name": "tourniquet_pen (step 3)"},
        {"id": 28, "name": "tourniquet_pen (step 4)"},
        {"id": 29, "name": "tourniquet_pen (step 5)"},
        {"id": 30, "name": "tourniquet_pen (step 6)"},
        {"id": 31, "name": "tourniquet_pen (step 7)"},
        {"id": 32, "name": "tourniquet_pen (step 8)"},
        {"id": 33, "name": "hand (step 1)"},
        {"id": 34, "name": "hand (step 2)"},
        {"id": 35, "name": "hand (step 3)"},
        {"id": 36, "name": "hand (step 4)"},
        {"id": 37, "name": "hand (step 5)"},
        {"id": 38, "name": "hand (step 6)"},
        {"id": 39, "name": "hand (step 7)"},
        {"id": 40, "name": "hand (step 8)"},
    ]
    for new_cat in new_cats:
        new_dset.add_category(name=new_cat["name"], id=new_cat["id"])

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        old_video = dset.index.videos[im["video_id"]]["name"]
        new_video = new_dset.index.name_to_video[old_video]

        del new_im["id"]
        new_im["video_id"] = new_video["id"]
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann["category_id"]]["name"]
            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()
            del new_ann["id"]
            new_ann["category_id"] = new_cat["id"]
            new_ann["image_id"] = new_gid

            new_dset.add_annotation(**new_ann)

    new_dset.fpath = f"{experiment_name}_{stage}_{split}.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def filter_kwcoco_conf_by_video(dset):
    if type(dset) == str:
        print(f"Loaded dset from file: {dset}")
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_anns = []
    for gid in sorted(gids):
        im = dset.imgs[gid]

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann["confidence"]
            video_id = im["video_id"]
            video = dset.index.videos[video_id]["name"]

            if "kitware" in video:
                continue
            else:
                # filter the BBN videos by 0.4 conf
                if conf < 0.4:
                    remove_anns.append(aid)

    print(f"removing {len(remove_anns)} annotations")
    dset.remove_annotations(remove_anns)

    dset.dump(dset.fpath, newlines=True)


def main():
    ptg_root = "/data/ptg/medical/bbn/"

    kw = "m2_all_data_cleaned_fixed_with_steps_results_train_activity.mscoco.json"

    n = kw[:-12].split("_")
    name = "_".join(n[:-1])
    split = n[-1]
    if split == "contact":
        split = "train_contact"
    if split == "activity":
        split = "train_activity"

    stage = "results"
    stage_dir = f"{ptg_root}/annotations/M2_Tourniquet/{stage}"
    exp = "m2_all_data_cleaned_fixed_with_steps"
    if stage == "stage1":
        save_dir = f"{stage_dir}/visualization_1/{split}"
    else:
        save_dir = f"{stage_dir}/{exp}/visualization/{split}"

    # save_dir = "visualization"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if stage == "stage1":
        visualize_kwcoco(f"{stage_dir}/{kw}", save_dir)
    else:
        visualize_kwcoco(f"{stage_dir}/{exp}/{kw}", save_dir)


if __name__ == "__main__":
    main()
