import os
import cv2
import yaml
import glob
import kwcoco
import kwimage
import textwrap
import warnings
import random
import matplotlib
import shutil

import numpy as np
import ubelt as ub
import pandas as pd
import ubelt as ub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from sklearn.preprocessing import normalize

from pathlib import Path
from PIL import Image

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    sanitize_str,
)
from angel_system.data.common.load_data import Re_order


def load_kwcoco(dset):
    """Load a kwcoco dataset from file

    :param dset: kwcoco object or a string pointing to a kwcoco file

    :return: The loaded kwcoco object
    :rtype: kwcoco.CocoDataset
    """
    # Load kwcoco file
    if type(dset) == str:
        dset_fn = dset
        dset = kwcoco.CocoDataset(dset_fn)
        dset.fpath = dset_fn
        print(f"Loaded dset from file: {dset_fn}")
    return dset


def add_activity_gt_to_kwcoco(topic, task, dset):
    """Takes an existing kwcoco file and fills in the "activity_gt"
    field on each image based on the activity annotations.

    This saves to a new file (the original kwcoco file name with "_fixed"
    appended to the end).

    :param dset: kwcoco object or a string pointing to a kwcoco file
    """
    if topic == "medical":
        from angel_system.data.medical.load_bbn_data import time_from_name
    elif topic == "cooking": 
        from angel_system.data.cooking.load_kitware_data import time_from_name
    
    # Load kwcoco file
    dset = load_kwcoco(dset)

    data_dir = f"/data/PTG/{topic}/"
    activity_gt_dir = f"{data_dir}/activity_anns"

    # Load activity config
    with open(
        f"config/activity_labels/{topic}/{task}.yaml", "r"
    ) as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]
    label_version = activity_config["version"]

    activity_gt_dir = f"{activity_gt_dir}/{task}_labels/"

    # Add ground truth to kwcoco
    for video_id in dset.index.videos.keys():
        video = dset.index.videos[video_id]
        video_name = video["name"]
        print(video_name)

        if "_extracted" in video_name:
            video_name = video_name.split("_extracted")[0]

        activity_gt_fn = f"{activity_gt_dir}/{video_name}_activity_labels_v{label_version}.csv"
        gt = activities_from_dive_csv(topic, activity_gt_fn)
        gt = objs_as_dataframe(gt)

        image_ids = dset.index.vidid_to_gids[video_id]

        # Update the activity gt for each image
        for gid in sorted(image_ids):
            im = dset.imgs[gid]
            frame_idx, time = time_from_name(im["file_name"])

            if time:
                matching_gt = gt.loc[(gt["start"] <= time) & (gt["end"] >= time)]
            else:
                matching_gt = gt.loc[(gt["start_frame"] <= frame_idx) & (gt["end_frame"] >= frame_idx)]

            if matching_gt.empty:
                label = "background"
                activity_label = label
            else:
                label = matching_gt.iloc[0]["class_label"]
                if type(label) == float or type(label) == int:
                    label = int(label)
                label = str(label)

                try:
                    activity = [
                        x
                        for x in activity_labels
                        if int(x["id"]) == int(float(label))
                    ]
                except:
                    activity = []

                if not activity:
                    if "timer" in label:
                        # Ignoring timer based labels
                        label = "background"
                        activity_label = label
                    else:
                        warnings.warn(
                            f"Label: {label} is not in the activity labels config, ignoring"
                        )
                        print(f"LABEL: {label}, {type(label)}")
                        continue
                else:
                    activity = activity[0]
                    activity_label = activity["label"]

            dset.imgs[gid]["activity_gt"] = activity_label

    # dset.fpath = dset.fpath.split(".")[0] + "_fixed.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    return dset


def visualize_kwcoco_by_label(dset=None, save_dir=""):
    """Draw the bounding boxes from the kwcoco file on
    the associated images

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param save_dir: Directory to save the images to
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    dset = load_kwcoco(dset)

    colors = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colors)

    obj_labels = [v["name"] for k, v in dset.cats.items()]

    empty_ims = 0

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im.get("video_id", None)

        fn = im["file_name"].split("/")[-1]
        gt = im.get("activity_gt", "")
        if not gt:
            gt = ""
        # act_pred = im.get("activity_pred", "")

        fig, ax = plt.subplots()
        # title = f"GT: {gt}, PRED: {act_pred}"
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = Image.open(im["file_name"])
        # image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)

        ax.imshow(image)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann.get("confidence", 1)
            # if conf < 0.1:
            #    continue

            x, y, w, h = ann["bbox"]  # xywh
            cat_id = ann["category_id"]
            cat = dset.cats[cat_id]["name"]

            label = f"{cat}: {round(conf, 2)}"

            color = colors[obj_labels.index(cat)]

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                clip_on=False,
            )

            ax.add_patch(rect)
            ax.annotate(label, (x, y), color="black", annotation_clip=False)

        video_dir = (
            f"{save_dir}/video_{img_video_id}/images/"
            if img_video_id is not None
            else f"{save_dir}/images/"
        )
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it

    plt.close("all")


def imgs_to_video(imgs_dir):
    """Convert directory of images to a video"""
    video_name = imgs_dir.split("/")[-1] + ".avi"

    images = glob.glob(f"{imgs_dir}/*.png")
    images = sorted(images, key=lambda x: time_from_name(x)[0])

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f"{imgs_dir}/{video_name}", 0, 15, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


def filter_kwcoco_by_conf(dset, conf_thr=0.4):
    """Filter the kwcoco dataset by confidence

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param conf_thr: Minimum confidence to be left in the dataset
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_anns = []
    for gid in sorted(gids):
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann["confidence"]

            if conf < conf_thr:
                remove_anns.append(aid)

    print(f"removing {len(remove_anns)} annotations")
    dset.remove_annotations(remove_anns)

    fpath = dset.fpath.split(".mscoco")[0]
    dset.fpath = f"{fpath}_conf_{conf_thr}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def background_images_dset(background_imgs):
    """Add images without annotations to a kwcoco dataset"""
    # Load kwcoco file
    import shutil

    dset = kwcoco.CocoDataset()

    for video in sorted(glob.glob(f"{background_imgs}/*/")):
        video_name = os.path.basename(os.path.normpath(video))
        video_lookup = dset.index.name_to_video
        if video_name in video_lookup:
            vid = video_lookup[video_name]["id"]
        else:
            vid = dset.add_video(name=video_name)

        for im in ub.ProgIter(glob.glob(f"{video}/*.png"), desc="Adding images"):
            new_dir = f"{video}/../"
            fn = os.path.basename(im)
            frame_num, time = time_from_name(fn)
            shutil.copy(im, new_dir)

            image = Image.open(im)
            w, h = image.size

            new_im = {
                "width": w,
                "height": h,
                "file_name": os.path.abspath(f"{new_dir}/{fn}"),
                "video_id": vid,
                "frame_index": frame_num,
            }
            new_gid = dset.add_image(**new_im)

    print(f"Images: {len(dset.imgs)}")
    dset.fpath = f"{background_imgs}/bkgd.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def dive_csv_to_kwcoco(dive_folder, object_config_fn, data_dir, dst_dir, output_dir=""):
    """Convert object annotations in DIVE csv file(s) to a kwcoco file

    :param dive_folder: Path to the csv files
    :param object_config_fn: Path to the object label config file
    """
    import shutil

    dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    title = object_config["title"].lower()
    dset.dataset["info"].append({f"{title}_object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        if object_label["label"] == "background":
            continue
        dset.add_category(name=object_label["label"], id=object_label["id"])

    # Add boxes
    for csv_file in ub.ProgIter(
        glob.glob(f"{dive_folder}/*.csv"), desc="Loading video annotations"
    ):
        print(csv_file)
        video_name = os.path.basename(csv_file).split("_object_labels")[0]

        video_lookup = dset.index.name_to_video
        if video_name in video_lookup:
            vid = video_lookup[video_name]["id"]
        else:
            vid = dset.add_video(name=video_name)

        dive_df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file}")
        for i, row in dive_df.iterrows():
            if i == 0:
                continue

            frame = row["2: Video or Image Identifier"]
            im_fp = f"{data_dir}/{video_name}_extracted/images/{frame}"

            frame_fn = f"{dst_dir}/{frame}"
            if not os.path.isfile(frame_fn):
                shutil.copy(im_fp, dst_dir)

            # Temp for coffee
            splits = frame.split("-")
            frame_num, time = time_from_name(splits[-1])

            image_lookup = dset.index.file_name_to_img
            if frame_fn in image_lookup:
                img_id = image_lookup[frame_fn]["id"]
            else:
                img_id = dset.add_image(
                    file_name=frame_fn,
                    video_id=vid,
                    frame_index=frame_num,
                    width=1280,
                    height=720,
                )

            bbox = (
                [
                    float(row["4-7: Img-bbox(TL_x"]),
                    float(row["TL_y"]),
                    float(row["BR_x"]),
                    float(row["BR_y)"]),
                ],
            )

            xywh = kwimage.Boxes([bbox], "tlbr").toformat("xywh").data[0][0].tolist()

            obj_id = row["10-11+: Repeated Species"]

            ann = {
                "area": xywh[2] * xywh[3],
                "image_id": img_id,
                "category_id": obj_id,
                "segmentation": [],
                "bbox": xywh,
            }

            dset.add_annotation(**ann)

    dset.fpath = f"{output_dir}/{title}_obj_annotations_v{label_ver}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def mixed_dive_csv_to_kwcoco(
    dive_folder, object_config_fn, data_dir, dst_dir, output_dir=""
):
    """Convert object annotations in DIVE csv file(s) to a kwcoco file,
    for the use case where there is one DIVE file for multiple videos

    :param dive_folder: Path to the csv files
    :param object_config_fn: Path to the object label config file
    """
    dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    title = object_config["title"].lower()
    dset.dataset["info"].append({f"{title}_object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        if object_label["label"] == "background":
            continue
        dset.add_category(name=object_label["label"], id=object_label["id"])

    # Add boxes
    for csv_file in ub.ProgIter(
        glob.glob(f"{dive_folder}/*.csv"), desc="Loading video annotations"
    ):
        print(csv_file)

        dive_df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file}")
        for i, row in dive_df.iterrows():
            if i == 0:
                continue

            frame = row["2: Video or Image Identifier"]
            img_fn = os.path.basename(frame)
            frame_num, time = time_from_name(img_fn)
            frame_fn = f"{dst_dir}/images/{img_fn}"
            assert os.path.isfile(frame_fn)

            # Attempt to find the original file
            original_file = glob.glob(f"{data_dir}/*/*/*/images/{img_fn}")
            # import pdb; pdb.set_trace()
            assert len(original_file) == 1
            original_folder = os.path.dirname(original_file[0])
            video_name = original_folder.split("/")[-2]
            if "_extracted" in video_name:
                video_name = video_name.split("_extracted")[0]
            print(video_name)

            video_lookup = dset.index.name_to_video
            if video_name in video_lookup:
                vid = video_lookup[video_name]["id"]
            else:
                vid = dset.add_video(name=video_name)

            image_lookup = dset.index.file_name_to_img
            if frame_fn in image_lookup:
                img_id = image_lookup[frame_fn]["id"]
            else:
                img_id = dset.add_image(
                    file_name=frame_fn,
                    video_id=vid,
                    frame_index=frame_num,
                    width=1280,
                    height=720,
                )

            bbox = (
                [
                    float(row["4-7: Img-bbox(TL_x"]),
                    float(row["TL_y"]),
                    float(row["BR_x"]),
                    float(row["BR_y)"]),
                ],
            )

            xywh = kwimage.Boxes([bbox], "tlbr").toformat("xywh").data[0][0].tolist()

            obj_id = row["10-11+: Repeated Species"]

            ann = {
                "area": xywh[2] * xywh[3],
                "image_id": img_id,
                "category_id": obj_id,
                "segmentation": [],
                "bbox": xywh,
            }

            dset.add_annotation(**ann)

    dset.fpath = f"{title}_obj_annotations_v{label_ver}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def update_obj_labels(dset, object_config_fn):
    """Change the object labels to match those provided
    in ``object_config_fn``
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    new_dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    new_dset.dataset["info"].append({"object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        new_dset.add_category(name=object_label["label"], id=object_label["id"])

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        # Add video
        if hasattr(im, "video_id"):
            old_video = dset.index.videos[im["video_id"]]["name"]
            new_video = new_dset.index.name_to_video[old_video]
            new_im["video_id"] = new_video["id"]

        del new_im["id"]
        new_im["file_name"] = im["file_name"]
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann["category_id"]]["name"]

            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()

            del new_ann["id"]
            new_ann["category_id"] = new_cat["id"]
            new_ann["image_id"] = new_gid

            new_dset.add_annotation(**new_ann)

    fpath = dset.fpath.split(".mscoco")[0]
    new_dset.fpath = f"{fpath}_new_obj_labels.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def remap_category_ids_demo(dset):
    """Adjust the category ids in a kwcoco dataset
    (From Jon Crall)
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    existing_cids = dset.categories().lookup("id")
    cid_mapping = {cid: cid - 1 for cid in existing_cids}

    for cat in dset.dataset["categories"]:
        old_cid = cat["id"]
        new_cid = cid_mapping[old_cid]
        cat["id"] = new_cid

    for ann in dset.dataset["annotations"]:
        old_cid = ann["category_id"]
        new_cid = cid_mapping[old_cid]
        ann["category_id"] = new_cid

    dset._build_index()
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")


def filter_kwcoco_by_class(dset, good_classes=[], bad_classes=[]):
    """Filter the kwcoco file to only include the labels in ``good_classes``
    and/or remove annotations with the labels ``bad_classes``
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    new_dset = kwcoco.CocoDataset()

    # Add categories
    classes = (
        good_classes
        if good_classes
        else [
            object_label["name"]
            for cat_id, object_label in dset.cats.items()
            if object_label["name"] not in bad_classes
        ]
    )
    print(classes)
    for object_label in classes:
        new_dset.add_category(name=object_label)

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        # Add video
        if hasattr(im, "video_id"):
            old_video = dset.index.videos[im["video_id"]]["name"]
            new_video = new_dset.index.name_to_video[old_video]
            new_im["video_id"] = new_video["id"]

        del new_im["id"]
        new_im["file_name"] = im["file_name"]
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann["category_id"]]["name"]
            if old_cat not in classes:
                continue

            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()

            del new_ann["id"]
            new_ann["category_id"] = new_cat["id"]
            new_ann["image_id"] = new_gid

            new_dset.add_annotation(**new_ann)

    # Remove any images without annotations now
    gid_to_aids = new_dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))
    remove_imgs = []
    for gid in sorted(gids):
        aids = gid_to_aids[gid]
        if len(aids) == 0:
            remove_imgs.append(gid)

    print(f"removing {len(remove_imgs)} images that no longer have annotations")
    new_dset.remove_images(remove_imgs)

    fpath = dset.fpath.split(".mscoco")[0]
    if good_classes:
        new_dset.fpath = f"{fpath}_{'_'.join(good_classes)}_only.mscoco.json"
    else:
        new_dset.fpath = f"{fpath}_without_{'_'.join(bad_classes)}.mscoco.json"

    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def filter_kwcoco_by_filename(dset, good_files_fn):
    """Filter the kwcoco dataset by filename

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param good_files_fn: Text file containing all good image filenames to keep
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    # Load good image names
    with open(good_files_fn, "r") as good_names_f:
        lines = good_names_f.readlines()
        good_names = [os.path.basename(l.strip()) for l in lines]
        print(good_names)

    remove_imgs = []
    for gid in sorted(gids):
        im = dset.imgs[gid]
        img_fn = im["file_name"]

        if os.path.basename(img_fn) not in good_names:
            remove_imgs.append(gid)

    print(f"removing {len(remove_imgs)} images")
    dset.remove_images(remove_imgs)

    fpath = dset.fpath.split(".mscoco")[0]
    dset.fpath = f"{fpath}_good_objects_only.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")
