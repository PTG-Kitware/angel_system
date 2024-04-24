import os
import kwcoco
import glob
import warnings
import re
import ubelt as ub

from angel_system.data.common.load_data import Re_order
from angel_system.data.common.kwcoco_utils import load_kwcoco



RE_FILENAME_TIME = re.compile(
    r"frame_(?P<frame>\d+)_(?P<ts>\d+(?:_|.)\d+).(?P<ext>\w+)"
)


def time_from_name(fname):
    """
    Extract the float timestamp from the filename.

    :param fname: Filename of an image in the format
        frame_<frame number>_<seconds>_<nanoseconds>.<extension>

    :return: timestamp (float) in seconds
    """
    fname = os.path.basename(fname)
    match = RE_FILENAME_TIME.match(fname)
    time = match.group("ts")
    if "_" in time:
        time = time.split("_")
        time = float(time[0]) + (float(time[1]) * 1e-9)
    elif "." in time:
        time = float(time)

    frame = match.group("frame")
    return int(frame), time

def object_label_fixes(obj_cat):
    # Fix some deprecated labels
    if obj_cat in ["timer", "timer (20)", "timer (30)", "timer (else)"]:
        obj_cat = "timer (on)"
    if obj_cat in ["kettle"]:
        obj_cat = "kettle (closed)"
    if obj_cat in ["water"]:
        obj_cat = "water jug (open)"
    if obj_cat in ["grinder (close)"]:
        obj_cat = "grinder (closed)"
    if obj_cat in ["thermometer (close)"]:
        obj_cat = "thermometer (closed)"
    if obj_cat in ["switch"]:
        obj_cat = "kettle switch"
    if obj_cat in ["lid (kettle)"]:
        obj_cat = "kettle lid"
    if obj_cat in ["lid (grinder)"]:
        obj_cat = "grinder lid"
    if obj_cat in ["coffee grounds + paper filter + filter cone"]:
        obj_cat = "coffee grounds + paper filter (quarter - open) + dripper"
    if obj_cat in ["coffee grounds + paper filter + filter cone + mug"]:
        obj_cat = "coffee grounds + paper filter (quarter - open) + dripper + mug"
    if obj_cat in ["paper filter + filter cone"]:
        obj_cat = "paper filter (quarter) + dripper"
    if obj_cat in ["paper filter + filter cone + mug"]:
        obj_cat = "paper filter (quarter) + dripper + mug"
    if obj_cat in ["used paper filter + filter cone"]:
        obj_cat = "used paper filter (quarter - open) + dripper"
    if obj_cat in ["used paper filter + filter cone + mug"]:
        obj_cat = "used paper filter (quarter) + dripper + mug"
    if obj_cat in ["filter cone"]:
        obj_cat = "dripper"
    if obj_cat in ["filter cone + mug"]:
        obj_cat = "dripper + mug"
    if obj_cat in ["water + coffee grounds + paper filter + filter cone + mug"]:
        obj_cat = "used paper filter (quarter - open) + dripper + mug"

    return obj_cat

def activity_label_fixes(activity_label):
    # Temp fix until we can update the groundtruth labels
    if activity_label in ["microwave-30-sec", "microwave-60-sec"]:
        activity_label = "microwave"
    if activity_label in ["stir-again"]:
        activity_label = "oatmeal-stir"
    if activity_label in [
        "measure-half-cup-water",
        "measure-12oz-water",
    ]:
        activity_label = "measure-water"
    if activity_label in ["insert-toothpick-1", "insert-toothpick-2"]:
        activity_label = "insert-toothpick"
    if activity_label in ["slice-tortilla", "continue-slicing"]:
        activity_label = "floss-slice-tortilla"
    if activity_label in ["steep", "check-thermometer"]:
        activity_label = "background"
    if activity_label in ["dq-clean-knife", "pinwheel-clean-knife"]:
        activity_label = "clean-knife"
    if activity_label in ["zero-scale", "scale-turn-on"]:
        activity_label = "scale-press-btn"
    if activity_label in ["pour-water-grounds-wet"]:
        activity_label = "pour-water-grounds-circular"

    return activity_label


def reorder_images(dset):
    # Data paths
    data_dir = "/data/PTG/cooking/"
    ros_bags_dir = f"{data_dir}/ros_bags/"
    coffee_ros_bags_dir = f"{ros_bags_dir}/coffee/coffee_extracted/"
    tea_ros_bags_dir = f"{ros_bags_dir}/tea/tea_extracted/"

    # Load kwcoco file
    dset = load_kwcoco(dset)
    gid_to_aids = dset.index.gid_to_aids

    new_dset = kwcoco.CocoDataset()

    new_dset.dataset["info"] = dset.dataset["info"].copy()
    for cat_id, cat in dset.cats.items():
        new_dset.add_category(**cat)

    for video_id, video in dset.index.videos.items():
        # Add video to new dataset
        if "_extracted" in video["name"]:
            video["name"] = video["name"].split("_extracted")[0]
        new_dset.add_video(**video)

        # Find folder of images for video
        video_name = video["name"]
        if "tea" in video_name:
            images_dir = f"{tea_ros_bags_dir}/{video_name}_extracted/images"
        else:
            images_dir = f"{coffee_ros_bags_dir}/{video_name}_extracted/images"

        images = glob.glob(f"{images_dir}/*.png")
        if not images:
            warnings.warn(f"No images found in {video_name}")
        images = Re_order(images, len(images))

        for image in images:
            # Find image in old dataset
            image_lookup = dset.index.file_name_to_img
            old_img = image_lookup[image]

            new_im = old_img.copy()
            del new_im["id"]

            new_gid = new_dset.add_image(**new_im)

            # Add annotations for image
            old_aids = gid_to_aids[old_img["id"]]
            old_anns = ub.dict_subset(dset.anns, old_aids)

            for old_aid, old_ann in old_anns.items():
                new_ann = old_ann.copy()

                del new_ann["id"]
                new_ann["image_id"] = new_gid
                new_dset.add_annotation(**new_ann)

    new_dset.fpath = dset.fpath.split(".mscoco.json")[0] + "_reordered_imgs.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved dset to {new_dset.fpath}")
