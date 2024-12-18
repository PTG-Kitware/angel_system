#!/usr/bin/env python3

from collections import defaultdict
import logging
from pathlib import Path
import random
from typing import Dict
from typing import Optional
from typing import Sequence
import warnings

import click
import cv2
import kwcoco
import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch
import ubelt as ub
from ultralytics import YOLO

from angel_system.object_detection.yolov8_detect import predict_hands


LOG = logging.getLogger(__name__)


def plot_one_box(xywh, img, color=None, label=None, line_thickness=1) -> None:
    """
    Plot one detection box into the given image with CV2.

    Based on the similar function from YOLO v7 plotting code.

    :param xywh: Extent of the box to plot in xywh format, where the xy is the
        upper-left coordinate of the box.
    :param img: is the image matrix to draw the box into.
    :param color: Optional RGB value tuple of the color to draw.
    :param label: Optional text label to draw for the box.
    :param line_thickness: Thickness of the box lines to draw.
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xywh[0]), int(xywh[1])), (int(xywh[0] + xywh[2]), int(xywh[1] + xywh[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


@click.command()
@click.help_option("-h", "--help")
@click.option(
    "-i", "--input-coco-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "MS-COCO file specifying image files to perform object detection over. "
        "Image and Video sections from this COCO file will be maintained in the "
        "output COCO file."
    ),
    required=True,
)
@click.option(
    "--img-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional override for the input COCO dataset bundle root. This is "
        "necessary when the input COCO file uses relative paths and the COCO "
        "file itself is not located in the bundle root directory."
    ),
)
@click.option(
    "-o", "--output-coco-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output COCO file to write object detection results.",
    required=True,
)
@click.option(
    "--model-hands", "hand_model_ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint for the Yolo v8 hand detector.",
    required=True,
)
@click.option(
    "--model-objects", "objs_model_ckpt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Model checkpoint for the Yolo v7 object detector.",
    required=True,
)
@click.option(
    "-e", "--exclude-obj-class",
    "obj_exclude_classes",
    multiple=True,
    help=(
        "Exclude these object classes from the class list provided by the "
        "object model. This is for when the object model was trained with "
        "some classes excluded, but YOLO provided the metadata for them "
        "anyway."
    )
)
@click.option(
    "--model-device",
    default="",
    help="The CUDA device to use, i.e. '0' or '0,1,2,3' or 'cpu'."
)
@click.option(
    "--obj-img-size",
    type=int,
    default=640,
    help=(
        "Data input size for the detection models for objects. This should be "
        "a multiple of the model's stride parameter."
    )
)
@click.option(
    "--hand-img-size",
    type=int,
    default=768,
    help=(
        "Data input size for the detection model for hands. This should be a "
        "multiple of the model's stride parameter."
    )
)
@click.option(
    "--conf-thresh",
    type=float,
    default=0.25,
    help=(
        "Object confidence threshold. Predicted objects with confidence less "
        "than this will not be considered for output."
    ),
)
@click.option(
    "--iou-thresh",
    type=float,
    default=0.45,
    help=(
        "IoU threshold used during NMS to filter out overlapping bounding "
        "boxes."
    ),
)
@click.option(
    "--tensorrt",
    is_flag=True,
    help=(
        "Export object and hand models to TensorRT and use FP16 to "
        "accelerate prediction performance."
    ),
)
@click.option(
    "--save-img", "save_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Optionally enable the plotting of detections back to the image and "
        "saving them out to disk, rooted in this directory. Only detections "
        "with confidence above our configured threshold will be considered "
        "for plotting."
    )
)
@click.option(
    "--top-k", "save_top_k",
    type=int,
    default=None,
    help=(
        "Optionally specify that only the top N confidence detections should "
        "be saved to the output images. If this is not provided, all "
        "detections with confidence above the --conf-thres value will be "
        "plotted. This only applies to objects, not detected hands by that "
        "respective model."
    )
)
@click.option(
    "--save-vid",
    is_flag=True,
    help=(
        "Optionally enable the creation of an MP4 video from the images "
        "rendered due to --save-img. This option only has an effect if the "
        "--save-img option is provided. The video file will be save next to "
        "the directory into which component images are saved."
    )
)
@torch.inference_mode()
def yolo_v11_inference_objects(
    input_coco_file: Path,
    img_root: Optional[Path],
    output_coco_file: Path,
    hand_model_ckpt: Path,
    objs_model_ckpt: Path,
    obj_exclude_classes: Sequence[str],
    model_device: str,
    obj_img_size: int,
    hand_img_size: int,
    conf_thresh: float,
    iou_thresh: float,
    tensorrt: bool,
    save_dir: Optional[Path],
    save_top_k: Optional[int],
    save_vid: bool,
):
    """
    Script for use in generating object detection results based on an input
    COCO file's video/image specifications.

    Expected use-case: generate object detections for video frames (images)
    that we have activity classification truth for.

    \b
    Example:
        python3 yolo_v11_inference_objects \\
    """
    logging.basicConfig(
        level=logging.INFO,
    )

    guiding_dset = kwcoco.CocoDataset(input_coco_file, bundle_dpath=img_root)

    # Prevent overwriting an existing file. These are expensive to compute so
    # we don't want to mess that up.
    if output_coco_file.is_file():
        raise ValueError(
            f"Output COCO file already exists, refusing to overwrite: "
            f"{output_coco_file}"
        )
    output_coco_file.parent.mkdir(parents=True, exist_ok=True)
    dset = kwcoco.CocoDataset()
    dset.fpath = output_coco_file.as_posix()

    object_model = YOLO(objs_model_ckpt, task="detect")
    LOG.info(
        "Loaded object model with classes:\n"
        + "\n".join(f'\t- [{n[0]}] "{n[1]}"' for n in object_model.names.items())
    )
    hand_model = YOLO(hand_model_ckpt, task="detect")
    LOG.info(
        "Loaded hand model with classes:\n"
        + "\n".join(f'\t- [{n[0]}] "{n[1]}"' for n in hand_model.names.items())
    )

    if tensorrt:
        LOG.info("Exporting object model to TensorRT")
        om_trt_path = object_model.export(
            format="engine",
            imgsz=obj_img_size,
            device=model_device,
            half=True,  # this was taking a long time to process?
            nms=True,
        )
        LOG.info("Loading TensorRT object model")
        object_model = YOLO(om_trt_path, task="detect")
        LOG.info("Exporting hand model to TensorRT")
        hm_trt_path = hand_model.export(
            format="engine",
            imgsz=hand_img_size,
            device=model_device,
            half=True,  # this was taking a long time to process?
            nms=True,
        )
        LOG.info("Loading TensorRT hand model")
        hand_model = YOLO(hm_trt_path, task="detect")

    model_half = model_device.lower() != "cpu"

    cls_names = [p[1] for p in sorted(object_model.names.items())]
    cls_colors = [[random.randint(0, 255) for _ in range(3)] for _ in cls_names]

    # Port over the videos and images sections from the input dataset to the
    # new one.
    dset.dataset['videos'] = guiding_dset.dataset['videos']
    dset.dataset['images'] = guiding_dset.dataset['images']
    dset.index.build(dset)
    # Equality can later be tested with:
    #   guiding_dset.index.videos == dset.index.videos
    #   guiding_dset.index.imgs == dset.index.imgs

    # Add categories
    for cls_name in obj_exclude_classes:
        if cls_name not in cls_names:
            warnings.warn(
                f"Requested exclusion of object class named \"{cls_name}\", "
                f"however this class is not present in the object model."
            )
    exclude_set = set(obj_exclude_classes)
    for i, object_label in enumerate(cls_names):
        if object_label not in exclude_set:
            dset.ensure_category(name=object_label, id=i)
        else:
            LOG.info(f"Excluding object model class: \"{object_label}\"")
    # Inject categories for the hand-model additions.
    left_hand_cid = dset.ensure_category(name="hand (left)")
    right_hand_cid = dset.ensure_category(name="hand (right)")
    hands_cat_to_cid = {"hand (left)": left_hand_cid,
                        "hand (right)": right_hand_cid}

    # model warm-up going into the prediction loop
    LOG.info("Warming up models...")
    warmup_image = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    object_model(source=warmup_image, device=model_device, half=model_half, verbose=False)
    hand_model(source=warmup_image, device=model_device, half=model_half, verbose=False)
    LOG.info("Warming up models... Done")

    # -------------------------------------------------------------------------
    # Generate object/hand predictions

    # Mapping of video_id to the filepath for which component frames have been
    # written out to.
    video_image_outputs: Dict[int, Dict[int, str]] = defaultdict(dict)
    # Dictionary of image output directories per video ID
    video_id_to_frame_dir: Dict[int, Path] = dict()

    object_predict_kwargs = dict(
        conf=conf_thresh,
        device=model_device,
        nms=True,
        half=model_half,
        verbose=False,
    )
    if obj_img_size is not None:
        object_predict_kwargs["imgsz"] = obj_img_size

    hand_predict_kwargs = dict(
        hand_model=hand_model,
        device=model_device,
        nms=True,
        half=model_half,
    )
    if hand_img_size is not None:
        hand_predict_kwargs["imgsz"] = hand_img_size

    for img_id in ub.ProgIter(
        dset.images(),
        desc="Processing Images",
        verbose=3,
    ):
        img_path: Path = dset.get_image_fpath(img_id)
        img0 = cv2.imread(img_path.as_posix())
        if img0 is None:
            raise RuntimeError(f"Failed to read image file: {img_path}")

        # returns list of length=num images, which is always 1 here.
        object_preds = object_model.predict(source=img0, **object_predict_kwargs)[0]

        hand_boxes, hand_labels, hand_confs = predict_hands(
            img0=img0,
            **hand_predict_kwargs,
        )

        # YOLO xywh output defines the xy as the cetner point, not the
        # upper-left as required by the COCO format, thus take the xyxy output
        # and subtracting out the upper-left.
        obj_box_xywh = object_preds.boxes.xyxy.cpu()
        obj_box_xywh[:, 2:] -= obj_box_xywh[:, :2]
        obj_box_areas = torch.multiply(obj_box_xywh[:, 2], obj_box_xywh[:, 3])
        for box_xywh, box_cls, box_conf, box_area in zip(
            obj_box_xywh.tolist(),
            object_preds.boxes.cls.to(int).tolist(),
            object_preds.boxes.conf.tolist(),
            obj_box_areas.tolist(),
        ):
            dset.add_annotation(
                image_id=img_id,
                category_id=box_cls,
                bbox=box_xywh,
                score=box_conf,
                area=box_area,
            )
            if save_dir is not None:
                plot_one_box(
                    box_xywh,
                    img0,
                    color=cls_colors[box_cls],
                    label=f"{cls_names[box_cls]} {box_conf:.2f}",
                )

        # Convert hand box XYXY coordinates into XYWH where XY is the
        # upper-left.
        hand_boxes_xywh = np.asarray(hand_boxes).reshape(-1, 4)
        hand_boxes_xywh[:, 2:] -= hand_boxes_xywh[:, :2]
        hand_areas = np.multiply(hand_boxes_xywh[:, 2], hand_boxes_xywh[:, 3])
        for box_xywh, box_lbl, box_conf, box_area in zip(
            hand_boxes_xywh.tolist(),
            hand_labels,
            hand_confs,
            hand_areas,
        ):
            box_cls = hands_cat_to_cid[box_lbl]
            dset.add_annotation(
                image_id=img_id,
                category_id=box_cls,
                bbox=box_xywh,
                score=box_conf,
                area=box_area,
            )
            if save_dir is not None:
                plot_one_box(
                    box_xywh,
                    img0,
                    color=[0, 0, 0],
                    label=f"{box_lbl} {box_conf:.2f}",
                )

        # Optionally draw object detection results to an image.
        # If we want to save as a video, also save the paths so we can create
        # the video after detecting everything.
        if save_dir is not None:
            vid_id = dset.index.imgs[img_id]["video_id"]
            if vid_id not in video_id_to_frame_dir:
                vid_obj = dset.index.videos[vid_id]
                save_imgs_dir = save_dir / Path(vid_obj["name"]).stem
                save_imgs_dir.mkdir(parents=True, exist_ok=True)
                video_id_to_frame_dir[vid_id] = save_imgs_dir
            save_path = (video_id_to_frame_dir[vid_id] / img_path.name).as_posix()
            if not cv2.imwrite(save_path, img0):
                raise RuntimeError(f"Failed to write debug image: {save_path}")
            img_obj = dset.index.imgs[img_id]
            video_image_outputs[vid_id][img_obj["frame_index"]] = save_path

    # If configured, create and save videos of debug images for each video
    # effectively processed.
    if save_dir and save_vid:
        for vid_id, frame_set in ub.ProgIter(
            video_image_outputs.items(),
            desc="Creating Videos",
            verbose=3,
        ):
            frame_set: Dict[int, str]
            vid_obj = dset.index.videos[vid_id]
            video_save_path = save_dir / f"{Path(vid_obj['name']).stem}-objects.mp4"
            vid_frames = [p[0] for p in sorted(frame_set.items())]
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(
                vid_frames,
                fps=vid_obj["framerate"]
            )
            clip.write_videofile(video_save_path.as_posix())
            LOG.info(f"Saved video to: {video_save_path}")

    LOG.info(f"Saving output COCO file... ({output_coco_file})")
    dset.dump(dset.fpath, newlines=True)
    LOG.info(f"Saved output COCO file: {output_coco_file}")


if __name__ == "__main__":
    yolo_v11_inference_objects()
