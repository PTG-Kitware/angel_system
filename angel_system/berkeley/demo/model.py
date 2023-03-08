# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.smoothing_tracker_utils import MC50_CATEGORIES as MC50_CATEGORIES
from detectron2.utils.tracker import Tracker

from demo.predictor import VisualizationDemo, VisualizationDemo_add_smoothing

# constants
WINDOW_NAME = "COCO detections"
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file."
                        # , default='/shared/niudt/detectron2/images/Videos/k2/4.MP4'
                        )
    # parser.add_argument(
    #     "--input",
    #     nargs="+",
    #     help="A list of space separated input images; "
    #     "or a single glob pattern such as 'directory/*.jpg'"
    #     , default= ['/shared/niudt/DATASET/PTG_Kitware/all_activities_6/images/*.png']
    # )

    # time_prefix = time.strftime("%Y-%m-%d %X").split(' ')[0]
    # save_dir = os.path.join('/shared/niudt/detectron2/DEMO_Results', time_prefix, 'MC_6/subaa_')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # parser.add_argument(
    #     "--output",
    #     help="A file or directory to save output visualizations. "
    #     "If not given, will show output in an OpenCV window."
    #     #, default='/shared/niudt/detectron2/test_results/kitchen305.jpg'
    #     , default=save_dir
    # )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False

def Re_order(image_list, image_number):
    img_id_list = []
    for img in image_list:
        img_id = int(img.split('/')[-1].split('_')[1])
        img_id_list.append(img_id)
    img_id_arr = np.array(img_id_list)
    s = np.argsort(img_id_arr)
    new_list = []
    for i in range(image_number):
        idx = s[i]
        new_list.append(image_list[idx])
    return new_list

def decode_prediction(predictions):
    s = 1

    if not predictions == None:
        pres = {}

        for i, instance_cls in enumerate(predictions[1]):
            pre = {}
            x = len(instance_cls.split(' ')[-1])
            cls_name = instance_cls[:-(x + 1)]
            # pre['category'] = cls_name
            pre['confidence_score'] = float(instance_cls.split(' ')[-1][:-1]) * 0.01
            pre['bbox'] = predictions[0][i]

            if predictions[2][i] == 1:
                pre['obj_obj_contact_sate'] = True
            else:
                pre['obj_obj_contact_sate'] = False

            if predictions[4][i] == 1:
                pre['obj_hand_contact_sate'] = True
            else:
                pre['obj_hand_contact_sate'] = False
            pres[cls_name] = pre

        return pres
    else:
        return None




def inference(frames):
    # transfer th RGB format to BGR format
    frames = frames[...,[2, 1, 0]]

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)



    demo = VisualizationDemo_add_smoothing(cfg, number_frames=len(frames), last_time=2, fps = 30)

    idx = 0

    preds = {}
    visualized_outputs = []
    for frame in frames:
        idx = idx + 1

        predictions, visualized_output = demo.run_on_image_smoothing_v2(frame, current_idx=idx)
        visualized_outputs.append(visualized_output)

        if decode_prediction(predictions) != None:
            preds[idx] = decode_prediction(predictions)

    return preds, visualized_outputs


