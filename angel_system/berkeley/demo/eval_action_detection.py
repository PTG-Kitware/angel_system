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

from predictor import VisualizationDemo, VisualizationDemo_add_smoothing, VisualizationDemo_eval

# constants
WINDOW_NAME = "COCO detections"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        default="../configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        , default= ['/shared/niudt/DATASET/PTG_Kitware/all_activities_6/images/*.png']
    )

    time_prefix = time.strftime("%Y-%m-%d %X").split(' ')[0]
    save_dir = os.path.join('/shared/niudt/detectron2/DEMO_Results', time_prefix, 'eval')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window."
        # , default='/shared/niudt/detectron2/test_results/kitchen305.jpg'
        , default=save_dir
    )

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



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    for i in range(1, 51, 1):
        if i == 42:
            continue
        # for i in [6]:
        v_id = 'all_activities_' + str(i)  + '/images/*.png'
        args.input = [os.path.join('/shared/niudt/DATASET/PTG_Kitware/', v_id)]
        time_prefix = time.strftime("%Y-%m-%d %X").split(' ')[0]
        save_root = os.path.join('/shared/niudt/detectron2/DEMO_Results', time_prefix, 'eval')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_dir = os.path.join(save_root, 'pred_' + str(i) + '.csv' )
        if os.path.exists(save_dir):
            print("%s existes"%i)
            continue



        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
            # re-oder the input
            input_list = Re_order(args.input, len(args.input))
            # demo = VisualizationDemo_add_smoothing(cfg, number_frames=len(input_list), last_time=2, fps = 30)
            demo = VisualizationDemo_eval(cfg, number_frames=len(input_list), last_time=2, fps=30)

        idx = 0
        for path in tqdm.tqdm(input_list[idx:], disable=not args.output):
            idx = idx + 1
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            # step_info = demo.run_on_image_smoothing(img, current_idx=idx)
            demo.run_on_image_smoothing(img, current_idx=idx)
            step_info = demo.tracker.step_info
            # print('idx: ', idx)
            # print('step_info: ', step_info)

        #save results

        results = []
        for _step in step_info[1][1:]:
            # _ = []
            # _.append(_step['sub-step'])
            # _.append(_step['start_frame'])
            # _.append(_step['end_frame'])
            results.append([_step['sub-step'], _step['start_frame'], _step['end_frame']])
        import pandas as pd
        results = pd.DataFrame(results, columns=['sub-step', 'start_frame', 'end_frame'])
        results.to_csv(save_dir)



