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
from PIL import Image, ImageDraw, ImageFont

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# from .net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz

# constants
WINDOW_NAME = "COCO detections"

color_rgb = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


hand_rgb = [(0, 90, 181), (220, 50, 32)]
hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]

obj_rgb = (255, 194, 10)
obj_rgba = (255, 194, 10, 70)


side_map = {'l':'Left', 'r':'Right'}
side_map2 = {0:'Left', 1:'Right'}
side_map3 = {0:'L', 1:'R'}
state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}


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
        default="/shared/niudt/detectron2/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_fine-tuning.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    # parser.add_argument("--video-input", help="Path to video file."
    #                     , default='/shared/niudt/detectron2/images/Videos/k2/4.MP4'
    #                     )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        # "or a single glob pattern such as 'directory/*.jpg'",
        # default= ['/shared/niudt/detectron2/images/kitchen.jpeg']
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
        default='../test_results/result4_2_ft.mp4'
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
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


def get_hand_target_info(dir):
    import pickle
    import numpy as np

    # dir

    #读取文件中的内容。注意和通常读取数据的区别之处
    df=open(dir,'rb')#注意此处是rb
    #此处使用的是load(目标文件)
    data = pickle.load(df)



    df.close()
    # __background__', 'targetobject', 'hand' cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds, :],
    #                                           offset_vector.squeeze(0)[inds, :],
    #                                           lr[inds, :], nc_prob[inds, :]), 1)

    return data

def draw_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font):

    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(obj_bbox, outline=obj_rgb, width=4, fill=obj_rgba)
    image.paste(mask, (0,0), mask)

    draw.rectangle([obj_bbox[0], max(0, obj_bbox[1]-30), obj_bbox[0]+32, max(0, obj_bbox[1]-30)+30], fill=(255, 255, 255), outline=obj_rgb, width=4)
    draw.text((obj_bbox[0]+5, max(0, obj_bbox[1]-30)-2), f'O', font=font, fill=(0,0,0)) #

    return image


def draw_hand_mask(image, draw, hand_idx, hand_bbox, hand_score, side, state, width, height, font):
    if side == 0:
        side_idx = 0
    elif side == 1:
        side_idx = 1
    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(hand_bbox, outline=hand_rgb[side_idx], width=4, fill=hand_rgba[side_idx])
    image.paste(mask, (0, 0), mask)
    # text

    draw = ImageDraw.Draw(image)
    draw.rectangle([hand_bbox[0], max(0, hand_bbox[1] - 30), hand_bbox[0] + 62, max(0, hand_bbox[1] - 30) + 30],
                   fill=(255, 255, 255), outline=hand_rgb[side_idx], width=4)
    draw.text((hand_bbox[0] + 6, max(0, hand_bbox[1] - 30) - 2),
              f'{side_map3[int(float(side))]}-{state_map2[int(float(state))]}', font=font, fill=(0, 0, 0))  #

    return image


def vis_detections_PIL(im, class_name, dets, thresh=0.8, font_path='/shared/niudt/doh_inf/hand_object_detector/lib/model/utils/times_b.ttf'):
    """Visual debugging of detections."""

    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size

    for hand_idx, i in enumerate(range(np.minimum(10, dets.shape[0]))):
        bbox = list(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, 4]
        lr = dets[i, -1]
        state = dets[i, 5]
        if score > thresh:
            image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)

    return image


def vis_detections_filtered_objects_PIL(im, obj_dets, hand_dets, thresh_hand=0.9, thresh_obj=0.35,
                                        font_path='/shared/niudt/doh_inf/hand_object_detector/lib/model/utils/times_b.ttf'):
    # convert to PIL
    im = im[:, :, ::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size

    if (obj_dets is not None) and (hand_dets is not None):
        # img_obj_id = filter_object(obj_dets, hand_dets)
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            if score > thresh_obj:
                # viz obj by PIL
                s = 1#image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height, font)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -2]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)






    elif hand_dets is not None:
        image = vis_detections_PIL(im, 'hand', hand_dets, thresh_hand, font_path)

    return image

if __name__ == "__main__":
    hand_target_info = get_hand_target_info(dir = '/shared/niudt/doh_ptg/hand_object_detector/info_result_Aug_9/k1/det.pkl')
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    num = 0
    root_dir = '/shared/niudt/doh_ptg/hand_object_detector/Inf_data/Aug9/k1'
    for i in os.listdir(root_dir):
        if i[-3:] == 'jpg':
            num = num + 1
    name_list = []
    for i in range(num):
        name = os.path.join(root_dir, '%05d' % (i + 1) + '.jpg')
        name_list.append(name)

    for idx, path in enumerate(name_list):
        print('number: ', idx)

        save_dir = './Aug9/k1/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        out_filename = os.path.join(save_dir, path.split('/')[-1][:-4] + '.png')

        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        ori_img = img
        # start_time = time.time()
        hand_box_info = hand_target_info[2][idx]
        object_box_info = hand_target_info[1][idx]
        # remove the wrong obj box
        if len(object_box_info) !=0:
            object_box_info_ = []
            for _box in object_box_info:
                area = abs(_box[2] - _box[0]) * abs(_box[3] - _box[1])
                percent = area / (img.shape[0] * img.shape[1])
                if percent <= 0.25:
                    object_box_info_.append(_box)
            object_box_info = np.array(object_box_info_)
        img = vis_detections_filtered_objects_PIL(img, object_box_info,
                                                                hand_box_info)
        img.save(out_filename)
        img = read_image(out_filename, format="BGR")
        if not len(object_box_info) == 0 :
            if object_box_info.shape[1] == 11:
                box = object_box_info[0][0:4]
                new_image = np.zeros(img.shape)
                pad = 0
                # new_image[int(box[1]) - pad : int(box[3]) + pad, int(box[0]) - pad : int(box[2]) + pad, :] = img[int(box[1]) - pad : int(box[3]) + pad, int(box[0]) - pad : int(box[2]) + pad, :]
                box = [int(box[1]) - pad, int(box[3]) + pad, int(box[0]) - pad, int(box[2]) + pad]
                predictions, visualized_output = demo.run_on_image_filter_iou(box, ori_img, img)



        # logger.info(
        #     "{}: {} in {:.2f}s".format(
        #         path,
        #         "detected {} instances".format(len(predictions["instances"]))
        #         if "instances" in predictions
        #         else "finished",
        #         time.time() - start_time,
        #     )
        # )
        else:
            # box = object_box_info[0][0:4]
            new_image = np.zeros(img.shape)
            # new_image[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :] = img[int(box[1]): int(box[3]),
            #                                                                    int(box[0]): int(box[2]), :]
            predictions, visualized_output = demo.run_on_image_mask(new_image, img)



        # print(predictions)
        visualized_output.save(out_filename)

    # elif args.webcam:
    #     assert args.input is None, "Cannot have both --input and --webcam!"
    #     assert args.output is None, "output not yet supported with --webcam!"
    #     cam = cv2.VideoCapture(0)
    #     for vis in tqdm.tqdm(demo.run_on_video(cam)):
    #         cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #         cv2.imshow(WINDOW_NAME, vis)
    #         if cv2.waitKey(1) == 27:
    #             break  # esc to quit
    #     cam.release()
    #     cv2.destroyAllWindows()
    # elif args.video_input:
    #     video = cv2.VideoCapture(args.video_input)
    #     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     frames_per_second = video.get(cv2.CAP_PROP_FPS)
    #     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    #     basename = os.path.basename(args.video_input)
    #     codec, file_ext = (
    #         ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
    #     )
    #     if codec == ".mp4v":
    #         warnings.warn("x264 codec not available, switching to mp4v")
    #     if args.output:
    #         if os.path.isdir(args.output):
    #             output_fname = os.path.join(args.output, basename)
    #             output_fname = os.path.splitext(output_fname)[0] + file_ext
    #         else:
    #             output_fname = args.output
    #         assert not os.path.isfile(output_fname), output_fname
    #         output_file = cv2.VideoWriter(
    #             filename=output_fname,
    #             # some installation of opencv may not support x264 (due to its license),
    #             # you can try other format (e.g. MPEG)
    #             fourcc=cv2.VideoWriter_fourcc(*codec),
    #             fps=float(frames_per_second),
    #             frameSize=(width, height),
    #             isColor=True,
    #         )
    #     assert os.path.isfile(args.video_input)
    #     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
    #         if args.output:
    #             output_file.write(vis_frame)
    #         else:
    #             cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
    #             cv2.imshow(basename, vis_frame)
    #             if cv2.waitKey(1) == 27:
    #                 break  # esc to quit
    #     video.release()
    #     if args.output:
    #         output_file.release()
    #     else:
    #         cv2.destroyAllWindows()






