from typing import Sequence
from typing import Tuple
from rich.progress import track
from pathlib import Path

import numpy as np
import numpy.typing as npt

####new packages that need to import
from demo.model import *
from detectron2.data.detection_utils import read_image
from utils.data.dataloaders.load_kitware_coffee_data import Re_order
from angel_system.berkeley.demo import predictor, model


gt_to_dive = False
if gt_to_dive:
    # Load coco file
    import kwcoco 
    coco_file = '/home/chris/Documents/video_grouped/2022-11-1/MC_13/ann_contact.json'
    output_file = 'output.csv'
    coco = kwcoco.CocoDataset(coco_file)

    # Fix the filenames
    keys = coco.imgs.keys()
    for k in keys:
        im = coco.imgs[k]
        fn_temp = im['file_name'].split('/')[-1].split('_')
        fn_temp[0] = fn_temp[0][-5:]
        im['file_name'] = '_'.join(fn_temp)

    # text = coco.dumps()
    # with open(output_file, 'w') as file:
    #     file.write(text)
    # for test:

    # - 1: Detection or Track Unique ID
    # - 2: Video or Image String Identifier
    # - 3: Unique Frame Integer Identifier
    # - 4: TL-x (top left of the image is the origin: 0,0)
    # - 5: TL-y
    # - 6: BR-x
    # - 7: BR-y
    # - 8: Auxiliary Confidence (how likely is this actually an object)
    # - 9: Target Length
    text = ''
    id = 1
    keys = coco.anns.keys()
    for k in keys:
        ann = coco.anns[k]
        text += str(id) + ','
        text += coco.imgs[ann['image_id']]['file_name'] + ','
        text += '0,'
        dets = np.array(ann['bbox'],dtype='float64')
        dets[2:] = dets[:2] -  dets[2:]
        diff = dets[:2] - dets[2:]
        dets[:2] += diff
        dets[2:] += diff
        text +=  str(dets[2]) + ','
        text +=  str(dets[3]) + ','
        text +=  str(dets[0]) + ','
        text +=  str(dets[1]) + ','
        text += '0.5,'
        text += '0,'
        text += coco.cats[ann['category_id']]['name']  + ','
        text += '0.5,'
        text += '\n'
        id += 1

        if ann['obj-obj_contact_state'] == 1:
            text += str(id) + ','
            text += coco.imgs[ann['image_id']]['file_name'] + ','
            text += '0,'
            text +=  str(dets[2]) + ','
            text +=  str(dets[3]) + ','
            text +=  str(dets[0]) + ','
            text +=  str(dets[1]) + ','
            text += '0.75,'
            text += '0,'
            text += 'obj-obj contact'  + ','
            text += '0.75,'
            text += '\n'
            id += 1



        if ann['obj-hand_contact_state'] == 1:
            text += str(id) + ','
            text += coco.imgs[ann['image_id']]['file_name'] + ','
            text += '0,'
            text +=  str(dets[2]) + ','
            text +=  str(dets[3]) + ','
            text +=  str(dets[0]) + ','
            text +=  str(dets[1]) + ','
            text += '1.0,'
            text += '0,'
            text += 'obj-hand contact'  + ','
            text += '1.0,'
            text += '\n'
            id += 1

    with open(output_file, 'w') as file:
        file.write(text)



#  Pick which video to run
task = 'coffee'
if task == 'coffee':
    coffee_root = '/Padlock_DT/Coffee'
    data_root = f'{coffee_root}/coffee_recordings/extracted'
    video_name = 'all_activities_2'
    path_root = f'{data_root}/{video_name}/_extracted/images'
    image_output_dir = f'{data_root}/{video_name}/preds/'

if task == 'M2':
    data_root = '/Padlock_DT/Release_v0.5'
    video_name = 'M2-1'
    path_root = f'{data_root}/M2_Tourniquet/Data/{video_name}/images'
    image_output_dir = f'{data_root}/M2_Tourniquet/Data/{video_name}/preds'

Path(image_output_dir).mkdir(parents=True, exist_ok=True)

batch_size = 500
# Glob up images
img_list = os.listdir(path_root)
image_list = []
for img in img_list:
    if 'png' in img:
        image_list.append(os.path.join(path_root, img))
# re-order the input
input_list = Re_order(image_list, len(image_list))


import multiprocessing as mp

mp.set_start_method("spawn", force=True)

parser = model.get_parser()
model_config = 'configs/MC50-InstanceSegmentation/mask_rcnn_R_101_FPN_1x_demo.yaml'

conf_thr = 0.7
args = parser.parse_args(f"--config-file {model_config} --confidence-threshold {conf_thr}".split())

setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))

cfg = setup_cfg(args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

demo = VisualizationDemo_add_smoothing(cfg, last_time=2, draw_output=True, tracking=True)
idx = 0
preds = {}
visualized_outputs = []

for img_path in track(input_list, total=len(input_list), show_speed=True):
    idx = idx + 1
    if idx < 300:
        continue
    img = read_image(img_path, format="RGB")

    frame = img[...,[2, 1, 0]]

    predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
        frame, current_idx=idx)
    print("frame: ", idx, "step: ", step_infos)

    if visualized_output:
        out_fn = image_output_dir + img_path.split('/')[-1]
        visualized_output.save(out_fn)
        # visualized_outputs.append(visualized_output)

    if decode_prediction(predictions) != None:
        preds[idx] = decode_prediction(predictions)

