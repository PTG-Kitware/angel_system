"""
Load object detections and adds hand-object and object-object labels
based on the ground truth annotations.

This should be run on videos not used during training. 
"""
import os
import kwimage
import kwcoco
import cv2
import glob
import numpy as np
import pandas as pd
import ubelt as ub
import pickle
import ast
import math
import pprint
import random 

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from detectron2.data.detection_utils import read_image
from demo import predictor, model
from detectron2.utils.visualizer import Visualizer

from utils.data.dataloaders.common import Re_order, time_from_name, preds_to_kwcoco
from utils.data.update_dets_utils import load_hl_hand_bboxes, replace_compound_label, find_closest_hands

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_model(config, conf_thr=0.01):
    # Load model
    root_dir = '/home/local/KHQ/hannah.defazio/angel_system'
    berkeley_configs_dir = f"{root_dir}/angel_system/berkeley/configs"
    model_config = f"{berkeley_configs_dir}/{config}"

    parser = model.get_parser()
    args = parser.parse_args(f"--config-file {model_config} --confidence-threshold {conf_thr}".split())
    print("Arguments: " + str(args))

    cfg = model.setup_cfg(args)
    print(f'Model: {cfg.MODEL.WEIGHTS}')

    demo = predictor.VisualizationDemo_add_smoothing(
        cfg,
        last_time=2,
        draw_output=False,
        tracking=False
    )
    print(f'Loaded {model_config}')
    return demo

def run_obj_detector(demo, stage, bbn_root, data_dirs, split, no_contact=False, add_hl_hands=True):
    """
    Run object detector trained without contact information 
    on all the videos associated with the task and clear any 
    contact predictions
    """
    preds = {}

    videos = []
    for x in split:
        x_split = x.split('_')
        if x_split[0] == 'tq':
            # BBN lab videos
            videos.append(f'{bbn_root}/{data_dirs[1]}/{x}')
        elif x_split[0] == 'kitware':
            # Kitware lab videos
            videos.append(f'{bbn_root}/{data_dirs[2]}/{x}')
        else:
            videos.append(f'{bbn_root}/{data_dirs[0]}/{x}')
    
    for video_folder in videos:#ub.ProgIter(videos, desc='Running detector on videos'):
        idx = 0
        video_name = video_folder.split('/')[-1]
        
        preds[video_name] = {}

        if add_hl_hands:
            all_hand_pose_2d_image_space = load_hl_hand_bboxes(video_folder + '/_extracted')

        video_images = glob.glob(f'{video_folder}/_extracted/images/*.png')
        input_list = Re_order(video_images, len(video_images))
        for image_fn in ub.ProgIter(input_list, desc=f'images in {video_name}'):
            frame, time_stamp = time_from_name(image_fn)
            
            image = read_image(image_fn, format='RGB')
            if stage == 'results':
                image = Image.fromarray(image)
                image = image.resize(size=(760, 428), resample=Image.BILINEAR)
                image = np.array(image)

            h, w, c = image.shape

            predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
                    image, current_idx=idx)
            decoded_preds = model.decode_prediction(predictions)
            using_contact = True if predictions[2] is not None else False

            if decoded_preds is not None:
                preds[video_name][frame] = decoded_preds
                
                if no_contact and using_contact:
                    for class_, dets in preds[video_name][frame].items():
                        for i in range(len(dets)):
                            # Clear contact states
                            preds[video_name][frame][class_][i]['obj_obj_contact_state'] = False
                            preds[video_name][frame][class_][i]['obj_hand_contact_state'] = False
                        
                # Image metadata needed later
                preds[video_name][frame]['meta'] = {
                    'file_name': image_fn, #bbn_root + '/' + data_dir + '/' + video_name + '/_extracted/images/'  + image_fn.split('/')[-1],
                    'im_size': {'height': h, 'width': w},
                    'frame_idx': frame
                }

                if add_hl_hands:
                    # Add HL hand bounding boxes if we have them
                    all_hands = all_hand_pose_2d_image_space[frame] if frame in all_hand_pose_2d_image_space.keys() else [] 
                    if all_hands != []:
                        for joints in all_hands:
                            keys = list(joints['joints'].keys())
                            hand_label = joints['hand']

                            all_x_values = [joints['joints'][k]['projected'][0] for k in keys]
                            all_y_values = [joints['joints'][k]['projected'][1] for k in keys]

                            hand_bbox = [min(all_x_values), min(all_y_values), 
                                        max(all_x_values), max(all_y_values)] # tlbr
                            
                            new_det = {
                                "confidence_score": 1,
                                "bbox": hand_bbox,
                            }
                            preds[video_name][frame][hand_label] = [new_det]

                            if using_contact:
                                con = {
                                    "obj_obj_contact_state": False,
                                    "obj_obj_contact_conf": 0,
                                    "obj_hand_contact_state": False,
                                    "obj_hand_contact_conf": 0,
                                }

                                preds[video_name][frame][hand_label] = [{**new_det, **con}]

            idx += 1

    return preds, using_contact

def update_preds(activity_data_loader, preds, using_contact, original_step_map, step_map, data_root, data_dirs,
                 experiment_flags):
    """
    Add the contact information back into the detections
    based on when the activities occurred and the contacts
    associated with the activities
    """
    activity_only_preds = {}
    for video_name in preds.keys():
        x_split = video_name.split('_')
        if x_split[0] == 'tq':
            videos_dir = f'{data_root}/{data_dirs[1]}'
            lab_data = True
        elif x_split[0] == 'kitware':
            # Kitware lab videos
            videos_dir = f'{data_root}/{data_dirs[2]}'
            lab_data = False
        else:
            videos_dir = f'{data_root}/{data_dirs[0]}'
            lab_data = False

        activity_only_preds[video_name] = {}
        
        gt_activity = activity_data_loader(videos_dir, video_name, original_step_map, lab_data,
                                           experiment_flags['using_inter_steps'], experiment_flags['using_before_finished_task'])

        for step, substeps in step_map.items():
            #print(step)
            inter_step = False
            if step[-2:] == '.5' or step == 'before' or step == 'finished':
                inter_step = True

            for sub_step in substeps:
                sub_step_str = sub_step[0].lower().strip().strip('.').strip()
                #print(sub_step_str)
                objects = sub_step[1]

                matching_gts = gt_activity[sub_step_str] if sub_step_str in gt_activity.keys() else {}
                #print('matching gt', matching_gts)
                
                matching_preds = {}
                for matching_gt in matching_gts:
                    matching_pred = {ts: preds[video_name][ts] for ts in preds[video_name].keys() 
                                    if matching_gt['start'] <= ts <= matching_gt['end']}
                    matching_preds.update(matching_pred)
                #print('matching preds', matching_preds)

                for frame in matching_preds.keys():
                    detected_classes = list(preds[video_name][frame].keys())
                    preds[video_name][frame]["meta"]["activity_gt"] = f'{sub_step_str} ({step})' if not inter_step else sub_step_str
                    
                    if not experiment_flags['no_contact']:
                        # Keep original detections
                        continue
                    
                    if sub_step_str == 'background':
                        # Remove any detections from background frames
                        for class_ in detected_classes:
                            if class_ != 'meta':
                                del preds[video_name][frame][class_]

                        activity_only_preds[video_name][frame] = preds[video_name][frame]
                        print('background frame ')
                        continue

                    found = []
                    for object_pair in objects:
                        # Determine if we found the objects relevant to the activity
                        found_items = 0
                        
                        obj_hand_contact_state = True if 'hand' in object_pair[0].lower() or 'hand' in object_pair[1].lower() else False
                        obj_obj_contact_state = not obj_hand_contact_state

                        if inter_step:
                            obj_hand_contact_state = False
                            obj_obj_contact_state = False

                        # Update contact metadata
                        for obj in object_pair:
                            
                            if obj == 'hand':
                                hand_labels = find_closest_hands(object_pair, detected_classes, preds[video_name][frame])
                                
                                if hand_labels is not None:
                                    found_items += 1
                                    if using_contact:
                                        for hand_label in hand_labels:
                                            for i in range(len(preds[video_name][frame][hand_label])):
                                                preds[video_name][frame][hand_label][i]['obj_hand_contact_state'] = obj_hand_contact_state

                            elif obj in detected_classes:
                                found_items += 1
                                if using_contact:
                                    for i in range(len(preds[video_name][frame][obj])):
                                        preds[video_name][frame][obj][i]['obj_hand_contact_state'] = obj_hand_contact_state
                                        preds[video_name][frame][obj][i]['obj_obj_contact_state'] = obj_obj_contact_state
                                        
                            elif '+' in obj:
                                # We might be missing part of a compound label
                                # Let's try to fix that
                                preds[video_name][frame], replaced = replace_compound_label(preds[video_name][frame], obj, detected_classes,
                                                                                            using_contact, obj_hand_contact_state, obj_obj_contact_state)
                                detected_classes = list(preds[video_name][frame].keys())               
                                if replaced is not None:
                                    found_items += 1

                        if found_items == 2:
                            # Only add frame if it has at least one full object pair
                            found.append(True)
                        else:
                            found.append(False)

                    # Add step number to class label
                    if experiment_flags['using_step_labels']:
                        for class_ in detected_classes:
                            if class_ == 'meta':
                                continue
                            #print(class_)
                            if '(step' not in class_ and '(before)' not in class_ and '(finished)' not in class_:
                                #print('updating label to include step')
                                preds[video_name][frame][f'{class_} ({step})'] = preds[video_name][frame].pop(class_)

                    detected_classes = list(preds[video_name][frame].keys())
            
                    if experiment_flags['filter_all_obj_frames']:
                        if all(found):
                            print('Got all objects needed')
                            activity_only_preds[video_name][frame] = preds[video_name][frame]
                    else:
                        activity_only_preds[video_name][frame] = preds[video_name][frame]

    preds = activity_only_preds if experiment_flags['filter_activity_frames'] else preds
    print("Updated contact metadata in predictions")
    return preds


def coffee_main():

    coffee_root = '/Padlock_DT/Coffee'
    ros_bags_dir = f'{coffee_root}/coffee_recordings/extracted'
    videos_dir = ros_bags_dir # Coffee specific

    training_split = {
        'train': [f'all_activities_{x}' for x in [2, 10, 25, 28, 35]],
        'val': [f'all_activities_{x}' for x in [24, 45]],
        'test': [f'all_activities_{x}' for x in [41, 42, 43]],
    } # Coffee specific

    from dataloaders.load_kitware_coffee_data import coffee_activity_data_loader
    activity_data_loader = coffee_activity_data_loader



def tourniquet_main(stage, using_inter_steps, using_before_finished_task):
    #demo = load_model(config='MC50-InstanceSegmentation/medical/M2/stage1/mask_rcnn_R_101_FPN_1x_BBN_M2_demo.yaml', conf_thr=0.4)
    demo = load_model(config='MC50-InstanceSegmentation/medical/M2/stage2/mask_rcnn_R_50_FPN_1x_BBN_M2_labels_with_steps_demo.yaml', conf_thr=0.01)

    bbn_root = '/data/ptg/medical/bbn/data'
    data_root ='Release_v0.5/v0.52'
    skill = 'M2_Tourniquet'
    m2_data_dir = f'{data_root}/{skill}/Data' # M2 specific
    lab_data_dir = 'M2_Lab_data/skills_by_frame/'
    kitware_dir = f'kitware_m2'

    m2_videos = [f'M2-{x}' for x in range(1, 139+1)]
    lab_videos = [f'tq_{x}' for x in range(1, 32+1)]
    kitware_videos = [f'kitware_m2_video_{x}' for x in range(1, 32+1)]

    ignore_videos = [f'M2-{x}' for x in [
        15, # bad video
        #46, 47, 48, 49, 50, 51, 53, 54, 56, 58, 62, 65, 66, 67, 68, 69, 102, 105, 135, 136, 138, # objs only in corners, might be able to crop? 
        78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 122, 125, 129 # no hope for these
        # currently ignoring these videos because of repeated objs in frame when an activity is happening
        ]
    ]

    all_videos = m2_videos + lab_videos + kitware_videos
    data_dirs = (m2_data_dir, lab_data_dir, kitware_dir)
    
    good_videos = [x for x in all_videos if x not in ignore_videos]
    random.shuffle(good_videos)

    num_videos = len(good_videos)
    print(f"Using {num_videos} videos")

    """
    i1 = int(0.5 * num_videos)
    i2 = int(0.3 * num_videos)
    i3 = i1 + i2
    i4 = int(0.1 * num_videos)

    training_split = {
        'train_contact': good_videos[0:i1], # 50%
        'train_activity': good_videos[i1:i3], # 30%
        'val': good_videos[i3:(i3+i4)], # 10%
        'test': good_videos[(i3+i4):] # 10%
    } # M2 specific
    """
    training_split = {
            'train_contact': ['M2-131', 'M2-63', 'M2-71', 'tq_2', 'M2-11', 'tq_4', 'M2-54', 'tq_12', 'M2-73', 'M2-62', 'tq_8', 'M2-46', 'M2-60', 'M2-24', 'M2-19', 'M2-2', 'M2-74', 'M2-28', 'M2-135', 'M2-130', 'M2-10', 'M2-61', 'tq_21', 'M2-137', 'M2-53', 'M2-40', 'M2-66', 'M2-138', 'M2-124', 'M2-102', 'M2-32', 'M2-29', 'tq_23', 'tq_3', 'tq_20', 'M2-41', 'tq_19', 'M2-1', 'tq_14', 'M2-127', 'tq_13', 'M2-23', 'M2-17', 'tq_10', 'M2-139', 'M2-55', 'M2-69', 'tq_11', 'M2-68', 'M2-13', 'M2-133', 'M2-121', 'M2-14', 'M2-50', 'tq_5', 'M2-44', 'M2-43', 'M2-35', 'M2-75', 'M2-37'],
            'train_activity': ['M2-38', 'M2-48', 'M2-123', 'M2-65', 'M2-59', 'M2-49', 'M2-45', 'M2-5', 'M2-7', 'tq_16', 'M2-56', 'M2-12', 'M2-67', 'M2-20', 'M2-64', 'M2-6', 'M2-57', 'M2-126', 'M2-77', 'M2-106', 'M2-120', 'M2-132', 'M2-76', 'M2-27', 'M2-136', 'M2-47', 'M2-3', 'M2-18', 'M2-26', 'tq_6', 'M2-25', 'M2-8', 'M2-34', 'tq_9', 'M2-21', 'tq_1'], 
            'val': ['M2-72', 'M2-58', 'M2-134', 'M2-105', 'tq_18', 'M2-33', 'tq_22', 'M2-9', 'M2-42', 'M2-30', 'M2-16', 'M2-128'], 
            'test': ['tq_17', 'M2-119', 'M2-51', 'M2-31', 'M2-22', 'M2-36', 'M2-39', 'M2-70', 'tq_15', 'M2-4', 'M2-52', 'tq_7']
    }
    training_split['train_contact'] = kitware_videos[:20] + [f'tq_{x}' for x in range(24, 28+1)] + training_split['train_contact']
    training_split['train_activity'] = kitware_videos[20:] + [f'tq_{x}' for x in range(29, 32+1)] + training_split['train_activity']
    
    print(training_split)

    from dataloaders.load_bbn_medical_data import bbn_activity_data_loader
    activity_data_loader = bbn_activity_data_loader

    # Update step map
    metadata = demo.metadata.as_dict()

    step_map = metadata['sub_steps']
    if using_inter_steps:
        steps = list(step_map.keys())
        for i, step in enumerate(steps[:-1]):
            step_map[f'{step}.5'] = [[f'In between {step} and {steps[i+1]}'.lower(), [['tourniquet_tourniquet', 'hand']]]]

    if using_before_finished_task:
        step_map['before'] = [['Not started'.lower(), [['tourniquet_tourniquet', 'hand']]]]
        step_map['finished'] = [['Finished'.lower(), [['tourniquet_tourniquet', 'hand']]]]
    print(f'step map: {step_map}')
    
    #data_dirs = (kitware_test_dir, lab_data_dir)
    #training_split['test'] = kitware_videos
    return demo, training_split, bbn_root, data_dirs, activity_data_loader, metadata, step_map


def main():
    experiment_name = 'm2_all_data_cleaned_fixed_with_steps'
    stage = 'results'

    print('Experiment: ', experiment_name)
    print('Stage: ', stage)
    
    # Various experiment flags
    experiment_flags = {
        'no_contact': False if stage == 'results' else True,
        'filter_activity_frames': False if stage == 'results' else True,
        'filter_all_obj_frames': False if stage == 'results' else True,
        'using_step_labels' : True,
        'using_inter_steps': False,
        'using_before_finished_task': False
    }
    print(f'experiment flags: {experiment_flags}')

    demo, training_split, data_root, data_dirs, activity_data_loader, metadata, step_map = tourniquet_main(stage, experiment_flags['using_inter_steps'], experiment_flags['using_before_finished_task'])
    
    if stage == 'stage2':
        splits = ['train_activity']#['val', 'train_contact', 'test', 'train_activity']
    else:
        splits = ['test', 'train_activity', 'val']

    for split in splits:
        print(f'{split}: {len(training_split[split])} videos')

        # Raw detector output
        preds_no_contact, using_contact = run_obj_detector(demo, stage,
                                            data_root, data_dirs,
                                            training_split[split], 
                                            no_contact=experiment_flags['no_contact'], 
                                            add_hl_hands=False)
        print(f'Using contact: {using_contact}')
        
        fn = f'temp/{experiment_name}_{split}_preds_no_contact.pickle'
        print('temp file: ', fn)
        with open(fn, 'wb') as fh:
            #preds_no_contact = pickle.load(fh)
            pickle.dump(preds_no_contact, fh)

        # Update contact info based on gt
        
        preds_with_contact = update_preds(activity_data_loader,
                                          preds_no_contact,
                                          using_contact,
                                          metadata['sub_steps'],
                                          step_map, 
                                          data_root, data_dirs, 
                                          experiment_flags)
        
        dset = preds_to_kwcoco(metadata, preds_with_contact, '', save_fn=f'{experiment_name}_{stage}_{split}.mscoco.json',
                               # assuming detector already has the right labels so these aren't needed here
                               using_step_labels=False if stage == 'results' else experiment_flags['using_step_labels'],
                               using_inter_steps=False if stage == 'results' else experiment_flags['using_inter_steps'],
                               using_before_finished_task=False if stage == 'results' else experiment_flags['using_before_finished_task'])

    # TODO: train on save_fn + save model


if __name__ == '__main__':
    main()
