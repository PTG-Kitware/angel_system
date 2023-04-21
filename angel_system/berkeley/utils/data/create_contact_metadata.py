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

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from detectron2.data.detection_utils import read_image
from angel_system.berkeley.demo import predictor, model
from detectron2.utils.visualizer import Visualizer

from angel_system.berkeley.utils.data.dataloaders.common import Re_order, time_from_name, preds_to_kwcoco

from dataloaders.load_kitware_coffee_data import coffee_activity_data_loader # Coffee specific
from dataloaders.load_bbn_medical_data import bbn_activity_data_loader # M2 specific


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_model(config, conf_thr=0.4):
    # Load model
    root_dir = '/angel_workspace'
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

def run_obj_detector(demo, videos_dir, split, no_contact=False, add_hl_hands=True):
    """
    Run object detector trained without contact information 
    on all the videos associated with the task and clear any 
    contact predictions
    """
    preds = {}

    videos = [f'{videos_dir}/{x}' for x in split] #glob.glob(f'{videos_dir}/*',recursive=True) 
    for video_folder in videos:#ub.ProgIter(videos, desc='Running detector on videos'):
        idx = 0
        video_name = video_folder.split('/')[-1]
        
        preds[video_name] = {}

        all_hand_pose_2d_image_space = load_hl_hand_bboxes(video_folder + '/_extracted')

        video_images = glob.glob(f'{video_folder}/_extracted/images/*.png')
        input_list = Re_order(video_images, len(video_images))
        for image_fn in ub.ProgIter(input_list, desc=f'images in {video_name}'):
            frame, time_stamp = time_from_name(image_fn)
            image = read_image(image_fn, format='RGB')
            h, w, c = image.shape

            predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
                    image, current_idx=idx)
            decoded_preds = model.decode_prediction(predictions)

            if decoded_preds is not None:
                preds[video_name][time_stamp] = decoded_preds
                
                if no_contact:
                    for class_ in preds[video_name][time_stamp].keys():
                        # Clear contact states
                        preds[video_name][time_stamp][class_]['obj_obj_contact_state'] = False
                        preds[video_name][time_stamp][class_]['obj_hand_contact_state'] = False
                    
                # Image metadata needed later
                preds[video_name][time_stamp]['meta'] = {
                    'file_name': video_name + '/_extracted/images/'  + image_fn.split('/')[-1],
                    'im_size': {'height': h, 'width': w},
                    'frame_idx': frame
                }

                if add_hl_hands:
                    # Add HL hand bounding boxes if we have them
                    all_hands = all_hand_pose_2d_image_space[time_stamp] if time_stamp in all_hand_pose_2d_image_space.keys() else [] 
                    if all_hands != []:
                        for joints in all_hands:
                            keys = list(joints['joints'].keys())
                            hand_label = joints['hand']

                            all_x_values = [joints['joints'][k]['projected'][0] for k in keys]
                            all_y_values = [joints['joints'][k]['projected'][1] for k in keys]

                            hand_bbox = [min(all_x_values), min(all_y_values), 
                                        max(all_x_values), max(all_y_values)] # tlbr
                            
                            preds[video_name][time_stamp][hand_label] = {
                                "confidence_score": 1,
                                "bbox": hand_bbox,
                                "obj_obj_contact_state": False,
                                "obj_obj_contact_conf": 0,
                                "obj_hand_contact_state": False,
                                "obj_hand_contact_conf": 0,
                            }

            idx += 1

    return preds


def load_hl_hand_bboxes(extracted_dir):
    fn = extracted_dir + '/_hand_pose_2d_data.json'

    if not os.path.exists(fn):
        return {}
    
    with open(fn, 'r') as f:
        hands = ast.literal_eval(f.read())
    
    all_hand_pose_2d = {}
    for hand_info in hands:
        time_stamp = float(hand_info["time_sec"]) + (float(hand_info["time_nanosec"]) * 1e-9)
        if time_stamp not in all_hand_pose_2d.keys():
            all_hand_pose_2d[time_stamp] = []

        hand = hand_info["hand"].lower()
        hand_label = f"hand ({hand})"

        joints = {}
        for joint in hand_info["joint_poses"]:
            #if joint['clipped'] == 0:
            joints[joint["joint"]] = joint # 2d position
        if joints != {}:
            all_hand_pose_2d[time_stamp].append({
                'hand': hand_label,
                'joints': joints
            })

    return all_hand_pose_2d
        

def replace_compound_label(preds, obj, detected_classes, obj_hand_contact_state=False, obj_obj_contact_state=False):
    """
    Check if some subset of obj is in `detected_classes`

    Designed to catch and correct incorrect compound classes
    Ex: obj is "mug + filter cone + filter" but we detected "mug + filter cone"
    """

    compound_objs = [x.strip() for x in obj.split('+')]
    detected_objs = [[y.strip() for y in x.split('+')]for x in detected_classes]

    replaced_classes = []
    for detected_class, objs in zip(detected_classes, detected_objs):
        for obj_ in objs:
            if obj_ in compound_objs:
                replaced_classes.append(detected_class)
                break

    if replaced_classes == []:
        # Case 0, We didn't detect any of the objects
        replaced = None
    elif len(replaced_classes) == 1:
        # Case 1, we detected a subset of the compound as one detection
        replaced = replaced_classes[0]
        #print(f'replaced {replaced} with {obj}')

        preds[obj] = preds.pop(replaced)
        preds[obj]['obj_hand_contact_state'] = obj_hand_contact_state
        preds[obj]['obj_obj_contact_state'] = obj_obj_contact_state
    else:
        # Case 2, the compound was detected as separate boxes
        replaced = replaced_classes
        #print(f'Combining {replaced} detections into compound \"{obj}\"')

        new_bbox = None
        new_conf = None
        for det_obj in replaced:
            bbox = preds[det_obj]['bbox']
            conf = preds[det_obj]['confidence_score']

            if new_bbox is None:
                new_bbox = bbox
            else:
                # Find mix of bboxes
                # TODO: first double check these are close enough that it makes sense to combine?
                new_tlx, new_tly, new_brx, new_bry = new_bbox
                tlx, tly, brx, bry = bbox

                new_bbox = [min(new_tlx, tlx), min(new_tly, tly),
                            max(new_brx, brx), max(new_bry, bry)]

            new_conf = conf if new_conf is None else \
                       (new_conf + conf) / 2 # average???

            # remove old preds
            preds.pop(det_obj)

        new_pred = {
            'confidence_score': new_conf,
            'bbox': new_bbox,
            'obj_obj_contact_state': obj_obj_contact_state,
            'obj_hand_contact_state': obj_hand_contact_state
        }
        preds[obj] = new_pred
    
    return preds, replaced


def find_closest_hands(object_pair, detected_classes, preds):
    # Determine what the hand label is in the video, if any
    # Fixes case where hand label has distinguishing information
    # ex: hand(right) vs hand (left)

    hand_labels = [h for h in detected_classes if 'hand' in h.lower()]
    if len(hand_labels) == 0:
        return None
    
    # find what object we should be interacting with
    try:
        obj = [o for o in object_pair if 'hand' not in o][0] # What to do if we don't have this???
        obj_bbox = preds[obj]["bbox"]
        w = abs(obj_bbox[2] - obj_bbox[0])
        h = abs(obj_bbox[1] - obj_bbox[3])
        obj_center = [obj_bbox[0] + (w/2), obj_bbox[1] + (h/2)]
    except:
        return None # TODO: temp???
    
    # Determine if any of the hands are close enough to the object to 
    # likely be an interaction
    min_dist = 180
    close_hands = []
    for i, hand_label in enumerate(hand_labels):
        hand_bbox = preds[hand_label]["bbox"]
        w = abs(hand_bbox[2] - hand_bbox[0])
        h = abs(hand_bbox[1] - hand_bbox[3])
        hand_center = [hand_bbox[0] + (w/2), hand_bbox[1] + (h/2)]
        dist = math.dist(obj_center, hand_center)

        if dist <= min_dist:
            close_hands.append(hand_label)

    hand_label = close_hands if len(close_hands) > 0 else None
    return hand_label


def update_preds(preds, step_map, videos_dir, no_contact=False):
    """
    Add the contact information back into the detections
    based on when the activities occurred and the contacts
    associated with the activities
    """
    activity_only_preds = {}
    for video_name in preds.keys():
        activity_only_preds[video_name] = {}
        gt_activity = coffee_activity_data_loader(video=video_name) # Coffee specific
        #gt_activity = bbn_activity_data_loader(videos_dir=videos_dir, video=video_name)

        step_map['background'] = [['background', []]] # Add background to loop

        for step, substeps in step_map.items():
            for sub_step in substeps:
                sub_step_str = sub_step[0].lower().strip().strip('.')
                objects = sub_step[1]

                matching_gts = gt_activity[sub_step_str] if sub_step_str in gt_activity.keys() else {}
                
                matching_preds = {}
                for matching_gt in matching_gts:
                    matching_pred = {ts: preds[video_name][ts] for ts in preds[video_name].keys() 
                                    if matching_gt['start'] <= ts <= matching_gt['end']}
                    matching_preds.update(matching_pred)
                
                for time_stamp in matching_preds.keys():
                    detected_classes = list(preds[video_name][time_stamp].keys())
                    preds[video_name][time_stamp]["meta"]["activity_gt"] = sub_step_str
                    
                    if not no_contact:
                        # Keep original detections
                        continue
                    
                    if sub_step_str == 'background':
                        # Remove any detections from background frames
                        for class_ in detected_classes:
                            if class_ != 'meta':
                                del preds[video_name][time_stamp][class_]

                        activity_only_preds[video_name][time_stamp] = preds[video_name][time_stamp]
                        print('background frame ')
                        continue

                    found = []
                    for object_pair in objects:
                        # Determine if we found the objects relevant to the activity
                        found_items = 0
                        
                        obj_hand_contact_state = True if 'hand' in object_pair[0].lower() or 'hand' in object_pair[1].lower() else False
                        obj_obj_contact_state = not obj_hand_contact_state

                        for obj in object_pair:
                            if obj == 'hand':
                                hand_labels = find_closest_hands(object_pair, detected_classes, preds[video_name][time_stamp])
                                if hand_labels is not None:
                                    found_items += 1
                                    for hand_label in hand_labels:
                                        preds[video_name][time_stamp][hand_label]['obj_hand_contact_state'] = True
                            elif obj in detected_classes:
                                found_items += 1
                                preds[video_name][time_stamp][obj]['obj_hand_contact_state'] = obj_hand_contact_state
                                preds[video_name][time_stamp][obj]['obj_obj_contact_state'] = obj_obj_contact_state
                            elif '+' in obj:
                                # We might be missing part of a compound label
                                # Let's try to fix that
                                preds[video_name][time_stamp], replaced = replace_compound_label(preds[video_name][time_stamp], obj, detected_classes,
                                                                  obj_hand_contact_state, obj_obj_contact_state)
                                detected_classes = list(preds[video_name][time_stamp].keys())               
                                if replaced is not None:
                                    found_items += 1
                        
                        if found_items == 2:
                            # Only add frame if it has at least one full object pair
                            found.append(True)
                        else:
                            found.append(False)        
                    if all(found):
                        print('Got all objects needed')
                        activity_only_preds[video_name][time_stamp] = preds[video_name][time_stamp]
                    
    preds = activity_only_preds if no_contact else preds
    print("Updated contact metadata in predictions")
    return preds

def main():
    demo = load_model(config='MC50-InstanceSegmentation/cooking/coffee/preds/mask_rcnn_R_50_FPN_1x_demo.yaml')
    #demo = load_model(config='MC50-InstanceSegmentation/cooking/coffee/training_data/mask_rcnn_R_101_FPN_1x_demo.yaml') # Coffee specific
    metadata = demo.metadata.as_dict()

    """
    bbn_root = '/Padlock_DT/Release_v0.5/v0.52'
    skill = 'M2_Tourniquet'
    skill_dir = f'{bbn_root}/{skill}/Data'
    videos_dir = skill_dir # M2 specific

    training_split = {
        'train': [f'M2-{x}' for x in range(1, 91)],
        'val': [f'M2-{x}' for x in range(91, 139)]
    } # M2 specific
    """

    coffee_root = '/Padlock_DT/Coffee'
    ros_bags_dir = f'{coffee_root}/coffee_recordings/extracted'
    videos_dir = ros_bags_dir # Coffee specific

    training_split = {
        'train': [f'all_activities_{x}' for x in [2, 10, 25, 28, 35]],
        'val': [f'all_activities_{x}' for x in [24, 45]],
        'test': [f'all_activities_{x}' for x in [41, 42, 43]],
    } # Coffee specific

    if not os.path.exists('temp'):
        os.makedirs('temp')

    no_contact = False

    for split in ['test']:#['train', 'val']:
        # Raw detector output
        preds_no_contact = run_obj_detector(demo, videos_dir, 
                                            split=training_split[split], 
                                            no_contact=no_contact, 
                                            add_hl_hands=False)

        with open(f'temp/{split}_preds_no_contact.pickle', 'wb') as fh:
            #preds_no_contact = pickle.load(fh)
            pickle.dump(preds_no_contact, fh)
        
        # Update contact info based on gt
        preds_with_contact = update_preds(preds_no_contact, metadata['original_sub_steps'], videos_dir, no_contact)
        
        
        with open(f'temp/{split}_preds_with_contact.pickle', 'wb') as fh:
            pickle.dump(preds_with_contact, fh)
            
        dset = preds_to_kwcoco(metadata, preds_with_contact, save_fn=f'coffee_preds_{split}.mscoco.json')

    # TODO: train on save_fn + save model


if __name__ == '__main__':
    main()
