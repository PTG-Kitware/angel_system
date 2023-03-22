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
import pandas as pd
import ubelt as ub
import pickle

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from detectron2.data.detection_utils import read_image
from angel_system.berkeley.demo import predictor, model
from detectron2.utils.visualizer import Visualizer

from angel_system.berkeley.utils.data.dataloaders.common import Re_order, time_from_name

from dataloaders.load_kitware_coffee_data import coffee_activity_data_loader # Coffee specific
from dataloaders.load_bbn_medical_data import bbn_activity_data_loader # M2 specific


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_model(config, conf_thr=0.7):
    # Load model
    root_dir = '/angel_workspace'
    berkeley_configs_dir = f"{root_dir}/angel_system/berkeley/configs"
    model_config = f"{berkeley_configs_dir}/{config}"

    parser = model.get_parser()
    args = parser.parse_args(f"--config-file {model_config} --confidence-threshold {conf_thr}".split())
    print("Arguments: " + str(args))

    cfg = model.setup_cfg(args)

    demo = predictor.VisualizationDemo_add_smoothing(
        cfg,
        last_time=2,
        draw_output=False,
        tracking=False
    )
    print(f'Loaded {model_config}')
    return demo

def run_obj_detector_no_contact(demo, videos_dir, split):
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
        print(video_name)
        
        preds[video_name] = {}

        video_images = glob.glob(f'{video_folder}/_extracted/images/*.png')
        input_list = Re_order(video_images, len(video_images))
        for image_fn in ub.ProgIter(input_list, desc='images'):
            image = read_image(image_fn, format='RGB')
            h, w, c = image.shape

            predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
                    image, current_idx=idx)
            decoded_preds = model.decode_prediction(predictions)

            frame, time_stamp = time_from_name(image_fn)
            
            if decoded_preds is not None:
                preds[video_name][time_stamp] = decoded_preds
                
                for class_ in preds[video_name][time_stamp].keys():
                    # Clear contact states
                    preds[video_name][time_stamp][class_]['obj_obj_contact_state'] = False
                    preds[video_name][time_stamp][class_]['obj_hand_contact_state'] = False

                preds[video_name][time_stamp]['meta'] = {
                    'file_name': video_name + '/_extracted/images/'  + image_fn.split('/')[-1],
                    'im_size': {'height': h, 'width': w},
                    'frame_idx': frame
                }

            idx += 1

    return preds



def update_contacts(preds, step_map, videos_dir):
    """
    Add the contact information back into the detections
    based on when the activities occurred and the contacts
    associated with the activities
    """

    for video_name in preds.keys():
        gt_activity = coffee_activity_data_loader(video=video_name) # Coffee specific
        #gt_activity = bbn_activity_data_loader(videos_dir=videos_dir, video=video_name)

        for step, substeps in step_map.items():
            for sub_step in substeps:
                sub_step_str = sub_step[0].lower().strip().strip('.')
                objects = sub_step[1]

                matching_gts = gt_activity[sub_step_str]
                
                matching_preds = {}
                for matching_gt in matching_gts:
                    matching_pred = {ts: preds[video_name][ts] for ts in preds[video_name].keys() 
                                    if matching_gt['start'] <= ts <= matching_gt['end']}
                    matching_preds.update(matching_pred)
                
                for object_pair in objects:
                    for time_stamp in matching_preds.keys():
                        detected_classes = preds[video_name][time_stamp].keys()
                        
                        if 'hand' in object_pair[0].lower() or 'hand' in object_pair[1].lower():
                            # Mark obj/hand contact
                            hand_labels = [h for h in detected_classes if 'hand' in h]
                            # Determine what the hand label is in the video, if any
                            # Fixes case where hand label has distinguishing information
                            # ex: hand(right) vs hand (left)
                            if len(hand_labels) > 1:
                                # find overlapping hand bbox?
                                # TODO
                                print('Multiple hands found in frame')
                                hand_label = hand_labels[0] # TODO: temporary!
                            elif len(hand_labels) == 1:
                                hand_label = hand_labels[0]
                            else:
                                hand_label = None

                            for obj in object_pair:
                                if obj == 'hand' and hand_label is not None:
                                    print(hand_label)
                                    preds[video_name][time_stamp][hand_label]['obj_hand_contact_state'] = True
                                elif obj in detected_classes:
                                    preds[video_name][time_stamp][obj]['obj_hand_contact_state'] = True

                        else:
                            # Mark obj/obj contact
                            for obj in object_pair:
                                if obj in detected_classes:
                                    preds[video_name][time_stamp][obj]['obj_obj_contact_state'] = True
                           
    print("Updated contact metadata in predictions")
    return preds

# Save
def preds_to_kwcoco(metadata, preds, save_fn='result-with-contact.mscoco.json'):
    """
    Save the predicitions in the json file
    format used by the detector training
    """
    dset = kwcoco.CocoDataset()

    for class_ in metadata['thing_classes']:
        dset.add_category(name=class_)

    for video_name, predictions in preds.items():
        dset.add_video(name=video_name)
        vid = dset.index.name_to_video[video_name]['id']

        for time_stamp in sorted(predictions.keys()):
            dets = predictions[time_stamp]
            fn = dets['meta']['file_name']
            
            dset.add_image(file_name=fn, video_id=vid, frame_index=dets['meta']['frame_idx'],
                           width=dets['meta']['im_size']['width'], height=dets['meta']['im_size']['height'])
            img = dset.index.file_name_to_img[fn]

            del dets['meta']

            for class_, det in dets.items():
                cat = dset.index.name_to_cat[class_]
                
                xywh = kwimage.Boxes([det['bbox']], 'tlbr').toformat('xywh').data[0].tolist()

                ann = {
                    'area': xywh[2] * xywh[3],
                    'image_id': img['id'],
                    'category_id': cat['id'],
                    'segmentation': [],
                    'bbox': xywh,
                    'confidence': det['confidence_score'],
                    'obj-obj_contact_state': det['obj_obj_contact_state'],
                    'obj-hand_contact_state': det['obj_hand_contact_state']
                }
                dset.add_annotation(**ann)
                
    dset.fpath = save_fn
    dset.dump(dset.fpath, newlines=True)
    print(f'Saved predictions with contact info to {save_fn}')

    return dset




def main():
    demo = load_model(config='MC50-InstanceSegmentation/cooking/coffee/training_data/mask_rcnn_R_50_FPN_1x_demo.yaml') # Coffee specific
    #demo = load_model(config='MC50-InstanceSegmentation/mask_rcnn_R_50_FPN_1x_BBN_M1_hands_M2_demo.yaml') # M2 specific
    metadata = demo.metadata.as_dict()

    """
    bbn_root = '/Padlock_DT/Release_v0.5'
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
        'val': [f'all_activities_{x}' for x in [24, 45]]
    } # Coffee specific

    if not os.path.exists('temp'):
        os.makedirs('temp')

    for split in ['train', 'val']:#
        preds_no_contact = run_obj_detector_no_contact(demo, videos_dir, split=training_split[split])
        
        fh = open(f'temp/{split}_preds_no_contact.pickle', 'wb')
        pickle.dump(preds_no_contact, fh)
        fh.close()
        
        preds_with_contact = update_contacts(preds_no_contact, metadata['original_sub_steps'], videos_dir)
        
        fh = open(f'temp/{split}_preds_with_contact.pickle', 'wb')
        pickle.dump(preds_with_contact, fh)
        fh.close()
        
        dset = preds_to_kwcoco(metadata, preds_with_contact, save_fn=f'coffee_preds_with_contact_fix_hands_{split}.mscoco.json')

    # TODO: train on save_fn + save model


if __name__ == '__main__':
    main()
