import re
import os
import cv2
import kwcoco
import textwrap

import numpy as np
import ubelt as ub
import pandas as pd

from pathlib import Path
from typing import List

from detectron2.data.detection_utils import read_image
from utils.data.dataloaders.structures import Activity


def Re_order(image_list, image_number):
    img_id_list = []
    for img in image_list:
        img_id, ts = time_from_name(img)
        img_id_list.append(img_id)
    img_id_arr = np.array(img_id_list)
    s = np.argsort(img_id_arr)
    new_list = []
    for i in range(image_number):
        idx = s[i]
        new_list.append(image_list[idx])
    return new_list

def find_matching_gt_activity(gt_activity, fn):
    fn = os.path.basename(fn)
    frame, time = time_from_name(fn)

    """
    gt_activity = {
        sub_step_str: [{
            'start': 123456,
            'end': 657899
        }]
    }
    """

    matching_gt = {}
    for sub_step_str, times in gt_activity.items():
        for gt_time in times:
            if gt_time['start'] <= time <= gt_time['end']:
                return sub_step_str
            
    return 'None'

# Save
def preds_to_kwcoco(metadata, preds, save_dir, save_fn='result-with-contact.mscoco.json',
                    using_step_labels=False, using_inter_steps=False, using_before_finished_task=False):
    """
    Save the predicitions in the json file
    format used by the detector training
    """
    import kwimage
    dset = kwcoco.CocoDataset()

    for class_ in metadata['thing_classes']:
        if not using_step_labels and not using_inter_steps:
            # add original classes from model
            dset.add_category(name=class_)

        if using_step_labels:
            for i in range(1, 9):
                dset.add_category(name=f'{class_} (step {i})')

                if using_inter_steps:
                    if i != 8:
                        dset.add_category(name=f'{class_} (step {i+0.5})')
        if using_before_finished_task:
            dset.add_category(name=f'{class_} (before)')
            dset.add_category(name=f'{class_} (finished)')

    for video_name, predictions in preds.items():
        dset.add_video(name=video_name)
        vid = dset.index.name_to_video[video_name]['id']

        for time_stamp in sorted(predictions.keys()):
            dets = predictions[time_stamp]
            fn = dets['meta']['file_name']

            activity_gt = dets['meta']['activity_gt'] if 'activity_gt' in dets['meta'].keys() else None
            
            dset.add_image(file_name=fn, video_id=vid, frame_index=dets['meta']['frame_idx'],
                           width=dets['meta']['im_size']['width'], height=dets['meta']['im_size']['height'],
                           activity_gt=activity_gt)
            img = dset.index.file_name_to_img[fn]

            del dets['meta']

            for class_, det in dets.items():
                for i in range(len(det)):
                    cat = dset.index.name_to_cat[class_]
                    
                    xywh = kwimage.Boxes([det[i]['bbox']], 'tlbr').toformat('xywh').data[0].tolist()

                    ann = {
                        'area': xywh[2] * xywh[3],
                        'image_id': img['id'],
                        'category_id': cat['id'],
                        'segmentation': [],
                        'bbox': xywh,
                        'confidence': det[i]['confidence_score']
                    }

                    if 'obj_obj_contact_state' in det[i].keys():
                        ann['obj-obj_contact_state'] = det[i]['obj_obj_contact_state']
                        ann['obj-obj_contact_conf'] = det[i]['obj_obj_contact_conf']
                    if 'obj_hand_contact_state' in det[i].keys():
                        ann['obj-hand_contact_state'] = det[i]['obj_hand_contact_state'] 
                        ann['obj-hand_contact_conf'] = det[i]['obj_hand_contact_conf']
                    
                    dset.add_annotation(**ann)
                
    dset.fpath = f'{save_dir}/{save_fn}' if save_dir != '' else save_fn
    dset.dump(dset.fpath, newlines=True)
    print(f'Saved predictions to {dset.fpath}')

    return dset

def print_class_freq(dset):
    freq_per_class = dset.category_annotation_frequency()
    stats = []

    for cat in dset.cats.values():
        freq = freq_per_class[cat['name']]
        class_ = {
            'id': cat['id'],
            'name': cat['name'],
            #'instances_count': freq,
            #'def': '',
            #'synonyms': [],
            #'image_count': freq,
            #'frequency': '',
            #'synset': ''
        }

        stats.append(class_)

    print(f'MC50_CATEGORIES = {stats}')

def visualize_kwcoco(dset=None, save_dir=''):
    import matplotlib.pyplot as plt
    from PIL import Image
    import matplotlib.patches as patches
    
    red_patch = patches.Patch(color='r', label='obj')
    green_patch = patches.Patch(color='g', label='obj-obj contact')
    blue_patch = patches.Patch(color='b', label='obj-hand contact')
    
    empty_ims = 0
    if type(dset) == str:
        print(f'Loaded dset from file: {dset}')
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im["video_id"]
        #if img_video_id == 3:
        #    continue
        
        fn = im['file_name'].split('/')[-1]
        gt = im['activity_gt']# if hasattr(im, 'activity_gt') else ''
        if not gt:
            gt = ''
        
        fig, ax = plt.subplots()
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = read_image(im['file_name'], format='RGB') # now assuming absolute path
        image = Image.fromarray(image)
        image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)
        
        ax.imshow(image)
        
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann['confidence']

            using_contact = False#True if 'obj-obj_contact_state' in ann.keys() else False
            
            
            x, y, w, h = ann['bbox'] # xywh
            cat = dset.cats[ann['category_id']]['name']
            if 'tourniquet_tourniquet' in cat:
                tourniquet_im = image[int(y):int(y+h), int(x):int(x+w), ::-1]

                m2_fn = fn[:-4] + '_tourniquet_chip.png'
                m2_out = f'{save_dir}/video_{img_video_id}/images/chipped'
                Path(m2_out).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f'{m2_out}/{m2_fn}', tourniquet_im)
            
            label = f"{cat}: {round(conf, 2)}"

            color = 'r'
            if using_contact and ann['obj-obj_contact_state']:
                color = 'g'
            if using_contact and ann['obj-hand_contact_state']:
                color = 'b'
            rect = patches.Rectangle((x, y), w, h, 
                                     linewidth=1, edgecolor=color, facecolor='none')

            ax.add_patch(rect)
            ax.annotate(label, (x, y), color='black')
        
        if using_contact:
            plt.legend(handles=[red_patch, green_patch, blue_patch], loc='lower left')
        
        video_dir = f'{save_dir}/video_{img_video_id}/images/'
        Path(video_dir).mkdir(parents=True, exist_ok=True)
        
        plt.savefig(f'{video_dir}/{fn}', )
        plt.close(fig) # needed to remove the plot because savefig doesn't clear it
    plt.close('all')


def main():
    #coffee_root = '/Padlock_DT/Coffee'
    #ros_bags_dir = f'{coffee_root}/coffee_recordings/extracted/'

    #visualize_kwcoco(ros_bags_dir, 1, f'{ros_bags_dir}/coffee_contact_preds_with_background_all_objs_only_train.mscoco.json')


    ptg_root = '/data/ptg/medical/bbn/'
   
    kw = 'm2_all_data_cleaned_fixed_with_steps_results_train_activity.mscoco.json'
    #kw = 'm2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps_no_contact_aug_results_test.mscoco.json'
    #kw = 'kitware_test_results_test.mscoco.json'
    #kw = 'm2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps_results_train_activity.mscoco.json'

    n = kw[:-12].split('_')
    name = '_'.join(n[:-1])
    split = n[-1]
    if split == 'contact':
        split = 'train_contact'
    if split == 'activity':
        split = 'train_activity'

    stage = 'results'
    stage_dir = f'{ptg_root}/annotations/M2_Tourniquet/{stage}'
    exp = 'm2_all_data_cleaned_fixed_with_steps'#'m2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps'
    if stage == 'stage1':
        save_dir = f'{stage_dir}/visualization_1/{split}'
    else:
        save_dir = f'{stage_dir}/{exp}/visualization/{split}'
    
    #save_dir = 'visualization'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    #visualize_kwcoco(kw, save_dir)
    if stage == 'stage1':
        visualize_kwcoco(f'{stage_dir}/{kw}', save_dir)
    else:
        visualize_kwcoco(f'{stage_dir}/{exp}/{kw}', save_dir)


if __name__ == '__main__':
    main()
