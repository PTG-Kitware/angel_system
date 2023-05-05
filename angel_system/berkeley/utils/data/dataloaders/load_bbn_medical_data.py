"""
Load object detections and adds hand-object and object-object labels
based on the ground truth annotations.

This should be run on videos not used during training. 
"""
import os
import re
import glob
import cv2
import kwcoco
import kwimage
import pandas as pd
import numpy as np


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

root_dir = '/data/ptg/medical/bbn/data/Release_v0.5/v0.52'
#root_dir = '/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/Release_v0.5'


def bbn_activity_data_loader(videos_dir, video, step_map=None,
                             lab_data=False, add_inter_steps=False, add_before_finished_task=False):
    # Load ground truth activity 
    if lab_data:
        skill_fn = glob.glob(f'{videos_dir}/{video}/*_skills_frame.txt')
        skill_fn = skill_fn[0]
    else:
        skill_fn = f'{videos_dir}/{video}/{video}.skill_labels_by_frame.txt'


    if os.path.isfile(f"{videos_dir}/{video}/THIS_DATA_SET_WAS_EXCLUDED") or not os.path.exists(skill_fn):
        print(f"{video} has no data")
        return {}

    skill_f = open(skill_fn)
    lines = skill_f.readlines()

    gt_activity = {}
    for line in lines:
        # start time, end time, class
        data = line.split("\t")

        class_label = data[2].lower().strip().strip('.')
        if class_label not in gt_activity.keys():
            gt_activity[class_label] = []
        try:
            start = int(data[0])
            end = int(data[1])
        except:
            continue

        gt_activity[class_label].append({
            'start': start,
            'end': end
        })

    skill_f.close()
    
    # Add in more time frames if applicable
    steps = list(step_map.keys()) if step_map else {}
    if add_inter_steps:
        print('Adding in-between steps')
        for i, step in enumerate(steps[:-1]):
            sub_step_str = step_map[step][0][0].lower().strip().strip('.').strip()
            next_sub_step_str = step_map[steps[i+1]][0][0].lower().strip().strip('.').strip()
            try:
                start = gt_activity[sub_step_str][0]['end']
                end = gt_activity[next_sub_step_str][0]['start']
            except:
                continue

            gt_activity[f'In between {step} and {steps[i+1]}'.lower()] =  [{ 'start': start, 'end': end }]

    if add_before_finished_task:
        print('Adding before and finished')
        # before task
        sub_step_str = step_map['step 1'][0][0].lower().strip().strip('.').strip()
        end = gt_activity[sub_step_str][0]['start'] # when the first step starts
        gt_activity['Not started'.lower()] = [{ 'start': 0, 'end': end }]
            
        # after task
        sub_step_str = step_map['step 8'][0][0].lower().strip().strip('.').strip()
        start = gt_activity[sub_step_str][0]['end'] # when the last step ends
        end = len(glob.glob(f'{videos_dir}/{video}/_extracted/images/*.png')) - 1
        gt_activity['Finished'.lower()] = [{ 'start': start, 'end': end }]
    print(video)
    print('gt', gt_activity)
    print(f"Loaded ground truth from {skill_fn}")

    return gt_activity


def bbn_medical_data_loader(skill, valid_classes='all', split='train', filter_repeated_objs=False):
    """
    Load the YoloModel data
    """
    data_dir = f'{root_dir}/{skill}/YoloModel'

    # Load class names
    classes_fn = f'{data_dir}/object_names.txt'
    with open(classes_fn, 'r') as f:
        cats = [l.strip() for l in f.readlines()]
    if valid_classes == 'all':
        valid_classes = cats

    # Load bboxes
    data = {}
    bboxes_dir = f'{data_dir}/LabeledObjects/{split}'

    for ann_fn in glob.glob(f'{bboxes_dir}/*.txt'):
        print(ann_fn)

        try:
            image_fn = ann_fn[:-3] + 'png'
            assert os.path.exists(image_fn)
        except AssertionError:
            image_fn = ann_fn[:-3] + 'jpg'
            assert os.path.exists(image_fn)

        image_name = image_fn[len(data_dir)+1:]

        image = cv2.imread(image_fn)
        im_h, im_w, c = image.shape

        used_classes = []
        data[image_name] = [{'im_size': {'height': im_h, 'width': im_w}}]

        with open(ann_fn, 'r') as f:
            dets = f.readlines()
        
        for det in dets:
            d = det.split(' ')
            cat = cats[int(d[0])]

            if cat not in valid_classes:
                continue

            if cat != 'hand':
                if filter_repeated_objs and cat in used_classes:
                    print(f'{image_name} has repeated {cat} objects, ignoring')
                    # Ignore this image
                    del data[image_name]
                    break

            used_classes.append(cat)

            # center
            x = float(d[1]) * im_w
            y = float(d[2]) * im_h
            w = float(d[3]) * im_w
            h = float(d[4]) * im_h
            
            x = x - (0.5 * w)
            y = y - (0.5 * h)

            # tl_x, tl_y, br_x, br_y
            bbox = [x, y, (x+w), (y+h)]
            #bbox = [x-(0.5*w), y-(0.5*h), x+w, y+h]

            data[image_name].append({
                'area': w*h,
                'cat': cat,
                'segmentation': [],
                'bbox': bbox,
                'obj-obj_contact_state': 0,
                'obj-hand_contact_state': 0,
            })
    
    return valid_classes, data

def data_loader(split):
    # Load gt bboxes for task
    task_classes, task_bboxes = bbn_medical_data_loader('M2_Tourniquet', split=split)

    # Combine task and person annotations
    #gt_bboxes = {**person_bboxes, **task_bboxes}
    #all_classes = person_classes + task_classes

    return task_classes, task_bboxes

def save_as_kwcoco(classes, data, save_fn='bbn-data.mscoco.json'):
    """
    Save the bboxes in the json file
    format used by the detector training
    """
    dset = kwcoco.CocoDataset()

    for class_ in classes:
        dset.add_category(name=class_)

    for im, bboxes in data.items():
        dset.add_image(file_name=im, width=bboxes[0]['im_size']['width'], height=bboxes[0]['im_size']['height'])
        img = dset.index.file_name_to_img[im]

        for bbox in bboxes[1:]:
            cat = dset.index.name_to_cat[bbox['cat']]
            
            xywh = kwimage.Boxes([bbox['bbox']], 'tlbr').toformat('xywh').data[0].tolist()
            
            ann = {
                'area': bbox['area'],
                'image_id': img['id'],
                'category_id': cat['id'],
                'segmentation': bbox['segmentation'],
                'bbox': xywh,
                'obj-obj_contact_state': bbox['obj-obj_contact_state'],
                'obj-hand_contact_state': bbox['obj-hand_contact_state']
            }
            dset.add_annotation(**ann)

    dset.fpath = save_fn
    dset.dump(dset.fpath, newlines=True)

    print_class_freq(dset)

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

def main():
    for split in ['train', 'test']:
        classes, gt_bboxes = data_loader(split)

        out = f'{root_dir}/M2_Tourniquet/YoloModel/M2_YoloModel_LO_{split}.mscoco.json'
        save_as_kwcoco(classes, gt_bboxes, save_fn=out)

    # TODO: train on out kwcoco file + save

if __name__ == '__main__':
    main()
