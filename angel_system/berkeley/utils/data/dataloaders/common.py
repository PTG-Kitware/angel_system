import re
import os
import kwcoco
import textwrap

import numpy as np
import ubelt as ub

from pathlib import Path
from detectron2.data.detection_utils import read_image


RE_FILENAME_TIME = re.compile(r"frame_(?P<frame>\d+)_(?P<ts>\d+(?:_|.)\d+).(?P<ext>\w+)")
def time_from_name(fname):
    """
    Extract the float timestamp from the filename.

    :param fname: Filename of an image in the format
        frame_<frame number>_<seconds>_<nanoseconds>.<extension>

    :return: timestamp (float) in seconds
    """
    fname = os.path.basename(fname)
    match = RE_FILENAME_TIME.match(fname)
    time = match.group('ts')
    if '_' in time:
        time = time.split('_')
        time = float(time[0]) + (float(time[1]) * 1e-9)
    elif '.' in time:
        time = float(time)

    frame = match.group('frame')
    return int(frame), time

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


    
def coffee_activity_data_loader(video='all_activities_11'):
    import pandas as pd
    # Load ground truth activity 
    #root_dir = '/media/hannah.defazio/Padlock_DT/Data/notpublic/PTG/Coffee/coffee_labels'
    root_dir = '/Padlock_DT/Coffee/coffee_labels'
    gt_dir = f'{root_dir}/Labels'

    gt_activity_fn = f'{gt_dir}/{video}.feather'
    print(f"Loaded ground truth from {gt_activity_fn}")

    gt = pd.read_feather(gt_activity_fn) # Keys: class, start_frame,  end_frame, exploded_ros_bag_path
    gt_activity = {}
    for i, row in gt.iterrows():
        class_label = row["class"].lower().strip().strip('.')
        if class_label not in gt_activity.keys():
            gt_activity[class_label] = []
        start_frame, start_time = time_from_name(row["start_frame"])
        end_frame, end_time = time_from_name(row["end_frame"])
        gt_activity[class_label].append({
            'start': start_frame,#start_time,
            'end': end_frame#end_time
        })
    print(f"Loaded ground truth from {gt_activity_fn}")


    """
    gt_activity = {
        sub_step_str: [{
            'start': 123456,
            'end': 657899
        }]
    }
    """
    return gt_activity


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
            if conf < 0.4:
                continue

            using_contact = True if 'obj-obj_contact_state' in ann.keys() else False
            
            
            box = ann['bbox']
            label = dset.cats[ann['category_id']]['name']

            color = 'r'
            if using_contact and ann['obj-obj_contact_state']:
                color = 'g'
            if using_contact and ann['obj-hand_contact_state']:
                color = 'b'
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], 
                                     linewidth=1, edgecolor=color, facecolor='none')

            ax.add_patch(rect)
            ax.annotate(label, (box[0], box[1]), color='black')
        
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
   
    kw = 'kitware_m2_results_test.mscoco.json'
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
    exp = 'm2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps_no_contact_aug'#'m2_with_lab_cleaned_fixed_data_with_inter_and_before_finished_steps'
    save_dir = f'{stage_dir}/{exp}/visualization/{split}'
    
    save_dir = 'visualization'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    visualize_kwcoco(kw, save_dir)
    #visualize_kwcoco(f'{stage_dir}/{exp}/{kw}', save_dir)


if __name__ == '__main__':
    main()
