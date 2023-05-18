import math
import kwcoco

import ubelt as ub

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
            all_hand_pose_2d[frame].append({
                'hand': hand_label,
                'joints': joints
            })

    return all_hand_pose_2d


def replace_compound_label(preds, obj, detected_classes, using_contact, obj_hand_contact_state=False, obj_obj_contact_state=False):
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

        if using_contact:
            for i in range(len(preds[obj])):
                preds[obj][i]['obj_hand_contact_state'] = obj_hand_contact_state
                preds[obj][i]['obj_obj_contact_state'] = obj_obj_contact_state
    else:
        # Case 2, the compound was detected as separate boxes
        replaced = replaced_classes
        #print(f'Combining {replaced} detections into compound \"{obj}\"')

        new_bbox = None
        new_conf = None
        for det_obj in replaced:
            assert len(preds[det_obj]) == 1
            bbox = preds[det_obj][0]['bbox']
            conf = preds[det_obj][0]['confidence_score']

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
        }
        if using_contact:
            new_pred['obj_obj_contact_state'] = obj_obj_contact_state
            new_pred['obj_hand_contact_state'] = obj_hand_contact_state

        preds[obj] = [new_pred]
    
    return preds, replaced

def find_closest_hands(object_pair, detected_classes, preds):
    # Determine what the hand label is in the video, if any
    # Fixes case where hand label has distinguishing information
    # ex: hand(right) vs hand (left)

    hand_labels = [h for h in detected_classes if 'hand' in h.lower()]
    
    if len(hand_labels) == 0:
        return None
    return hand_labels
    
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


def filter_kwcoco(dset, split):
    experiment_name = 'm2_all_data_cleaned_fixed_with_steps'
    stage = 'stage2'

    print('Experiment: ', experiment_name)
    print('Stage: ', stage)

    if type(dset) == str:
        print(f'Loaded dset from file: {dset}')
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    # Remove in-between categories
    remove_cats = []
    for cat_id in dset.cats:
        cat_name = dset.cats[cat_id]['name']
        if '.5)' in cat_name or '(before)' in cat_name or '(finished)' in cat_name:
            remove_cats.append(cat_id)        
            
    print(f'removing cat ids: {remove_cats}')
    dset.remove_categories(remove_cats)

    # Remove images with these 
    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_images = []
    remove_anns = []
    for gid in sorted(gids):
        im = dset.imgs[gid]
        
        fn = im['file_name'].split('/')[-1]
        gt = im['activity_gt']

        if gt == 'not started' or 'in between' in gt or gt == 'finished':
            remove_images.append(gid)

        """
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann['confidence']
            if conf < 0.4:
                remove_anns.append(aid)
        """

    #print(f'removing {len(remove_anns)} annotations')       
    #dset.remove_annotations(remove_anns)

    print(f'removing {len(remove_images)} images (and associated annotations)')
    dset.remove_images(remove_images)

    # Save to a new dataset to adjust ids
    new_dset = kwcoco.CocoDataset()
    new_cats = [{"id": 1, "name": "tourniquet_tourniquet (step 1)"}, {"id": 2, "name": "tourniquet_tourniquet (step 2)"}, {"id": 3, "name": "tourniquet_tourniquet (step 3)"}, {"id": 4, "name": "tourniquet_tourniquet (step 4)"}, {"id": 5, "name": "tourniquet_tourniquet (step 5)"}, {"id": 6, "name": "tourniquet_tourniquet (step 6)"}, {"id": 7, "name": "tourniquet_tourniquet (step 7)"}, {"id": 8, "name": "tourniquet_tourniquet (step 8)"}, {"id": 9, "name": "tourniquet_label (step 1)"}, {"id": 10, "name": "tourniquet_label (step 2)"}, {"id": 11, "name": "tourniquet_label (step 3)"}, {"id": 12, "name": "tourniquet_label (step 4)"}, {"id": 13, "name": "tourniquet_label (step 5)"}, {"id": 14, "name": "tourniquet_label (step 6)"}, {"id": 15, "name": "tourniquet_label (step 7)"}, {"id": 16, "name": "tourniquet_label (step 8)"}, {"id": 17, "name": "tourniquet_windlass (step 1)"}, {"id": 18, "name": "tourniquet_windlass (step 2)"}, {"id": 19, "name": "tourniquet_windlass (step 3)"}, {"id": 20, "name": "tourniquet_windlass (step 4)"}, {"id": 21, "name": "tourniquet_windlass (step 5)"}, {"id": 22, "name": "tourniquet_windlass (step 6)"}, {"id": 23, "name": "tourniquet_windlass (step 7)"}, {"id": 24, "name": "tourniquet_windlass (step 8)"}, {"id": 25, "name": "tourniquet_pen (step 1)"}, {"id": 26, "name": "tourniquet_pen (step 2)"}, {"id": 27, "name": "tourniquet_pen (step 3)"}, {"id": 28, "name": "tourniquet_pen (step 4)"}, {"id": 29, "name": "tourniquet_pen (step 5)"}, {"id": 30, "name": "tourniquet_pen (step 6)"}, {"id": 31, "name": "tourniquet_pen (step 7)"}, {"id": 32, "name": "tourniquet_pen (step 8)"}, {"id": 33, "name": "hand (step 1)"}, {"id": 34, "name": "hand (step 2)"}, {"id": 35, "name": "hand (step 3)"}, {"id": 36, "name": "hand (step 4)"}, {"id": 37, "name": "hand (step 5)"}, {"id": 38, "name": "hand (step 6)"}, {"id": 39, "name": "hand (step 7)"}, {"id": 40, "name": "hand (step 8)"}]
    for new_cat in new_cats:
        new_dset.add_category(name=new_cat['name'], id=new_cat['id'])

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        old_video = dset.index.videos[im['video_id']]['name']
        new_video = new_dset.index.name_to_video[old_video]

        del new_im['id']
        new_im['video_id'] = new_video['id']
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann['category_id']]['name']
            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()
            del new_ann['id']
            new_ann['category_id'] = new_cat['id']
            new_ann['image_id'] = new_gid

            new_dset.add_annotation(**new_ann)

    new_dset.fpath = f'{experiment_name}_{stage}_{split}.mscoco.json'
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f'Saved predictions to {new_dset.fpath}')



def filter_kwcoco_conf_by_video(dset):
    if type(dset) == str:
        print(f'Loaded dset from file: {dset}')
        dset = kwcoco.CocoDataset(dset)
        print(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_anns =[]
    for gid in sorted(gids):
        im = dset.imgs[gid]

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann['confidence']
            video_id = im['video_id']
            video = dset.index.videos[video_id]['name']

            if 'kitware' in video:
                continue
            else:
                # filter the BBN videos by 0.4 conf 
                if conf < 0.4:
                    remove_anns.append(aid)

    print(f'removing {len(remove_anns)} annotations')       
    dset.remove_annotations(remove_anns)

    dset.dump(dset.fpath, newlines=True)
