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
    # TODO: Update for multiple hand outputs
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

