import os
from .MC50 import register_MC50_instances


# cooking_data_root = '/angel_workspace/angel_system/berkeley/'
#cooking_data_root ='/media/hannah.defazio/Padlock_DT6/Data/notpublic/PTG'
cooking_im_data_root = '/data/PTG/cooking/images'
cooking_ann_data_root = '/data/PTG/cooking/object_anns'
_PREDEFINED_SPLITS_KITWARE_COOKING = {
    "KITWARE_COOKING_COFFEE": {
        "COFFEE_UCB_train": (
            "",
            f"{cooking_ann_data_root}/coffee/berkeley/fine-tuning_plus_bkgd.mscoco.json"
            #f"{cooking_ann_data_root}/coffee/berkeley/coffee_obj_annotations_v2.3_plus_bkgd.mscoco.json"
        ),
        "COFFEE_UCB_val": (
            "",
            f"{cooking_ann_data_root}/coffee/berkeley/fine-tuning_plus_bkgd.mscoco.json"
            #f"{cooking_ann_data_root}/coffee/berkeley/coffee_obj_annotations_v2.3_plus_bkgd.mscoco.json"
        ),
    
        # Generated annotations on videos with contact metadata
        "COFFEE_train": ("", f"{cooking_ann_data_root}/Coffee/annotations/coffee_no_background/coffee_no_background_stage2_train_contact.mscoco.json"),
        "COFFEE_val": ("", f"{cooking_ann_data_root}/Coffee/annotations/coffee_no_background/coffee_no_background_stage2_val.mscoco.json"),
    },
    "KITWARE_COOKING_COFFEE_TEA": {
        "COFFEE_TEA_UCB_train": (
            "",
            f"{cooking_ann_data_root}/coffee+tea/berkeley/coffee_v2.3_and_tea_v2.2_obj_annotations_plus_bkgd.mscoco.json"
        ),
        "COFFEE_TEA_UCB_val": (
            "",
            f"{cooking_ann_data_root}/coffee+tea/berkeley/coffee_v2.3_and_tea_v2.2_obj_annotations_plus_bkgd.mscoco.json"
        )
    }
}

ALL_COOKING_CATEGORIES = {
    # original
    'KITWARE_COOKING_COFFEE': [{'id': 1, 'name': 'coffee + mug'}, {'id': 2, 'name': 'coffee bag'}, {'id': 3, 'name': 'coffee beans + container'}, {'id': 4, 'name': 'coffee beans + container + scale'}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone'}, {'id': 6, 'name': 'coffee grounds + paper filter + filter cone + mug'}, {'id': 7, 'name': 'container'}, {'id': 8, 'name': 'container + scale'}, {'id': 9, 'name': 'filter cone'}, {'id': 10, 'name': 'filter cone + mug'}, {'id': 11, 'name': 'grinder (close)'}, {'id': 12, 'name': 'grinder (open)'}, {'id': 13, 'name': 'hand (left)'}, {'id': 14, 'name': 'hand (right)'}, {'id': 15, 'name': 'kettle'}, {'id': 16, 'name': 'kettle (open)'}, {'id': 17, 'name': 'lid (grinder)'}, {'id': 18, 'name': 'lid (kettle)'}, {'id': 19, 'name': 'measuring cup'}, {'id': 20, 'name': 'mug'}, {'id': 21, 'name': 'paper filter'}, {'id': 22, 'name': 'paper filter (quarter)'}, {'id': 23, 'name': 'paper filter (semi)'}, {'id': 24, 'name': 'paper filter + filter cone'}, {'id': 25, 'name': 'paper filter + filter cone + mug'}, {'id': 26, 'name': 'paper filter bag'}, {'id': 27, 'name': 'paper towel'}, {'id': 28, 'name': 'scale (off)'}, {'id': 29, 'name': 'scale (on)'}, {'id': 30, 'name': 'switch'}, {'id': 31, 'name': 'thermometer (close)'}, {'id': 32, 'name': 'thermometer (open)'}, {'id': 33, 'name': 'timer'}, {'id': 34, 'name': 'timer (20)'}, {'id': 35, 'name': 'timer (30)'}, {'id': 36, 'name': 'timer (else)'}, {'id': 37, 'name': 'trash can'}, {'id': 38, 'name': 'used paper filter'}, {'id': 39, 'name': 'used paper filter + filter cone'}, {'id': 40, 'name': 'used paper filter + filter cone + mug'}, {'id': 41, 'name': 'water'}, {'id': 42, 'name': 'water + coffee grounds + paper filter + filter cone + mug'}],
    # v2.3
    #'KITWARE_COOKING_COFFEE': [{"id": 1, "name": "hand (left)"},{"id": 2, "name": "hand (right)"},{"id": 3, "name": "paper towel"},{"id": 4, "name": "paper towel sheet"},{"id": 5, "name": "water jug lid"},{"id": 6, "name": "water jug (open)"},{"id": 7, "name": "water jug (closed)"},{"id": 8, "name": "kettle lid"},{"id": 9, "name": "kettle (open)"},{"id": 10, "name": "kettle (closed)"},{"id": 11, "name": "measuring cup"},{"id": 12, "name": "container"},{"id": 13, "name": "mug"},{"id": 14, "name": "kettle switch"},{"id": 15, "name": "thermometer (open)"},{"id": 16, "name": "thermometer (closed)"},{"id": 17, "name": "timer (on)"},{"id": 18, "name": "timer (off)"},{"id": 19, "name": "trash can"},{"id": 20, "name": "coffee + mug"},{"id": 21, "name": "coffee bag"},{"id": 22, "name": "coffee beans + container"},{"id": 23, "name": "coffee beans + container + scale"},{"id": 24, "name": "coffee grounds + paper filter (quarter - open) + dripper"},{"id": 25, "name": "coffee grounds + paper filter (quarter - open) + dripper + mug"},{"id": 26, "name": "container + scale"},{"id": 27, "name": "dripper"},{"id": 28, "name": "dripper + mug"},{"id": 29, "name": "grinder lid"},{"id": 30, "name": "grinder (open)"},{"id": 31, "name": "grinder (closed)"},{"id": 32, "name": "paper filter"},{"id": 33, "name": "paper filter (semi)"},{"id": 34, "name": "paper filter (quarter)"},{"id": 35, "name": "used paper filter (quarter) + dripper"},{"id": 36, "name": "paper filter (quarter) + dripper"},{"id": 37, "name": "paper filter (quarter) + dripper + mug"},{"id": 38, "name": "paper filter (quarter - open) + dripper"},{"id": 39, "name": "paper filter (quarter - open) + dripper + mug"},{"id": 40, "name": "paper filter bag"},{"id": 41, "name": "stack of paper filters"},{"id": 42, "name": "scale (on)"},{"id": 43, "name": "scale (off)"},{"id": 44, "name": "used paper filter"},{"id": 45, "name": "used paper filter (quarter - open) + dripper"},{"id": 46, "name": "used paper filter (quarter - open) + dripper + mug"},{"id": 47, "name": "used paper filter (quarter) + dripper + mug"}],
    'KITWARE_COOKING_COFFEE_TEA': [{"id": 1, "name": "hand (left)"},{"id": 2, "name": "hand (right)"},{"id": 3, "name": "paper towel"},{"id": 4, "name": "paper towel sheet"},{"id": 5, "name": "water jug lid"},{"id": 6, "name": "water jug (open)"},{"id": 7, "name": "water jug (closed)"},{"id": 8, "name": "kettle lid"},{"id": 9, "name": "kettle (open)"},{"id": 10, "name": "kettle (closed)"},{"id": 11, "name": "measuring cup"},{"id": 12, "name": "container"},{"id": 13, "name": "mug"},{"id": 14, "name": "kettle switch"},{"id": 15, "name": "thermometer (open)"},{"id": 16, "name": "thermometer (closed)"},{"id": 17, "name": "timer (on)"},{"id": 18, "name": "timer (off)"},{"id": 19, "name": "trash can"},{"id": 20, "name": "coffee + mug"},{"id": 21, "name": "coffee bag"},{"id": 22, "name": "coffee beans + container"},{"id": 23, "name": "coffee beans + container + scale"},{"id": 24, "name": "coffee grounds + paper filter (quarter - open) + dripper"},{"id": 25, "name": "coffee grounds + paper filter (quarter - open) + dripper + mug"},{"id": 26, "name": "container + scale"},{"id": 27, "name": "dripper"},{"id": 28, "name": "dripper + mug"},{"id": 29, "name": "grinder lid"},{"id": 30, "name": "grinder (open)"},{"id": 31, "name": "grinder (closed)"},{"id": 32, "name": "paper filter"},{"id": 33, "name": "paper filter (semi)"},{"id": 34, "name": "paper filter (quarter)"},{"id": 35, "name": "used paper filter (quarter) + dripper"},{"id": 36, "name": "paper filter (quarter) + dripper"},{"id": 37, "name": "paper filter (quarter) + dripper + mug"},{"id": 38, "name": "paper filter (quarter - open) + dripper"},{"id": 39, "name": "paper filter (quarter - open) + dripper + mug"},{"id": 40, "name": "paper filter bag"},{"id": 41, "name": "stack of paper filters"},{"id": 42, "name": "scale (on)"},{"id": 43, "name": "scale (off)"},{"id": 44, "name": "used paper filter"},{"id": 45, "name": "used paper filter (quarter - open) + dripper"},{"id": 46, "name": "used paper filter (quarter - open) + dripper + mug"},{"id": 47, "name": "used paper filter (quarter) + dripper + mug"},{"id": 48, "name": "spoon"},{"id": 49, "name": "jar of honey (open)"},{"id": 50, "name": "jar of honey (closed)"},{"id": 51, "name": "tea bag (dry)"},{"id": 52, "name": "tea bag (wet)"},{"id": 53, "name": "mug + tea bag"},{"id": 54, "name": "mug + water"},{"id": 55, "name": "mug + tea bag + water"},{"id": 56, "name": "mug + tea"}]
}

def _get_cooking_instances_meta_v1(dataset_name):
    # assert len(LVIS_V1_CATEGORIES) == 20
    MC50_CATEGORIES = ALL_COOKING_CATEGORIES[dataset_name]
    
    cat_ids = [k["id"] for k in MC50_CATEGORIES]
    # Double check we have the right number of object ids
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"

    # Ensure that the category list is sorted by id
    MC50_categories = sorted(MC50_CATEGORIES, key=lambda x: x["id"])
    # thing_classes = [k["synonyms"][0] for k in lvis_categories]
    thing_classes = [k["name"] for k in MC50_categories]

    if dataset_name == 'KITWARE_COOKING_COFFEE':
        from angel_system.berkeley.data.objects import coffee_activity_objects

        meta = {
            "thing_classes": thing_classes,
            "sub_steps": coffee_activity_objects.sub_steps,
            "original_sub_steps": coffee_activity_objects.original_sub_steps,
            "contact_pairs_details": coffee_activity_objects.contact_pairs_details,
            "CONTACT_PAIRS": coffee_activity_objects.CONTACT_PAIRS,
            "States_Pairs": coffee_activity_objects.States_Pairs,
            "R_class": coffee_activity_objects.R_class,
            "allow_repeat_obj": coffee_activity_objects.allow_repeat_obj
        }
        
    elif dataset_name == 'KITWARE_COOKING_COFFEE_TEA':
        from angel_system.berkeley.data.objects import coffee_activity_objects
        from angel_system.berkeley.data.objects import tea_activity_objects

        all_sub_steps = coffee_activity_objects.sub_steps.copy()
        for k, v in tea_activity_objects.sub_steps.items():
            all_sub_steps[k].extend(v)

        all_original_sub_steps = coffee_activity_objects.original_sub_steps.copy()
        for k, v in tea_activity_objects.original_sub_steps.items():
            all_original_sub_steps[k].extend(v)

        meta = {
            "thing_classes": thing_classes,
            "sub_steps": all_sub_steps,
            "original_sub_steps": all_original_sub_steps,
            "contact_pairs_details": coffee_activity_objects.contact_pairs_details + tea_activity_objects.contact_pairs_details,
            "CONTACT_PAIRS": coffee_activity_objects.CONTACT_PAIRS + tea_activity_objects.CONTACT_PAIRS,
            "States_Pairs": coffee_activity_objects.States_Pairs + tea_activity_objects.States_Pairs,
            "R_class": coffee_activity_objects.R_class + tea_activity_objects.R_class,
            "allow_repeat_obj": coffee_activity_objects.allow_repeat_obj + tea_activity_objects.allow_repeat_obj
        }
    
    return meta

def register_all_kitware_cooking_data():
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_KITWARE_COOKING.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_MC50_instances(
                key,
                _get_cooking_instances_meta_v1(dataset_name),
                json_file,
                image_root,
            )

