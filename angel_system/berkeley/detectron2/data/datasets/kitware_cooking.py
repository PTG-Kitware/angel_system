import os
from .MC50 import register_MC50_instances


# cooking_data_root = '/angel_workspace/angel_system/berkeley/'
#cooking_data_root ='/media/hannah.defazio/Padlock_DT6/Data/notpublic/PTG'
cooking_im_data_root = '/data/PTG/cooking/images'
cooking_ann_data_root = '/data/PTG/cooking/annotations'
_PREDEFINED_SPLITS_KITWARE_COOKING = {
    "KITWARE_COOKING_COFFEE": {
        "COFFEE_UCB_train": (
            "",
            f"{cooking_ann_data_root}/coffee/berkeley/2022-11-05_whole/fine-tuning_plus_bkgd.mscoco.json"
        ),
        "COFFEE_UCB_val": (
            "",
            f"{cooking_ann_data_root}/coffee/berkeley/2022-11-05_whole/fine-tuning_plus_bkgd.mscoco.json"
        ),
    
        # Generated annotations on videos with contact metadata
        "COFFEE_train": ("", f"{cooking_ann_data_root}/Coffee/annotations/coffee_no_background/coffee_no_background_stage2_train_contact.mscoco.json"),
        "COFFEE_val": ("", f"{cooking_ann_data_root}/Coffee/annotations/coffee_no_background/coffee_no_background_stage2_val.mscoco.json"),
    },
    "KITWARE_COOKING_COFFEE_TEA": {
        "COFFEE_TEA_UCB_train": (
            "",
            f"{cooking_ann_data_root}/coffee+tea/berkeley/coffee_and_tea_obj_annotations.mscoco.json"
        ),
        "COFFEE_TEA_UCB_val": (
            "",
            f"{cooking_ann_data_root}/coffee+tea/berkeley/coffee_and_tea_obj_annotations.mscoco.json"
        )
    }
}

ALL_COOOKING_CATEGORIES = {
    # old order
    'KITWARE_COOKING_COFFEE': [{'id': 1, 'name': 'coffee + mug'}, {'id': 2, 'name': 'coffee bag'}, {'id': 3, 'name': 'coffee beans + container'}, {'id': 4, 'name': 'coffee beans + container + scale'}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone'}, {'id': 6, 'name': 'coffee grounds + paper filter + filter cone + mug'}, {'id': 7, 'name': 'container'}, {'id': 8, 'name': 'container + scale'}, {'id': 9, 'name': 'filter cone'}, {'id': 10, 'name': 'filter cone + mug'}, {'id': 11, 'name': 'grinder (close)'}, {'id': 12, 'name': 'grinder (open)'}, {'id': 13, 'name': 'hand (left)'}, {'id': 14, 'name': 'hand (right)'}, {'id': 15, 'name': 'kettle'}, {'id': 16, 'name': 'kettle (open)'}, {'id': 17, 'name': 'lid (grinder)'}, {'id': 18, 'name': 'lid (kettle)'}, {'id': 19, 'name': 'measuring cup'}, {'id': 20, 'name': 'mug'}, {'id': 21, 'name': 'paper filter'}, {'id': 22, 'name': 'paper filter (quarter)'}, {'id': 23, 'name': 'paper filter (semi)'}, {'id': 24, 'name': 'paper filter + filter cone'}, {'id': 25, 'name': 'paper filter + filter cone + mug'}, {'id': 26, 'name': 'paper filter bag'}, {'id': 27, 'name': 'paper towel'}, {'id': 28, 'name': 'scale (off)'}, {'id': 29, 'name': 'scale (on)'}, {'id': 30, 'name': 'switch'}, {'id': 31, 'name': 'thermometer (close)'}, {'id': 32, 'name': 'thermometer (open)'}, {'id': 33, 'name': 'timer'}, {'id': 34, 'name': 'timer (20)'}, {'id': 35, 'name': 'timer (30)'}, {'id': 36, 'name': 'timer (else)'}, {'id': 37, 'name': 'trash can'}, {'id': 38, 'name': 'used paper filter'}, {'id': 39, 'name': 'used paper filter + filter cone'}, {'id': 40, 'name': 'used paper filter + filter cone + mug'}, {'id': 41, 'name': 'water'}, {'id': 42, 'name': 'water + coffee grounds + paper filter + filter cone + mug'}],
    'KITWARE_COOKING_COFFEE_TEA': [{"id": 1, "name": "hand (left)"},{"id": 2, "name": "hand (right)"},{"id": 3, "name": "paper towel"},{"id": 4, "name": "water"},{"id": 5, "name": "kettle lid"},{"id": 6, "name": "kettle (open)"},{"id": 7, "name": "measuring cup"},{"id": 8, "name": "mug"},{"id": 9, "name": "switch"},{"id": 10, "name": "thermometer (close)"},{"id": 11, "name": "thermometer (open)"},{"id": 12, "name": "timer (on)"},{"id": 13, "name": "timer (off)"},{"id": 14, "name": "kettle (closed)"},{"id": 15, "name": "trash can"},{"id": 16, "name": "coffee + mug"},{"id": 17, "name": "coffee bag"},{"id": 18, "name": "coffee beans + container"},{"id": 19, "name": "coffee beans + container + scale"},{"id": 20, "name": "coffee grounds + paper filter + filter cone"},{"id": 21, "name": "coffee grounds + paper filter + filter cone + mug"},{"id": 22, "name": "container"},{"id": 23, "name": "container + scale"},{"id": 24, "name": "filter cone"},{"id": 25, "name": "filter cone + mug"},{"id": 26, "name": "grinder (close)"},{"id": 27, "name": "grinder (open)"},{"id": 28, "name": "lid (grinder)"},{"id": 29, "name": "lid (kettle)"},{"id": 30, "name": "paper filter"},{"id": 31, "name": "paper filter (quarter)"},{"id": 32, "name": "paper filter (semi)"},{"id": 33, "name": "paper filter + filter cone"},{"id": 34, "name": "paper filter + filter cone + mug"},{"id": 35, "name": "paper filter bag"},{"id": 36, "name": "scale (off)"},{"id": 37, "name": "scale (on)"},{"id": 38, "name": "used paper filter"},{"id": 39, "name": "used paper filter + filter cone"},{"id": 40, "name": "used paper filter + filter cone + mug"},{"id": 41, "name": "water + coffee grounds + paper filter + filter cone + mug"},{"id": 42, "name": "spoon"},{"id": 43, "name": "jar of honey (close)"},{"id": 44, "name": "jar of honey (open)"},{"id": 45, "name": "tea bag"},{"id": 46, "name": "mug + tea bag"},{"id": 47, "name": "mug + water"},{"id": 48, "name": "mug + tea bag + water"},{"id": 49, "name": "mug + tea"}],
    # new order
    # TODO: support non sequential object ids
    #'KITWARE_COOKING_COFFEE':[{"id": 1, "name": "hand (left)"},{"id": 2, "name": "hand (right)"},{"id": 3, "name": "paper towel"},{"id": 4, "name": "water"},{"id": 15, "name": "kettle lid"},{"id": 16, "name": "kettle (open)"},{"id": 17, "name": "measuring cup"},{"id": 18, "name": "mug"},{"id": 19, "name": "switch"},{"id": 20, "name": "thermometer (close)"},{"id": 21, "name": "thermometer (open)"},{"id": 22, "name": "timer (on)"},{"id": 23, "name": "timer (off)"},{"id": 24, "name": "kettle (closed)"},{"id": 26, "name": "trash can"},{"id": 27, "name": "coffee + mug"},{"id": 28, "name": "coffee bag"},{"id": 29, "name": "coffee beans + container"},{"id": 30, "name": "coffee beans + container + scale"},{"id": 31, "name": "coffee grounds + paper filter + filter cone"},{"id": 32, "name": "coffee grounds + paper filter + filter cone + mug"},{"id": 33, "name": "container"},{"id": 34, "name": "container + scale"},{"id": 35, "name": "filter cone"},{"id": 36, "name": "filter cone + mug"},{"id": 37, "name": "grinder (close)"},{"id": 38, "name": "grinder (open)"},{"id": 39, "name": "lid (grinder)"},{"id": 40, "name": "lid (kettle)"},{"id": 41, "name": "paper filter"},{"id": 42, "name": "paper filter (quarter)"},{"id": 43, "name": "paper filter (semi)"},{"id": 44, "name": "paper filter + filter cone"},{"id": 45, "name": "paper filter + filter cone + mug"},{"id": 46, "name": "paper filter bag"},{"id": 47, "name": "scale (off)"},{"id": 48, "name": "scale (on)"},{"id": 50, "name": "used paper filter"},{"id": 51, "name": "used paper filter + filter cone"},{"id": 52, "name": "used paper filter + filter cone + mug"},{"id": 53, "name": "water + coffee grounds + paper filter + filter cone + mug"}],
    #'KITWARE_COOKING_COFFEE_TEA':[{"id": 1, "name": "hand (left)"},{"id": 2, "name": "hand (right)"},{"id": 3, "name": "paper towel"},{"id": 4, "name": "water"},{"id": 12, "name": "spoon"},{"id": 13, "name": "jar of honey (close)"},{"id": 14, "name": "jar of honey (open)"},{"id": 15, "name": "kettle lid"},{"id": 16, "name": "kettle (open)"},{"id": 17, "name": "measuring cup"},{"id": 18, "name": "mug"},{"id": 19, "name": "switch"},{"id": 20, "name": "thermometer (close)"},{"id": 21, "name": "thermometer (open)"},{"id": 22, "name": "timer (on)"},{"id": 23, "name": "timer (off)"},{"id": 24, "name": "kettle (closed)"},{"id": 26, "name": "trash can"},{"id": 27, "name": "coffee + mug"},{"id": 28, "name": "coffee bag"},{"id": 29, "name": "coffee beans + container"},{"id": 30, "name": "coffee beans + container + scale"},{"id": 31, "name": "coffee grounds + paper filter + filter cone"},{"id": 32, "name": "coffee grounds + paper filter + filter cone + mug"},{"id": 33, "name": "container"},{"id": 34, "name": "container + scale"},{"id": 35, "name": "filter cone"},{"id": 36, "name": "filter cone + mug"},{"id": 37, "name": "grinder (close)"},{"id": 38, "name": "grinder (open)"},{"id": 39, "name": "lid (grinder)"},{"id": 40, "name": "lid (kettle)"},{"id": 41, "name": "paper filter"},{"id": 42, "name": "paper filter (quarter)"},{"id": 43, "name": "paper filter (semi)"},{"id": 44, "name": "paper filter + filter cone"},{"id": 45, "name": "paper filter + filter cone + mug"},{"id": 46, "name": "paper filter bag"},{"id": 47, "name": "scale (off)"},{"id": 48, "name": "scale (on)"},{"id": 50, "name": "used paper filter"},{"id": 51, "name": "used paper filter + filter cone"},{"id": 52, "name": "used paper filter + filter cone + mug"},{"id": 53, "name": "water + coffee grounds + paper filter + filter cone + mug"},{"id": 94, "name": "tea bag"},{"id": 95, "name": "mug + tea bag"},{"id": 96, "name": "mug + water"},{"id": 97, "name": "mug + tea bag + water"},{"id": 98, "name": "mug + tea"}]
}

def _get_cooking_instances_meta_v1(dataset_name):
    # assert len(LVIS_V1_CATEGORIES) == 20
    MC50_CATEGORIES = ALL_COOOKING_CATEGORIES[dataset_name]
    
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

