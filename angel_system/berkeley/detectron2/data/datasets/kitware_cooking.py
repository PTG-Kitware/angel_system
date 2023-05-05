import os
from .MC50 import register_MC50_instances


cooking_data_root = '/angel_workspace/angel_system/berkeley/'
_PREDEFINED_SPLITS_KITWARE_COOKING = {
    "KITWARE_COOKING_COFFEE": {
        "COFFEE_UCB_train": (f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/images", f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/fine-tuning.json"),
        "COFFEE_UCB_val": (f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/images", f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/fine-tuning.json"),
    
        # Generated annotations on videos with contact metadata
        "COFFEE_train": (f"{cooking_data_root}/coffee_recordings/extracted", f"{cooking_data_root}/coffee_recordings/extracted/coffee_preds_with_all_objects_train.mscoco.json"),
        "COFFEE_train_with_background": (f"{cooking_data_root}/coffee_recordings/extracted", f"{cooking_data_root}/coffee_recordings/extracted/coffee_contact_preds_with_background_all_objs_only_train.mscoco.json"),
        "COFFEE_val": (f"{cooking_data_root}/coffee_recordings/extracted", f"{cooking_data_root}/coffee_recordings/extracted/coffee_preds_with_all_objects_val.mscoco.json"),
    },
}

ALL_COOOKING_CATEGORIES = {
    'KITWARE_COOKING_COFFEE': [{'id': 1, 'name': 'coffee + mug'}, {'id': 2, 'name': 'coffee bag'}, {'id': 3, 'name': 'coffee beans + container'}, {'id': 4, 'name': 'coffee beans + container + scale'}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone'}, {'id': 6, 'name': 'coffee grounds + paper filter + filter cone + mug'}, {'id': 7, 'name': 'container'}, {'id': 8, 'name': 'container + scale'}, {'id': 9, 'name': 'filter cone'}, {'id': 10, 'name': 'filter cone + mug'}, {'id': 11, 'name': 'grinder (close)'}, {'id': 12, 'name': 'grinder (open)'}, {'id': 13, 'name': 'hand (left)'}, {'id': 14, 'name': 'hand (right)'}, {'id': 15, 'name': 'kettle'}, {'id': 16, 'name': 'kettle (open)'}, {'id': 17, 'name': 'lid (grinder)'}, {'id': 18, 'name': 'lid (kettle)'}, {'id': 19, 'name': 'measuring cup'}, {'id': 20, 'name': 'mug'}, {'id': 21, 'name': 'paper filter'}, {'id': 22, 'name': 'paper filter (quarter)'}, {'id': 23, 'name': 'paper filter (semi)'}, {'id': 24, 'name': 'paper filter + filter cone'}, {'id': 25, 'name': 'paper filter + filter cone + mug'}, {'id': 26, 'name': 'paper filter bag'}, {'id': 27, 'name': 'paper towel'}, {'id': 28, 'name': 'scale (off)'}, {'id': 29, 'name': 'scale (on)'}, {'id': 30, 'name': 'switch'}, {'id': 31, 'name': 'thermometer (close)'}, {'id': 32, 'name': 'thermometer (open)'}, {'id': 33, 'name': 'timer'}, {'id': 34, 'name': 'timer (20)'}, {'id': 35, 'name': 'timer (30)'}, {'id': 36, 'name': 'timer (else)'}, {'id': 37, 'name': 'trash can'}, {'id': 38, 'name': 'used paper filter'}, {'id': 39, 'name': 'used paper filter + filter cone'}, {'id': 40, 'name': 'used paper filter + filter cone + mug'}, {'id': 41, 'name': 'water'}, {'id': 42, 'name': 'water + coffee grounds + paper filter + filter cone + mug'}]
}

def _get_cooking_instances_meta_v1(dataset_name):
    # assert len(LVIS_V1_CATEGORIES) == 20
    MC50_CATEGORIES = ALL_COOOKING_CATEGORIES[dataset_name]
    cat_ids = [k["id"] for k in MC50_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    MC50_categories = sorted(MC50_CATEGORIES, key=lambda x: x["id"])
    # thing_classes = [k["synonyms"][0] for k in lvis_categories]
    thing_classes = [k["name"] for k in MC50_categories]

    if dataset_name == 'KITWARE_COOKING_COFFEE':
        from angel_system.berkeley.utils.data.objects import coffee_activity_objects as cooking_activity_objects

    meta = {
        "thing_classes": thing_classes,
        #"class_image_count": MC50_CATEGORIES,
        "sub_steps": cooking_activity_objects.sub_steps,
        "original_sub_steps": cooking_activity_objects.original_sub_steps,
        "contact_pairs_details": cooking_activity_objects.contact_pairs_details,
        "CONTACT_PAIRS": cooking_activity_objects.CONTACT_PAIRS,
        "States_Pairs": cooking_activity_objects.States_Pairs
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

