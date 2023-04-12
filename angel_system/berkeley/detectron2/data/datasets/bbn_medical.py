import os
from .MC50 import get_MC50_instances_meta, register_MC50_instances


bbn_root = '/Padlock_DT/Release_v0.5'
_PREDEFINED_SPLITS_BBN = {
    "BBN_M2": {
        "BBN_M2_train": (f"{bbn_root}", f"{bbn_root}/M1_hands_and_M2_train.mscoco.json"),
        "BBN_M2_val": (f"{bbn_root}", f"{bbn_root}/M1_hands_and_M2_train.mscoco.json"),
    },
}


ALL_BBN_CATEGORIES = {
    'BBN_M2': [{'id': 1, 'name': 'USER_HAND', 'instances_count': 562, 'def': '', 'synonyms': [], 'image_count': 562, 'frequency': '', 'synset': ''}, {'id': 2, 'name': 'tourniquet_tourniquet', 'instances_count': 3459, 'def': '', 'synonyms': [], 'image_count': 3459, 'frequency': '', 'synset': ''}, {'id': 3, 'name': 'tourniquet_label', 'instances_count': 2495, 'def': '', 'synonyms': [], 'image_count': 2495, 'frequency': '', 'synset': ''}, {'id': 4, 'name': 'tourniquet_windlass', 'instances_count': 2145, 'def': '', 'synonyms': [], 'image_count': 2145, 'frequency': '', 'synset': ''}, {'id': 5, 'name': 'tourniquet_pen', 'instances_count': 1256, 'def': '', 'synonyms': [], 'image_count': 1256, 'frequency': '', 'synset': ''}]
}

def _get_bbn_instances_meta_v1(dataset_name):
    # assert len(LVIS_V1_CATEGORIES) == 20
    MC50_CATEGORIES = ALL_BBN_CATEGORIES[dataset_name]
    cat_ids = [k["id"] for k in MC50_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    MC50_categories = sorted(MC50_CATEGORIES, key=lambda x: x["id"])
    # thing_classes = [k["synonyms"][0] for k in lvis_categories]
    thing_classes = [k["name"] for k in MC50_categories]

    if dataset_name == 'BBN_M2':
        import utils.data.objects.bbn_M2_tourniquet_activity_objects as bbn_activity_objects

    meta = {
        "thing_classes": thing_classes,
        "class_image_count": MC50_CATEGORIES,
        "sub_steps": bbn_activity_objects.sub_steps,
        "contact_pairs_details": bbn_activity_objects.contact_pairs_details,
        "CONTACT_PAIRS": bbn_activity_objects.CONTACT_PAIRS,
        "States_Pairs": bbn_activity_objects.States_Pairs
    }
    return meta

def register_all_bbn_data():
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_BBN.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_MC50_instances(
                key,
                _get_bbn_instances_meta_v1(dataset_name),
                json_file,
                image_root,
            )
