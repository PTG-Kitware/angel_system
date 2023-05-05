import os
from .MC50 import get_MC50_instances_meta, register_MC50_instances


bbn_root = '/data/ptg/medical/bbn/Release_v0.5/'
_PREDEFINED_SPLITS_BBN = {
    "BBN_M2": {
        "BBN_M2_YoloModel_train": (f"{bbn_root}/v0.52/M2_Tourniquet/YoloModel/", f"{bbn_root}/v0.52/M2_Tourniquet/YoloModel/M2_YoloModel_LO_train.mscoco.json"),
        "BBN_M2_YoloModel_val": (f"{bbn_root}/v0.52/M2_Tourniquet/YoloModel/", f"{bbn_root}/v0.52/M2_Tourniquet/YoloModel/M2_YoloModel_LO_test.mscoco.json"),

        "BBN_M2_train": (f"{bbn_root}/v0.52/M2_Tourniquet/Data", f"{bbn_root}/v0.52/M2_Tourniquet/Data/M2_preds_with_all_objects_train.mscoco.json"),
        "BBN_M2_val": (f"{bbn_root}/v0.52/M2_Tourniquet/Data", f"{bbn_root}/v0.52/M2_Tourniquet/Data/M2_preds_with_all_objects_val.mscoco.json"),
    },
}


ALL_BBN_CATEGORIES = {
    "BBN_M2": {
        "BBN_M2_YoloModel": [{"id": 1, "name": "tourniquet_tourniquet"}, {"id": 2, "name": "tourniquet_label"}, {"id": 3, "name": "tourniquet_windlass"}, {"id": 4, "name": "tourniquet_pen"}, {"id": 5, "name": "hand"}],
        "BBN_M2": [{"id": 1, "name": "tourniquet_tourniquet (step 1)"}, {"id": 2, "name": "tourniquet_tourniquet (step 2)"}, {"id": 3, "name": "tourniquet_tourniquet (step 3)"}, {"id": 4, "name": "tourniquet_tourniquet (step 4)"}, {"id": 5, "name": "tourniquet_tourniquet (step 5)"}, {"id": 6, "name": "tourniquet_tourniquet (step 6)"}, {"id": 7, "name": "tourniquet_tourniquet (step 7)"}, {"id": 8, "name": "tourniquet_tourniquet (step 8)"}, {"id": 9, "name": "tourniquet_tourniquet (step 9)"}, {"id": 10, "name": "tourniquet_label (step 1)"}, {"id": 11, "name": "tourniquet_label (step 2)"}, {"id": 12, "name": "tourniquet_label (step 3)"}, {"id": 13, "name": "tourniquet_label (step 4)"}, {"id": 14, "name": "tourniquet_label (step 5)"}, {"id": 15, "name": "tourniquet_label (step 6)"}, {"id": 16, "name": "tourniquet_label (step 7)"}, {"id": 17, "name": "tourniquet_label (step 8)"}, {"id": 18, "name": "tourniquet_label (step 9)"}, {"id": 19, "name": "tourniquet_windlass (step 1)"}, {"id": 20, "name": "tourniquet_windlass (step 2)"}, {"id": 21, "name": "tourniquet_windlass (step 3)"}, {"id": 22, "name": "tourniquet_windlass (step 4)"}, {"id": 23, "name": "tourniquet_windlass (step 5)"}, {"id": 24, "name": "tourniquet_windlass (step 6)"}, {"id": 25, "name": "tourniquet_windlass (step 7)"}, {"id": 26, "name": "tourniquet_windlass (step 8)"}, {"id": 27, "name": "tourniquet_windlass (step 9)"}, {"id": 28, "name": "tourniquet_pen (step 1)"}, {"id": 29, "name": "tourniquet_pen (step 2)"}, {"id": 30, "name": "tourniquet_pen (step 3)"}, {"id": 31, "name": "tourniquet_pen (step 4)"}, {"id": 32, "name": "tourniquet_pen (step 5)"}, {"id": 33, "name": "tourniquet_pen (step 6)"}, {"id": 34, "name": "tourniquet_pen (step 7)"}, {"id": 35, "name": "tourniquet_pen (step 8)"}, {"id": 36, "name": "tourniquet_pen (step 9)"}, {"id": 37, "name": "hand (step 1)"}, {"id": 38, "name": "hand (step 2)"}, {"id": 39, "name": "hand (step 3)"}, {"id": 40, "name": "hand (step 4)"}, {"id": 41, "name": "hand (step 5)"}, {"id": 42, "name": "hand (step 6)"}, {"id": 43, "name": "hand (step 7)"}, {"id": 44, "name": "hand (step 8)"}, {"id": 45, "name": "hand (step 9)"}]
    }
}

def _get_bbn_instances_meta_v1(dataset_name, key):
    # assert len(LVIS_V1_CATEGORIES) == 20
    key = '_'.join(key.split('_')[:-1])
    #print(key)
    MC50_CATEGORIES = ALL_BBN_CATEGORIES[dataset_name][key]
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
        #"class_image_count": MC50_CATEGORIES,
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
                _get_bbn_instances_meta_v1(dataset_name, key),
                json_file,
                image_root,
            )
