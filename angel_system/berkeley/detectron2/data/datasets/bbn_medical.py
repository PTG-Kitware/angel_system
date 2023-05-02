import os
from .MC50 import get_MC50_instances_meta, register_MC50_instances


bbn_root = '/data/ptg/medical/bbn/'
anns_root = f'{bbn_root}/annotations/'
_PREDEFINED_SPLITS_BBN = {
    "BBN_M2": {
        "BBN_M2_YoloModel_train": ("", f"{anns_root}/M2_Tourniquet/stage1/M2_YoloModel_LO_stage1_train.mscoco.json"),
        "BBN_M2_YoloModel_val": ("", f"{anns_root}/M2_Tourniquet/stage1/M2_YoloModel_LO_stage_test.mscoco.json"),
        
        "BBN_M2_train": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_data/M2_with_lab_preds_cleaned_data_train_contact.mscoco.json"),
        "BBN_M2_val": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_data/M2_with_lab_preds_cleaned_data_val.mscoco.json"),

        "BBN_M2_labels_with_steps_train": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_fixed_data_with_steps/M2_with_lab_cleaned_fixed_data_with_steps_stage2_train_contact.mscoco.json"),
        "BBN_M2_labels_with_steps_val": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_fixed_data_with_steps/M2_with_lab_cleaned_fixed_data_with_steps_stage2_val.mscoco.json"),
        
        "BBN_M2_labels_with_inter_steps_train": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_fixed_data_with_inter_steps/M2_with_lab_cleaned_fixed_data_with_inter_steps_stage2_train_contact.mscoco.json"),
        "BBN_M2_labels_with_inter_steps_val": ("", f"{anns_root}/M2_Tourniquet/stage2/m2_with_lab_cleaned_fixed_data_with_inter_steps/M2_with_lab_cleaned_fixed_data_with_inter_steps_stage2_val.mscoco.json"),
        
    },
}


ALL_BBN_CATEGORIES = {
    "BBN_M2": {
        "BBN_M2_YoloModel": [{"id": 1, "name": "tourniquet_tourniquet"}, {"id": 2, "name": "tourniquet_label"}, {"id": 3, "name": "tourniquet_windlass"}, {"id": 4, "name": "tourniquet_pen"}, {"id": 5, "name": "hand"}],
        "BBN_M2": [{"id": 1, "name": "tourniquet_tourniquet"}, {"id": 2, "name": "tourniquet_label"}, {"id": 3, "name": "tourniquet_windlass"}, {"id": 4, "name": "tourniquet_pen"}, {"id": 5, "name": "hand"}],
        "BBN_M2_labels_with_steps": [{"id": 1, "name": "tourniquet_tourniquet (step 1)"}, {"id": 2, "name": "tourniquet_tourniquet (step 2)"}, {"id": 3, "name": "tourniquet_tourniquet (step 3)"}, {"id": 4, "name": "tourniquet_tourniquet (step 4)"}, {"id": 5, "name": "tourniquet_tourniquet (step 5)"}, {"id": 6, "name": "tourniquet_tourniquet (step 6)"}, {"id": 7, "name": "tourniquet_tourniquet (step 7)"}, {"id": 8, "name": "tourniquet_tourniquet (step 8)"}, {"id": 9, "name": "tourniquet_label (step 1)"}, {"id": 10, "name": "tourniquet_label (step 2)"}, {"id": 11, "name": "tourniquet_label (step 3)"}, {"id": 12, "name": "tourniquet_label (step 4)"}, {"id": 13, "name": "tourniquet_label (step 5)"}, {"id": 14, "name": "tourniquet_label (step 6)"}, {"id": 15, "name": "tourniquet_label (step 7)"}, {"id": 16, "name": "tourniquet_label (step 8)"}, {"id": 17, "name": "tourniquet_windlass (step 1)"}, {"id": 18, "name": "tourniquet_windlass (step 2)"}, {"id": 19, "name": "tourniquet_windlass (step 3)"}, {"id": 20, "name": "tourniquet_windlass (step 4)"}, {"id": 21, "name": "tourniquet_windlass (step 5)"}, {"id": 22, "name": "tourniquet_windlass (step 6)"}, {"id": 23, "name": "tourniquet_windlass (step 7)"}, {"id": 24, "name": "tourniquet_windlass (step 8)"}, {"id": 25, "name": "tourniquet_pen (step 1)"}, {"id": 26, "name": "tourniquet_pen (step 2)"}, {"id": 27, "name": "tourniquet_pen (step 3)"}, {"id": 28, "name": "tourniquet_pen (step 4)"}, {"id": 29, "name": "tourniquet_pen (step 5)"}, {"id": 30, "name": "tourniquet_pen (step 6)"}, {"id": 31, "name": "tourniquet_pen (step 7)"}, {"id": 32, "name": "tourniquet_pen (step 8)"}, {"id": 33, "name": "hand (step 1)"}, {"id": 34, "name": "hand (step 2)"}, {"id": 35, "name": "hand (step 3)"}, {"id": 36, "name": "hand (step 4)"}, {"id": 37, "name": "hand (step 5)"}, {"id": 38, "name": "hand (step 6)"}, {"id": 39, "name": "hand (step 7)"}, {"id": 40, "name": "hand (step 8)"}],
        "BBN_M2_labels_with_inter_steps": [{"id": 1, "name": "tourniquet_tourniquet (step 1)"},{"id": 2, "name": "tourniquet_tourniquet (step 1.5)"},{"id": 3, "name": "tourniquet_tourniquet (step 2)"},{"id": 4, "name": "tourniquet_tourniquet (step 2.5)"},{"id": 5, "name": "tourniquet_tourniquet (step 3)"},{"id": 6, "name": "tourniquet_tourniquet (step 3.5)"},{"id": 7, "name": "tourniquet_tourniquet (step 4)"},{"id": 8, "name": "tourniquet_tourniquet (step 4.5)"},{"id": 9, "name": "tourniquet_tourniquet (step 5)"},{"id": 10, "name": "tourniquet_tourniquet (step 5.5)"},{"id": 11, "name": "tourniquet_tourniquet (step 6)"},{"id": 12, "name": "tourniquet_tourniquet (step 6.5)"},{"id": 13, "name": "tourniquet_tourniquet (step 7)"},{"id": 14, "name": "tourniquet_tourniquet (step 7.5)"},{"id": 15, "name": "tourniquet_tourniquet (step 8)"},{"id": 16, "name": "tourniquet_label (step 1)"},{"id": 17, "name": "tourniquet_label (step 1.5)"},{"id": 18, "name": "tourniquet_label (step 2)"},{"id": 19, "name": "tourniquet_label (step 2.5)"},{"id": 20, "name": "tourniquet_label (step 3)"},{"id": 21, "name": "tourniquet_label (step 3.5)"},{"id": 22, "name": "tourniquet_label (step 4)"},{"id": 23, "name": "tourniquet_label (step 4.5)"},{"id": 24, "name": "tourniquet_label (step 5)"},{"id": 25, "name": "tourniquet_label (step 5.5)"},{"id": 26, "name": "tourniquet_label (step 6)"},{"id": 27, "name": "tourniquet_label (step 6.5)"},{"id": 28, "name": "tourniquet_label (step 7)"},{"id": 29, "name": "tourniquet_label (step 7.5)"},{"id": 30, "name": "tourniquet_label (step 8)"},{"id": 31, "name": "tourniquet_windlass (step 1)"},{"id": 32, "name": "tourniquet_windlass (step 1.5)"},{"id": 33, "name": "tourniquet_windlass (step 2)"},{"id": 34, "name": "tourniquet_windlass (step 2.5)"},{"id": 35, "name": "tourniquet_windlass (step 3)"},{"id": 36, "name": "tourniquet_windlass (step 3.5)"},{"id": 37, "name": "tourniquet_windlass (step 4)"},{"id": 38, "name": "tourniquet_windlass (step 4.5)"},{"id": 39, "name": "tourniquet_windlass (step 5)"},{"id": 40, "name": "tourniquet_windlass (step 5.5)"},{"id": 41, "name": "tourniquet_windlass (step 6)"},{"id": 42, "name": "tourniquet_windlass (step 6.5)"},{"id": 43, "name": "tourniquet_windlass (step 7)"},{"id": 44, "name": "tourniquet_windlass (step 7.5)"},{"id": 45, "name": "tourniquet_windlass (step 8)"},{"id": 46, "name": "tourniquet_pen (step 1)"},{"id": 47, "name": "tourniquet_pen (step 1.5)"},{"id": 48, "name": "tourniquet_pen (step 2)"},{"id": 49, "name": "tourniquet_pen (step 2.5)"},{"id": 50, "name": "tourniquet_pen (step 3)"},{"id": 51, "name": "tourniquet_pen (step 3.5)"},{"id": 52, "name": "tourniquet_pen (step 4)"},{"id": 53, "name": "tourniquet_pen (step 4.5)"},{"id": 54, "name": "tourniquet_pen (step 5)"},{"id": 55, "name": "tourniquet_pen (step 5.5)"},{"id": 56, "name": "tourniquet_pen (step 6)"},{"id": 57, "name": "tourniquet_pen (step 6.5)"},{"id": 58, "name": "tourniquet_pen (step 7)"},{"id": 59, "name": "tourniquet_pen (step 7.5)"},{"id": 60, "name": "tourniquet_pen (step 8)"},{"id": 61, "name": "hand (step 1)"},{"id": 62, "name": "hand (step 1.5)"},{"id": 63, "name": "hand (step 2)"},{"id": 64, "name": "hand (step 2.5)"},{"id": 65, "name": "hand (step 3)"},{"id": 66, "name": "hand (step 3.5)"},{"id": 67, "name": "hand (step 4)"},{"id": 68, "name": "hand (step 4.5)"},{"id": 69, "name": "hand (step 5)"},{"id": 70, "name": "hand (step 5.5)"},{"id": 71, "name": "hand (step 6)"},{"id": 72, "name": "hand (step 6.5)"},{"id": 73, "name": "hand (step 7)"},{"id": 74, "name": "hand (step 7.5)"},{"id": 75, "name": "hand (step 8)"}]
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
