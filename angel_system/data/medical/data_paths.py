###############
# GYGES
###############
ptg_root = "/home/local/KHQ/hannah.defazio/angel_system/"
activity_config_path = f"{ptg_root}/config/activity_labels/medical"
object_config_path = f"{ptg_root}/config/object_labels/medical"

data_dir = "/data/PTG/medical/"
activity_gt_dir = f"{data_dir}/activity_anns"
objects_dir = f"{data_dir}/object_anns"
ros_bags_dir = f"{data_dir}/ros_bags/"
bbn_data_dir = f"/data/PTG/medical/bbn_data"

# M2
# ------
m2_activity_gt_dir = f"{activity_gt_dir}/m2_labels/"
m2_activity_config_fn = f"{activity_config_path}/recipe_m2.yaml"

m2_ros_bags_dir = f"{ros_bags_dir}/m2/m2_extracted/"
m2_bbn_data_dir = f"{bbn_data_dir}/Release_v0.5/v0.52/M2_Tourniquet/Data"

m2_training_split = {
    "train_activity": [
        f"{m2_bbn_data_dir}/M2-{x}"
        for x in [1, 2, 4, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30,
            31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            56, 57, 58, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            80, 81, 82, 83, 85, 86, 87, 89, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104,
            105, 107, 108, 112, 113, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124,
            125, 126, 127, 129, 131, 132, 133, 134, 135, 136
        ]
    ],
    "val": [ # GSP testing
        f"{m2_bbn_data_dir}/M2-{x}" for x in [5, 6, 15, 24, 28, 37, 44, 59, 78, 79, 88, 90, 106, 110, 111, 121, 130, 138]
    ],
    "test": [ # GSP training
        f"{m2_bbn_data_dir}/M2-{x}"
        for x in [3, 14, 25, 55, 62, 84, 96, 109, 128, 137, 139]
    ],
}

m2_obj_dets_dir = f"{objects_dir}/m2"
m2_obj_config = f"{object_config_path}/task_m2.yaml"


def grab_data(recipe, machine):
    if machine == "gyges":
        if recipe == "m2":
            return (
                ptg_root,
                data_dir,
                m2_activity_config_fn,
                m2_activity_gt_dir,
                m2_ros_bags_dir,
                m2_training_split,
                m2_obj_dets_dir,
                m2_obj_config,
            )
        
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
