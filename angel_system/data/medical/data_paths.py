import os

###############
# GYGES
###############
ptg_root = "/home/local/KHQ/peri.akiva/angel_system/"
activity_config_path = f"{ptg_root}/config/activity_labels/medical"
object_config_path = f"{ptg_root}/config/object_labels/medical"

data_dir = "/data/PTG/medical/"
activity_gt_dir = f"{data_dir}/activity_anns"
objects_dir = f"{data_dir}/object_anns"
ros_bags_dir = f"{data_dir}/ros_bags/"
bbn_data_dir = f"/data/PTG/medical/bbn_data"

KNOWN_BAD_VIDEOS = ["M2-15"]  # Videos without any usable data

TASK_TO_NAME = {
    'm1': "M1_Trauma_Assessment",
    'm2': "M2_Tourniquet",
    'm3': "M3_Pressure_Dressing",
    'm4': "M4_Wound_Packing",
    'm5': "M5_X-Stat",
    'r18': "R18_Chest_Seal",
}

# M2
# ------
m2_activity_gt_dir = f"{activity_gt_dir}/m2_labels/"
m2_activity_config_fn = f"{activity_config_path}/recipe_m2.yaml"

# M3

# M5
# R18 /data/PTG/medical/bbn_data/Release_v0.5/v0.56
bbn_data_root = f"{bbn_data_dir}/Release_v0.5/v0.56"
m2_bbn_data_dir = f"{bbn_data_dir}/Release_v0.5/v0.52/M2_Tourniquet/Data"
m2_training_split = {
    "train_activity": [
        f"{m2_bbn_data_dir}/M2-{x}"
        for x in [1, 7, 13, 19, 21, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40,
            52, 53, 57, 58, 60, 63, 64, 70, 71, 72, 73, 74, 75, 76, 77, 119, 122, 124,
            132, 133,
        ]
        # These videos have multiples of objects during the activities:
        # 2, 4, 8, 9, 10, 11, 12, 16, 17, 18, 20, 22,
        # 23, 41, 42, 43, 45, 46, 47, 48, 49, 50, 51,
        # 54, 56, 61, 65, 66, 67, 68, 69, 80, 81, 82,
        # 83, 85, 86, 87, 89, 91, 92, 93, 94, 95, 97,
        # 98, 99, 100, 101, 102, 103, 104, 105, 107,
        # 108, 112, 113, 114, 115, 116, 117, 118, 120,
        # 123, 125, 126, 127, 129, 131, 134, 135, 136
    ],
    "val": [  # GSP testing
        f"{m2_bbn_data_dir}/M2-{x}"
        for x in [5, 6, 24, 28, 37, 59]
        # 15 is a bad video
        # These videos have multiples of objects during the activities:
        # 44, 78, 79, 88, 90, 106, 110, 111, 121, 130, 138
    ],
    "test": [  # GSP training
        f"{m2_bbn_data_dir}/M2-{x}"
        for x in [25, 55]
        # These videos have multiples of objects during the activities:
        # 3, 14, 62, 84, 96, 109, 128, 137, 139
    ],
}

m2_obj_dets_dir = f"{objects_dir}/m2"
m2_obj_config = f"{object_config_path}/task_m2.yaml"

# M2 Lab
# ------
m2_lab_bbn_data_dir = f"{bbn_data_dir}/M2_Lab_data/skills_by_frame"
m2_lab_training_split = {
    "train_activity": [],
    "val": [],
    "test": [
        f"{m2_lab_bbn_data_dir}/tq_{x}"
        for x in [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30]
        ],
}


# M2 Kitware
# ----------
m2_kitware_data_dir = f"{ros_bags_dir}/M2/M2_extracted/"
m2_kitware_training_split = {
    "train_activity": [],
    "val": [],
    "test": [
        f"{m2_kitware_data_dir}/kitware_m2_video_{x}_extracted"
        for x in [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ]
    ],
}


def grab_data(skill, machine="gyges"):
    if machine == "gyges":
        skill_data_root = f"{bbn_data_root}/{TASK_TO_NAME[skill]}/Data"
        videos = os.listdir(skill_data_root)
        
        videos_paths = [f"{skill_data_root}/{video}" for video in videos]
        
        # print(f"videos_paths: {videos_paths}")
        # print(f"m2_training_split: {m2_training_split}")
        # if skill == "m2":
        return (
                ptg_root,
                data_dir,
                m2_activity_config_fn,
                m2_activity_gt_dir,
                None,
                videos_paths,
                m2_obj_dets_dir,
                m2_obj_config,
            )
        
        
        # elif skill == "m2_lab":
        #     return (
        #         ptg_root,
        #         data_dir,
        #         None,
        #         None,
        #         m2_kitware_data_dir,
        #         m2_lab_training_split,
        #         None,
        #         None,
        #     )
        # elif skill == "m2_kitware":
        #     return (
        #         ptg_root,
        #         data_dir,
        #         None,
        #         None,
        #         None,
        #         m2_kitware_training_split,
        #         None,
        #         None,
        #     )

        # else:
        #     raise NotImplementedError
    else:
        raise NotImplementedError
