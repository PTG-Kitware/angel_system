###############
# GYGES
###############
ptg_root = "/home/local/KHQ/hannah.defazio/angel_system/"
activity_config_path = f"{ptg_root}/config/activity_labels"
object_config_path = f"{ptg_root}/config/object_labels"

data_dir = "/data/PTG/cooking/"
activity_gt_dir = f"{data_dir}/activity_anns"
objects_dir = f"{data_dir}/object_anns"
ros_bags_dir = f"{data_dir}/ros_bags/"

# Coffee
# ------
coffee_activity_gt_dir = f"{activity_gt_dir}/coffee_labels/"
coffee_activity_config_fn = f"{activity_config_path}/recipe_coffee.yaml"

coffee_ros_bags_dir = f"{ros_bags_dir}/coffee/coffee_extracted/"

coffee_training_split = {
    "train_activity": [
        f"{coffee_ros_bags_dir}/all_activities_{x}_extracted"
        for x in [
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
            21,
            22,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            47,
            48,
            49,
        ]
    ],
    "val": [
        f"{coffee_ros_bags_dir}/all_activities_{x}_extracted" for x in [23, 24, 42, 46]
    ],
    "test": [
        f"{coffee_ros_bags_dir}/all_activities_{x}_extracted"
        for x in [20, 33, 39, 50, 51, 52, 53, 54]
    ],
}

coffee_obj_dets_dir = f"{objects_dir}/coffee"
coffee_obj_config = f"{object_config_path}/recipe_coffee.yaml"

# Tea
# ---
tea_activity_gt_dir = f"{activity_gt_dir}/tea_labels/"
tea_activity_config_fn = f"{activity_config_path}/recipe_tea.yaml"

tea_ros_bags_dir = f"{ros_bags_dir}/tea/tea_extracted/"  # Tea specific

tea_training_split = {
    "train_activity": [
        f"{tea_ros_bags_dir}/kitware_tea_video_{x}_extracted"
        for x in [
            2,
            4,
            7,
            8,
            10,
            11,
            13,
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
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
        ]
    ],
    "val": [
        f"{tea_ros_bags_dir}/kitware_tea_video_{x}_extracted" for x in [25, 26, 31]
    ],
    "test": [
        f"{tea_ros_bags_dir}/kitware_tea_video_{x}_extracted"
        for x in [27, 28, 29, 30, 32, 33, 34]
    ],
}

tea_obj_dets_dir = f"{objects_dir}/tea"
tea_obj_config = f"{object_config_path}/recipe_tea.yaml"

# Dessert Quesadilla
# ------------------
dq_activity_gt_dir = f"{activity_gt_dir}/dessert_quesadilla_labels/"
dq_activity_config_fn = f"{activity_config_path}/recipe_dessert_quesadilla.yaml"

dq_ros_bags_dir = (
    f"{ros_bags_dir}/dessert_quesadilla/dessert_quesadilla_extracted/"  # DQ specific
)

dq_training_split = {
    "train_activity": [
        f"{dq_ros_bags_dir}/kitware_dessert_quesadilla_video_{x}_extracted"
        for x in [
            7,
            9,
            10,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            23,
            25,
            27,
            29,
            30,
            31,
            32,
            33,
            34,
            38,
            40,
            41,
        ]
    ],
    "val": [
        f"{dq_ros_bags_dir}/kitware_dessert_quesadilla_video_{x}_extracted"
        for x in [11, 20, 24, 44]
    ],
    "test": [
        f"{dq_ros_bags_dir}/kitware_dessert_quesadilla_video_{x}_extracted"
        for x in [8, 12, 22, 26, 28, 35, 42, 43]
    ],
}

dq_obj_dets_dir = f"{objects_dir}/dessert_quesadilla"
dq_obj_config = f"{object_config_path}/recipe_dessert_quesadilla.yaml"

# Oatmeal
# ---
oatmeal_activity_gt_dir = f"{activity_gt_dir}/oatmeal_labels/"
oatmeal_activity_config_fn = f"{activity_config_path}/recipe_oatmeal.yaml"

oatmeal_ros_bags_dir = f"{ros_bags_dir}/oatmeal/oatmeal_extracted/"  # oatmeal specific

oatmeal_training_split = {
    "train_activity": [
        f"{oatmeal_ros_bags_dir}/kitware_oatmeal_video_{x}_extracted"
        for x in [
            11,
            12,
            13,
            14,
            15,
            19,
            22,
            26,
            29,
            34,
            35,
            36,
            39,
            42,
            43,
            46,
            47,
            53,
            54,
            55,
            57,
            58,
            59,
            60,
            61,
        ]
    ],
    "val": [
        f"{oatmeal_ros_bags_dir}/kitware_oatmeal_video_{x}_extracted"
        for x in [31, 40, 49, 56]
    ],
    "test": [
        f"{oatmeal_ros_bags_dir}/kitware_oatmeal_video_{x}_extracted"
        for x in [16, 17, 18, 25, 32, 33, 38, 45]
    ],
}

oatmeal_obj_dets_dir = f"{objects_dir}/oatmeal"
oatmeal_obj_config = f"{object_config_path}/recipe_oatmeal.yaml"

# Pinwheels
# ---------
pinwheels_activity_gt_dir = f"{activity_gt_dir}/pinwheels_labels/"
pinwheels_activity_config_fn = f"{activity_config_path}/recipe_pinwheel.yaml"

pinwheels_ros_bags_dir = f"{ros_bags_dir}/pinwheels/pinwheels_extracted/"

pinwheels_training_split = {
    "train_activity": [
        f"{pinwheels_ros_bags_dir}/kitware_pinwheel_video_{x}_extracted"
        for x in [
            3,
            11,
            13,
            14,
            15,
            17,
            18,
            19,
            31,
            35,
            36,
            37,
            38,
            39,
            40,
            42,
            43,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            56,
            58,
        ]
    ],
    "val": [
        f"{pinwheels_ros_bags_dir}/kitware_pinwheel_video_{x}_extracted"
        for x in [20, 33, 54, 57]
    ],
    "test": [
        f"{pinwheels_ros_bags_dir}/kitware_pinwheel_video_{x}_extracted"
        for x in [9, 12, 16, 32, 34, 44, 45]
    ],
}

pinwheels_obj_dets_dir = f"{objects_dir}/pinwheels"
pinwheels_obj_config = f"{object_config_path}/recipe_pinwheels.yaml"

# All
# ---
all_activity_config_fn = f"{activity_config_path}/all_recipe_labels.yaml"


def grab_data(recipe, machine):
    if machine == "gyges":
        if recipe == "coffee":
            return (
                ptg_root,
                data_dir,
                coffee_activity_config_fn,
                coffee_activity_gt_dir,
                coffee_ros_bags_dir,
                coffee_training_split,
                coffee_obj_dets_dir,
                coffee_obj_config,
            )
        elif recipe == "tea":
            return (
                ptg_root,
                data_dir,
                tea_activity_config_fn,
                tea_activity_gt_dir,
                tea_ros_bags_dir,
                tea_training_split,
                tea_obj_dets_dir,
                tea_obj_config,
            )
        elif recipe == "dessertquesadilla":
            return (
                ptg_root,
                data_dir,
                dq_activity_config_fn,
                dq_activity_gt_dir,
                dq_ros_bags_dir,
                dq_training_split,
                dq_obj_dets_dir,
                dq_obj_config,
            )
        elif recipe == "oatmeal":
            return (
                ptg_root,
                data_dir,
                oatmeal_activity_config_fn,
                oatmeal_activity_gt_dir,
                oatmeal_ros_bags_dir,
                oatmeal_training_split,
                oatmeal_obj_dets_dir,
                oatmeal_obj_config,
            )
        elif recipe == "pinwheel":
            return (
                ptg_root,
                data_dir,
                pinwheels_activity_config_fn,
                pinwheels_activity_gt_dir,
                pinwheels_ros_bags_dir,
                pinwheels_training_split,
                pinwheels_obj_dets_dir,
                pinwheels_obj_config,
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
