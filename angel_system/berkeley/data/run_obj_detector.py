import os
import pickle
import glob
import ubelt as ub

from detectron2.data.detection_utils import read_image
from demo import predictor, model

from angel_system.data.common.kwcoco_utils import preds_to_kwcoco
from angel_system.data.common.load_data import Re_order, time_from_name


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_model(config, conf_thr=0.01):
    # Load model
    root_dir = "/home/local/KHQ/hannah.defazio/angel_system"
    #root_dir = "/home/local/KHQ/hannah.defazio/projects/PTG/angel_system"
    berkeley_configs_dir = f"{root_dir}/angel_system/berkeley/configs"
    model_config = f"{berkeley_configs_dir}/{config}"

    parser = model.get_parser()
    args = parser.parse_args(
        f"--config-file {model_config} --confidence-threshold {conf_thr}".split()
    )
    print("Arguments: " + str(args))

    cfg = model.setup_cfg(args)
    print(f"Model: {cfg.MODEL.WEIGHTS}")

    demo = predictor.VisualizationDemo_add_smoothing(
        cfg, last_time=2, draw_output=False, tracking=False
    )
    print(f"Loaded {model_config}")
    return demo

def run_obj_detector(
    demo, stage, split, no_contact=False, add_hl_hands=True
):
    """
    Run object detector trained without contact information
    on all the videos associated with the task and clear any
    contact predictions
    """
    preds = {}

    for (
        video_folder
    ) in split:
        idx = 0
        video_name = video_folder.split("/")[-1]

        preds[video_name] = {}

        video_images = glob.glob(f"{video_folder}_extracted/images/*.png")
        input_list = Re_order(video_images, len(video_images))
        for image_fn in ub.ProgIter(input_list, desc=f"images in {video_name}"):
            frame, time_stamp = time_from_name(image_fn)

            image = read_image(image_fn, format="RGB")
            # Only for medical
            #if stage == "results":
            #    image = Image.fromarray(image)
            #    image = image.resize(size=(760, 428), resample=Image.BILINEAR)
            #    image = np.array(image)

            h, w, c = image.shape

            predictions, step_infos, visualized_output = demo.run_on_image_smoothing_v2(
                image, current_idx=idx
            )
            decoded_preds = model.decode_prediction(predictions)
            using_contact = True if predictions[2] is not None else False

            if decoded_preds is not None:
                preds[video_name][frame] = decoded_preds

                if no_contact and using_contact:
                    print("Clearing contact states from detections")
                    for class_, dets in preds[video_name][frame].items():
                        for i in range(len(dets)):
                            # Clear contact states
                            preds[video_name][frame][class_][i][
                                "obj_obj_contact_state"
                            ] = False
                            preds[video_name][frame][class_][i][
                                "obj_hand_contact_state"
                            ] = False

                # Image metadata needed later
                preds[video_name][frame]["meta"] = {
                    "file_name": image_fn,
                    "im_size": {"height": h, "width": w},
                    "frame_idx": frame,
                    "time_stamp": time_stamp
                }

            idx += 1

    return preds, using_contact

def tea_main(stage):
    model_dir = "MC50-InstanceSegmentation/cooking/tea"
    model_dir = "MC50-InstanceSegmentation/cooking/coffee+tea"
    # Model
    demo = load_model(
        #config=f"{model_dir}/stage2/mask_rcnn_R_50_FPN_1x_demo.yaml",
        config=f"{model_dir}/stage1/mask_rcnn_R_50_FPN_1x_demo.yaml",
        #conf_thr=0.4
        conf_thr=0.01,
    )

    # Data
    tea_root = "/data/PTG/cooking/ros_bags/tea/"
    #coffee_root = "/media/hannah.defazio/Padlock_DT6/Data/notpublic/PTG/Coffee"
    ros_bags_dir = "tea_extracted/" # Tea specific

    training_split = {
        "train_activity": [f"kitware_tea_video_{x}" for x in [2, 4, 7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]],
        "val": [f"kitware_tea_video_{x}" for x in [25, 26, 31]],
        "test": [f"kitware_tea_video_{x}" for x in [27, 28, 29, 30, 32, 33, 34]],
    }  # Tea specific

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print(videos)
    print("\n")

    # Add data root to splits
    for split, videos in training_split.items():
        new_videos = []
        for x in videos:
            new_videos.append(f"{tea_root}/{ros_bags_dir}/{x}")
        training_split[split] = new_videos

    # Activity data loader
    from angel_system.data.load_kitware_cooking_data import tea_activity_data_loader

    activity_data_loader = tea_activity_data_loader

    # Step map
    metadata = demo.metadata.as_dict()

    return (
        demo,
        training_split,
        activity_data_loader,
        metadata,
    )

def coffee_main(stage):
    model_dir = "MC50-InstanceSegmentation/cooking/coffee"
    model_dir = "MC50-InstanceSegmentation/cooking/coffee+tea"
    # Model
    demo = load_model(
        #config=f"{model_dir}/stage2/mask_rcnn_R_50_FPN_1x_demo.yaml",
        config=f"{model_dir}/stage1/mask_rcnn_R_50_FPN_1x_demo.yaml",
        #conf_thr=0.4
        conf_thr=0.01,
    )

    # Data
    coffee_root = "/data/users/hannah.defazio/ptg_nas/data_copy/"
    #coffee_root = "/media/hannah.defazio/Padlock_DT6/Data/notpublic/PTG/Coffee"
    ros_bags_dir = "coffee_extracted/" # Coffee specific

    training_split = {
        "train_activity": [f"all_activities_{x}" for x in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 47, 48, 49]],
        "val": [f"all_activities_{x}" for x in [23, 24, 42, 46]],
        "test": [f"all_activities_{x}" for x in [20, 33, 39, 50, 51, 52, 53, 54]],
    }  # Coffee specific

    print("\nTraining split:")
    for split_name, videos in training_split.items():
        print(f"{split_name}: {len(videos)} videos")
        print(videos)
    print("\n")

    # Add data root to splits
    for split, videos in training_split.items():
        new_videos = []
        for x in videos:
            x_split = x.split("_")
            if x_split[0] + '_' + x_split[1]  == "all_activities":
                # Kitware Eval1 engineering dataset
                new_videos.append(f"{coffee_root}/{ros_bags_dir}/{x}")
            else:
                print(f"Unknown video source for: {x}")
                return
        training_split[split] = new_videos

    # Activity data loader
    from angel_system.data.load_kitware_cooking_data import coffee_activity_data_loader

    activity_data_loader = coffee_activity_data_loader

    # Step map
    metadata = demo.metadata.as_dict()

    return (
        demo,
        training_split,
        activity_data_loader,
        metadata,
    )

def tourniquet_main(stage, using_inter_steps, using_before_finished_task):
    # demo = load_model(config='MC50-InstanceSegmentation/medical/M2/stage1/mask_rcnn_R_101_FPN_1x_BBN_M2_demo.yaml', conf_thr=0.4)
    demo = load_model(
        config="MC50-InstanceSegmentation/medical/M2/stage2/mask_rcnn_R_50_FPN_1x_BBN_M2_labels_with_steps_demo.yaml",
        conf_thr=0.01,
    )

    bbn_root = "/data/ptg/medical/bbn/data"
    data_root = "Release_v0.5/v0.52"
    skill = "M2_Tourniquet"
    m2_data_dir = f"{data_root}/{skill}/Data"  # M2 specific
    lab_data_dir = "M2_Lab_data/skills_by_frame/"
    kitware_dir = f"kitware_m2"

    m2_videos = [f"M2-{x}" for x in range(1, 139 + 1)]
    lab_videos = [f"tq_{x}" for x in range(1, 32 + 1)]
    kitware_videos = [f"kitware_m2_video_{x}" for x in range(1, 32 + 1)]

    ignore_videos = [
        f"M2-{x}"
        for x in [
            15,  # bad video
            # 46, 47, 48, 49, 50, 51, 53, 54, 56, 58, 62, 65, 66, 67, 68, 69, 102, 105, 135, 136, 138, # objs only in corners, might be able to crop?
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            103,
            104,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            122,
            125,
            129  # no hope for these
            # currently ignoring these videos because of repeated objs in frame when an activity is happening
        ]
    ]

    all_videos = m2_videos + lab_videos + kitware_videos
    data_dirs = (m2_data_dir, lab_data_dir, kitware_dir)

    good_videos = [x for x in all_videos if x not in ignore_videos]
    random.shuffle(good_videos)

    num_videos = len(good_videos)
    print(f"Using {num_videos} videos")

    """
    i1 = int(0.5 * num_videos)
    i2 = int(0.3 * num_videos)
    i3 = i1 + i2
    i4 = int(0.1 * num_videos)

    training_split = {
        'train_contact': good_videos[0:i1], # 50%
        'train_activity': good_videos[i1:i3], # 30%
        'val': good_videos[i3:(i3+i4)], # 10%
        'test': good_videos[(i3+i4):] # 10%
    } # M2 specific
    """
    training_split = {
        "train_contact": [
            "M2-131",
            "M2-63",
            "M2-71",
            "tq_2",
            "M2-11",
            "tq_4",
            "M2-54",
            "tq_12",
            "M2-73",
            "M2-62",
            "tq_8",
            "M2-46",
            "M2-60",
            "M2-24",
            "M2-19",
            "M2-2",
            "M2-74",
            "M2-28",
            "M2-135",
            "M2-130",
            "M2-10",
            "M2-61",
            "tq_21",
            "M2-137",
            "M2-53",
            "M2-40",
            "M2-66",
            "M2-138",
            "M2-124",
            "M2-102",
            "M2-32",
            "M2-29",
            "tq_23",
            "tq_3",
            "tq_20",
            "M2-41",
            "tq_19",
            "M2-1",
            "tq_14",
            "M2-127",
            "tq_13",
            "M2-23",
            "M2-17",
            "tq_10",
            "M2-139",
            "M2-55",
            "M2-69",
            "tq_11",
            "M2-68",
            "M2-13",
            "M2-133",
            "M2-121",
            "M2-14",
            "M2-50",
            "tq_5",
            "M2-44",
            "M2-43",
            "M2-35",
            "M2-75",
            "M2-37",
        ],
        "train_activity": [
            "M2-38",
            "M2-48",
            "M2-123",
            "M2-65",
            "M2-59",
            "M2-49",
            "M2-45",
            "M2-5",
            "M2-7",
            "tq_16",
            "M2-56",
            "M2-12",
            "M2-67",
            "M2-20",
            "M2-64",
            "M2-6",
            "M2-57",
            "M2-126",
            "M2-77",
            "M2-106",
            "M2-120",
            "M2-132",
            "M2-76",
            "M2-27",
            "M2-136",
            "M2-47",
            "M2-3",
            "M2-18",
            "M2-26",
            "tq_6",
            "M2-25",
            "M2-8",
            "M2-34",
            "tq_9",
            "M2-21",
            "tq_1",
        ],
        "val": [
            "M2-72",
            "M2-58",
            "M2-134",
            "M2-105",
            "tq_18",
            "M2-33",
            "tq_22",
            "M2-9",
            "M2-42",
            "M2-30",
            "M2-16",
            "M2-128",
        ],
        "test": [
            "tq_17",
            "M2-119",
            "M2-51",
            "M2-31",
            "M2-22",
            "M2-36",
            "M2-39",
            "M2-70",
            "tq_15",
            "M2-4",
            "M2-52",
            "tq_7",
        ],
    }
    training_split["train_contact"] = (
        kitware_videos[:20]
        + [f"tq_{x}" for x in range(24, 28 + 1)]
        + training_split["train_contact"]
    )
    training_split["train_activity"] = (
        kitware_videos[20:]
        + [f"tq_{x}" for x in range(29, 32 + 1)]
        + training_split["train_activity"]
    )

    print(training_split)

    # Add data root to splits
    for split in training_split:
        new_split = []
        for x in split:
            x_split = x.split("_")
            if x_split[0] == "tq":
                # BBN lab videos
                new_split.append(f"{bbn_root}/{data_dirs[1]}/{x}")
            elif x_split[0] == "kitware":
                # Kitware lab videos
                new_split.append(f"{bbn_root}/{data_dirs[2]}/{x}")
            else:
                new_split.append(f"{bbn_root}/{data_dirs[0]}/{x}")
        training_split[split] = new_split

    from angel_system.data.load_bbn_medical_data import bbn_activity_data_loader

    activity_data_loader = bbn_activity_data_loader

    # Update step map
    metadata = demo.metadata.as_dict()

    step_map = update_step_map(metadata["step_map"],
                               ["tourniquet_tourniquet", "hand"],
                               using_inter_steps, using_before_finished_task)
    print(f"step map: {step_map}")

    # data_dirs = (kitware_test_dir, lab_data_dir)
    # training_split['test'] = kitware_videos
    return (
        demo,
        training_split,
        bbn_root,
        data_dirs,
        activity_data_loader,
        metadata,
        step_map,
    )

def main():
    experiment_name = "coffee_and_tea"
    stage = "results"

    print("Experiment: ", experiment_name)
    print("Stage: ", stage)

    (
        demo,
        training_split,
        activity_data_loader,
        metadata,
    ) = tea_main(
        stage,
    )

    splits = ["train_activity", "test"]

    if not os.path.exists("temp"):
        os.mkdir("temp")

    for split in splits:
        print(f"{split}: {len(training_split[split])} videos")

        # Raw detector output
        preds, using_contact = run_obj_detector(
            demo,
            stage,
            training_split[split],
            add_hl_hands=False,
        )
        print(f"Using contact: {using_contact}")

        fn = f"temp/{experiment_name}_{split}_preds.pickle"
        print("temp file: ", fn)
        with open(fn, "wb") as fh:
            #preds = pickle.load(fh)
            pickle.dump(preds, fh)

        dset = preds_to_kwcoco(
            metadata,
            preds,
            "",
            save_fn=f"{experiment_name}_{stage}_{split}.mscoco.json",
        )

if __name__ == "__main__":
    main()
