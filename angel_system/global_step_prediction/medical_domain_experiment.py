import yaml
import os
import seaborn as sn
import numpy as np
import kwcoco
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndi

from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
)


def run_inference_all_vids(coco, config, extra_output_suffix=""):
    all_vid_ids = np.unique(np.asarray(coco.images().lookup("video_id")))
    avg_probs = None
    for vid_id in all_vid_ids:
        # if vid_id == 18:
        #     continue
        print(f"vid_id {vid_id}===========================")

        step_predictor = GlobalStepPredictor(
            recipe_types=['r18'],
            activity_config_fpath=config['r18'],
            recipe_config_dict=config,
        )
        # Add a second coffee predictor
        # step_predictor.initialize_new_recipe_tracker("coffee")

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(coco_test)
            np.save(
                "model_files/global_step_predictor_act_avgs_all_classes.npy", avg_probs
            )
            print(f"average_probs = {avg_probs}")

        image_ids = coco.index.vidid_to_gids[vid_id]
        video_dset = coco.subset(gids=image_ids, copy=True)

        # All N activity confs x each video frame
        activity_confs = video_dset.images().lookup("activity_conf")
        activity_gts = video_dset.images().lookup("activity_gt")

        def get_unique(activity_ids):
            """
            Get unique list indexes without sorting the list.
            """
            indexes = np.unique(activity_ids, return_index=True)[1]
            return [activity_ids[index] for index in sorted(indexes)]

        print(f"unique activities: {get_unique(activity_gts)}")

        step_predictor.process_new_confidences(activity_confs)

        recipe_type = step_predictor.determine_recipe_from_gt_first_activity(
            activity_gts
        )
        print(f"recipe_type = {recipe_type}")
        if recipe_type == "unknown_recipe_type":
            print("skipping plotting.")
            continue
        config_fn = config[recipe_type]
        (
            granular_step_gts,
            granular_step_gts_no_background,
            broad_step_gts,
            broad_step_gts_no_background,
        ) = step_predictor.get_gt_steps_from_gt_activities(video_dset, config_fn)

        print(f"unique broad steps: {get_unique(broad_step_gts)}")

        step_predictor.plot_gt_vs_predicted_one_recipe(
            granular_step_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_granular_{extra_output_suffix}",
            granular_or_broad="granular",
        )
        step_predictor.plot_gt_vs_predicted_one_recipe(
            broad_step_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_broad_{extra_output_suffix}",
            granular_or_broad="broad",
        )


if __name__ == "__main__":
    """
    coco_val = kwcoco.CocoDataset(
        "/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/yolo_all_recipes_sample_rate_2/runs/2023-10-25_12-08-48/val_activity_preds_epoch69.mscoco.json"
    )
    coco_test = kwcoco.CocoDataset(
        "/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/yolo_all_recipes_sample_rate_2/runs/2023-10-25_12-08-48/test_activity_preds.mscoco.json"
    )
    """
    # coco_val = kwcoco.CocoDataset(
    #     "/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/yolo_all_recipes_additional_objs_bkgd_sample_rate_2/runs/2023-11-02_00-52-03/val_activity_preds_epoch43.mscoco.json"
    # )
    coco_test = kwcoco.CocoDataset(
        "/data/users/peri.akiva/PTG/medical/training/activity_classifier/TCN_HPL/logs/p_r18_feat_v6_with_pose_v3_aug_False_reshuffle_True/runs/2024-03-11_09-44-03/test_activity_preds.mscoco.json"
    )

    config = {
        "m2": "/home/local/KHQ/peri.akiva/projects/angel_system/config/tasks/m2.yaml",
        "m3": "/home/local/KHQ/peri.akiva/projects/angel_system/config/tasks/m3.yaml",
        "m5": "/home/local/KHQ/peri.akiva/projects/angel_system/config/tasks/m5.yaml",
        "r18": "/home/local/KHQ/peri.akiva/projects/angel_system/config/tasks/r18.yaml",
    }

    run_inference_all_vids(coco_test, config, extra_output_suffix="test_set")
    # run_inference_all_vids(coco_val, recipe_config, extra_output_suffix="val_set")

    """
    # 2 Coffee videos interleaved ===========================

    print(f"2 Coffee vids interleaved===========================")
    # Spliced videos:
    step_predictor = GlobalStepPredictor()
    # Add a second coffee predictor
    # step_predictor.initialize_new_recipe_tracker("coffee")
    # Use the avg_probs we already computed...
    if avg_probs is not None:
        step_predictor.get_average_TP_activations_from_array(avg_probs)
    else:
        avg_probs = step_predictor.compute_average_TP_activations(coco_test)
        print(f"average_probs = {avg_probs}")
    # All N activity confs x each video frame
    vid_id = 1
    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)
    activity_confs = video_dset.images().lookup("activity_conf")[0:6000]
    step_gts = get_gt_steps_from_gt_activities(video_dset)[0][0:6000]
    # part 2
    vid_id = 2
    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)
    activity_confs.extend(video_dset.images().lookup("activity_conf")[0:6000])
    step_gts.extend(get_gt_steps_from_gt_activities(video_dset)[0][0:6000])
    # part 3
    vid_id = 1
    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)
    activity_confs.extend(video_dset.images().lookup("activity_conf")[6001:])
    step_gts.extend(get_gt_steps_from_gt_activities(video_dset)[0][6001:])
    # part 4
    vid_id = 2
    image_ids = coco_test.index.vidid_to_gids[vid_id]
    video_dset = coco_test.subset(gids=image_ids, copy=True)
    activity_confs.extend(video_dset.images().lookup("activity_conf")[6001:])
    step_gts.extend(get_gt_steps_from_gt_activities(video_dset)[0][6001:])

    step_predictor.process_new_confidences(activity_confs)

    step_predictor.plot_gt_vs_predicted_one_recipe(
        step_gts, fname_suffix="2_coffee_vids_interleaved_1"
    )
    """
