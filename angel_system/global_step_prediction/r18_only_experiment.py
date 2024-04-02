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


def run_inference_all_vids(
    coco_train, coco_test, recipe_config, extra_output_suffix=""
):
    all_vid_ids = np.unique(np.asarray(coco_test.images().lookup("video_id")))
    avg_probs = None
    for vid_id in all_vid_ids:
        if vid_id == 18:
            continue
        print(f"vid_id {vid_id}===========================")

        step_predictor = GlobalStepPredictor(
            recipe_types=["r18"],
            activity_config_fpath="config/activity_labels/r18.yaml",
            recipe_config_dict={"r18": "config/tasks/r18.yaml"},
            # threshold_multiplier=0.3,
            # threshold_frame_count=2
        )
        # Add a second coffee predictor
        # step_predictor.initialize_new_recipe_tracker("coffee")

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(coco_train)
            np.save(
                "model_files/global_step_predictor_act_avgs_all_classes.npy", avg_probs
            )
            print(f"average_probs = {avg_probs}")

        image_ids = coco_test.index.vidid_to_gids[vid_id]
        video_dset = coco_test.subset(gids=image_ids, copy=True)

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
        config_fn = recipe_config[recipe_type]
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
    coco_train = kwcoco.CocoDataset(
        "/data/users/cameron.johnson/datasets/R18/TCN_HPL/logs/p_r18_feat_v6_with_pose_v3_aug_False_reshuffle_True/runs/2024-03-11_09-44-03/test_activity_preds.mscoco.json"
    )
    # Same file for now since I don't have another.
    coco_test = kwcoco.CocoDataset(
        "/data/users/cameron.johnson/datasets/R18/TCN_HPL/logs/p_r18_feat_v6_with_pose_v3_aug_False_reshuffle_True/runs/2024-03-11_09-44-03/test_activity_preds.mscoco.json"
    )

    recipe_config = {"r18": "config/tasks/r18.yaml"}

    run_inference_all_vids(
        coco_train, coco_test, recipe_config, extra_output_suffix="test_set"
    )
    # run_inference_all_vids(coco_val, recipe_config, extra_output_suffix="val_set")
