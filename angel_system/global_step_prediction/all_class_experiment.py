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


if __name__ == "__main__":
    coco_val = kwcoco.CocoDataset(
        "/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/yolo_all_recipes_sample_rate_2/runs/2023-10-25_12-08-48/val_activity_preds_epoch69.mscoco.json"
    )
    coco_test = kwcoco.CocoDataset(
        "/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/yolo_all_recipes_sample_rate_2/runs/2023-10-25_12-08-48/test_activity_preds.mscoco.json"
    )

    recipe_config = {
        "coffee": "config/tasks/recipe_coffee.yaml",
        "tea": "config/tasks/recipe_tea.yaml",
        "dessert_quesadilla": "config/tasks/recipe_dessertquesadilla.yaml",
        "oatmeal": "config/tasks/recipe_oatmeal.yaml",
        "pinwheel": "config/tasks/recipe_pinwheel.yaml",
    }

    # Train
    avg_probs = None

    all_vid_ids = np.unique(np.asarray(coco_val.images().lookup("video_id")))

    for vid_id in all_vid_ids:
        print(f"vid_id {vid_id}===========================")

        step_predictor = GlobalStepPredictor()
        # Add a second coffee predictor
        step_predictor.initialize_new_recipe_tracker("coffee")

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(coco_test)
            np.save(
                "model_files/global_step_predictor_act_avgs_all_classes.npy", avg_probs
            )
            print(f"average_probs = {avg_probs}")

        image_ids = coco_test.index.vidid_to_gids[vid_id]
        video_dset = coco_test.subset(gids=image_ids, copy=True)

        # All N activity confs x each video frame
        activity_confs = video_dset.images().lookup("activity_conf")
        activity_gts = video_dset.images().lookup("activity_gt")

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

        step_predictor.plot_gt_vs_predicted_one_recipe(
            granular_step_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_granular",
            granular_or_broad="granular",
        )
        step_predictor.plot_gt_vs_predicted_one_recipe(
            broad_step_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_broad",
            granular_or_broad="broad",
        )

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
