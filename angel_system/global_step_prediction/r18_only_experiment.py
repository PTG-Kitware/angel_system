import yaml
import os
import seaborn as sn
import numpy as np
import kwcoco
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import scipy.ndimage as ndi

#from angel_system.global_step_prediction.global_step_predictor import (
from global_step_predictor import (
    GlobalStepPredictor,
)


def run_inference_all_vids(
    coco_train, coco_test, recipe_config, extra_output_suffix=""
):
    all_vid_ids = np.unique(np.asarray(coco_test.images().lookup("video_id")))
    avg_probs = None
    preds, gt = [], []
    for vid_id in all_vid_ids:
        print(f"vid_id {vid_id}===========================")

        step_predictor = GlobalStepPredictor(
            recipe_types=["r18"],
            activity_config_fpath="/home/local/KHQ/cameron.johnson/code/tmp_hannah_code/angel_system/config/activity_labels/medical/r18.yaml",
            #activity_config_fpath="/data/PTG/medical/training/activity_classifier/TCN_HPL/logs/r18_pro_data_top_1_objs_feat_v6_NEW_ORDER_win_25/runs/2024-05-08_12-05-20/test_activity_preds.mscoco.json",
            recipe_config_dict={
                "r18": "/home/local/KHQ/cameron.johnson/code/tmp_hannah_code/angel_system/config/tasks/medical/r18.yaml"
            },
            # threshold_multiplier=0.3,
            # threshold_frame_count=2
        )

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(coco_train)
            np.save(
                "/home/local/KHQ/cameron.johnson/code/tmp_hannah_code/angel_system/model_files/global_step_predictor_act_avgs_all_classes.npy",
                avg_probs,
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

        _, granular_preds, granular_gt = step_predictor.plot_gt_vs_predicted_one_recipe(
            granular_step_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_granular_{extra_output_suffix}",
            granular_or_broad="granular",
        )
        # _, broad_preds, broad_gt = step_predictor.plot_gt_vs_predicted_one_recipe(
        #     broad_step_gts,
        #     recipe_type,
        #     fname_suffix=f"{str(vid_id)}_broad_{extra_output_suffix}",
        #     granular_or_broad="broad",
        # )

        # print(f"broad_gt len: {len(broad_gt)}")
        # print(f"broad_preds len: {len(broad_preds)}")
        # print(f"granular_gt len: {len(granular_gt)}")
        # print(f"granular_preds len: {len(granular_preds)}")

        min_length = min(len(granular_preds), len(granular_gt))

        preds.extend(granular_preds[:min_length])
        gt.extend(granular_gt[:min_length])

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    num_act_classes = len(step_predictor.activity_config["labels"])
    fig, ax = plt.subplots(figsize=(num_act_classes, num_act_classes))

    print(f"gt len: {len(gt)}")
    print(f"preds len: {len(preds)}")
    print(f"labels: {step_predictor.activity_config['labels']}")
    label_ids = [item["id"] for item in step_predictor.activity_config["labels"]]
    labels = [item["full_str"] for item in step_predictor.activity_config["labels"]]

    broad_cm = confusion_matrix(gt, preds, labels=label_ids, normalize="true")

    # granular_cm = confusion_matrix(
    #     granular_step_gts,
    #     granular_preds,
    #     labels=step_predictor.activity_config["labels"],
    #     normalize="true"
    # )

    sns.heatmap(broad_cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    # ax.set_title(f'CM GSP Accuracy: {acc:.4f}')
    ax.xaxis.set_ticklabels(labels, rotation=25)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    # fig.savefig(f"{self.hparams.output_dir}/confusion_mat_val_acc_{acc:.4f}.png", pad_inches=5)
    fig.savefig(f"confusion_mat_gsp.png", pad_inches=5)


if __name__ == "__main__":
    coco_train = kwcoco.CocoDataset(
        "/data/PTG/medical/training/activity_classifier/TCN_HPL/logs/r18_pro_data_top_1_objs_feat_v6_NEW_ORDER_win_25/runs/2024-05-08_12-05-20/test_activity_preds.mscoco.json"
    )
    # Same file for now since I don't have another.
    coco_test = kwcoco.CocoDataset(
        "/data/PTG/medical/training/activity_classifier/TCN_HPL/logs/r18_pro_data_top_1_objs_feat_v6_NEW_ORDER_win_25/runs/2024-05-08_12-05-20/test_activity_preds.mscoco.json"
    )

    recipe_config = {"r18": "config/tasks/medical/r18.yaml"}

    run_inference_all_vids(
        coco_train, coco_test, recipe_config, extra_output_suffix="test_set"
    )
