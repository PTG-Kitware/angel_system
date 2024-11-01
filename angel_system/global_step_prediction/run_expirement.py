"""
Runs the global step predictor on the test set and saves the confusion matrix
as well as the average probablitites file (npy) for the given medical task.
"""
import numpy as np
import kwcoco
from pathlib import Path
import click
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
)


def run_inference_all_vids(
    coco_truth: kwcoco.CocoDataset,
    coco_train: kwcoco.CocoDataset,
    coco_test: kwcoco.CocoDataset,
    recipe_config: dict,
    extra_output_suffix="",
    medical_task="r18",
    code_dir=Path("."),
    out_file=Path("./confusion_mat_gsp.png"),
) -> None:
    """
    Run inference on all data in the train set
    """

    all_vid_ids = np.unique(np.asarray(coco_test.images().lookup("video_id")))
    avg_probs = None
    preds, gt = [], []
    for vid_id in all_vid_ids:
        print(f"vid_id {vid_id}===========================")

        act_path = code_dir / "config/activity_labels/medical" / f"{medical_task}.yaml"

        step_predictor = GlobalStepPredictor(
            recipe_types=[f"{medical_task}"],
            activity_config_fpath=act_path.as_posix(),
            # activity_config_fpath="/data/PTG/medical/training/activity_classifier/TCN_HPL/logs/r18_pro_data_top_1_objs_feat_v6_NEW_ORDER_win_25/runs/2024-05-08_12-05-20/test_activity_preds.mscoco.json",
            recipe_config_dict={
                f"{medical_task}": code_dir
                / "config/tasks/medical"
                / f"{medical_task}.yaml"
            },
            # threshold_multiplier=0.3,
            # threshold_frame_count=2
        )

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(coco_train)
            save_file = (
                code_dir
                / "model_files"
                / "task_monitor"
                / f"global_step_predictor_act_avgs_{medical_task}.npy"
            )
            print(f"Saving average probs to {save_file}")
            np.save(
                save_file,
                avg_probs,
            )
            print(f"average_probs = {avg_probs}")

        image_ids = coco_test.index.vidid_to_gids[vid_id]
        test_video_dset = coco_test.subset(gids=image_ids, copy=True)
        truth_video_dset = coco_truth.subset(gids=image_ids, copy=True)

        # All N activity confs x each video frame
        activity_confs = test_video_dset.annots().get("prob")
        activity_gts = truth_video_dset.annots().get("category_id")

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
        ) = step_predictor.get_gt_steps_from_gt_activities(truth_video_dset, config_fn)

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
    print(f"Saving confusion matrix to {out_file}")
    fig.savefig(out_file.as_posix(), pad_inches=5)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "medical_task",
    type=str,
)
@click.argument(
    "coco_truth",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    default="./stuff/bbn_working/m2_activity_truth_coco_3_tourns_1.json",
)
@click.argument(
    "coco_train",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    default="./stuff/m2_activity_preds_3_tourns_1.mscoco.json",
)
@click.argument(
    "coco_test",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    default="./stuff/m2_activity_preds_3_tourns_1.mscoco.json",
)
@click.option(
    "--code_dir",
    type=click.Path(
        exists=True, file_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    default=".",
    help="The path to the code directory",
)
@click.option(
    "--out_file",
    type=click.Path(readable=True, resolve_path=True, path_type=Path),
    default="./confusion_mat_gsp.png",
    help="The path to where to save the output file",
)
def run_expirement(
    medical_task: str,
    coco_truth: Path,
    coco_train: Path,
    coco_test: Path,
    code_dir: Path,
    out_file: Path,
) -> None:
    """
    Runs the experiment for the given medical task

    \b
    Positional Arguments:
        medical_task: str: The medical task to run the experiment on (e.g. r18, m2, m3)
        coco_truth: Path: The path to the coco file containing the truth data
        coco_train: Path: The path to the coco file containing the training data
        coco_test: Path: The path to the coco file containing the test data

    \b
    Optional Arguments:
        code_dir: Path: The path to the code directory
        out_file: Path: The path to where to save the output file
    """
    recipe_config = {f"{medical_task}": f"config/tasks/medical/{medical_task}.yaml"}

    print(f"Running medical task: {medical_task}")
    print(f"coco_truth = {coco_truth}")
    print(f"coco_train = {coco_train}")
    print(f"coco_test = {coco_test}")

    run_inference_all_vids(
        kwcoco.CocoDataset(coco_truth),
        kwcoco.CocoDataset(coco_train),
        kwcoco.CocoDataset(coco_test),
        recipe_config,
        extra_output_suffix="test_set",
        medical_task=medical_task,
        code_dir=code_dir,
        out_file=out_file,
    )


if __name__ == "__main__":
    run_expirement()
