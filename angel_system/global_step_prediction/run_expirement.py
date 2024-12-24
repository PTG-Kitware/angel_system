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
    avg_smd, avg_smd_normd = np.array([]), np.array([])
    mean_F1s = np.array([])
    for vid_id in all_vid_ids:
        print(f"vid_id {vid_id}===========================")

        act_path = code_dir / "config/activity_labels/medical" / f"{medical_task}.yaml"

        # Get framerate
        framerate = round(coco_test.videos(video_ids=[vid_id]).peek()['framerate'])
        # Assuming FR = 15 or 30
        assert framerate in [15, 30], f"framerate rounded to {framerate}"

        step_predictor = GlobalStepPredictor(
            recipe_types=[f"{medical_task}"],
            activity_config_fpath=act_path.as_posix(),
            # activity_config_fpath="/data/PTG/medical/training/activity_classifier/TCN_HPL/logs/r18_pro_data_top_1_objs_feat_v6_NEW_ORDER_win_25/runs/2024-05-08_12-05-20/test_activity_preds.mscoco.json",
            recipe_config_dict={
                f"{medical_task}": code_dir
                / "config/tasks/medical"
                / f"{medical_task}.yaml"
            },
            threshold_multiplier=0.6,
            threshold_frame_count=5
        )

        if avg_probs is not None:
            step_predictor.get_average_TP_activations_from_array(avg_probs)
        else:
            avg_probs = step_predictor.compute_average_TP_activations(
                coco_train, coco_test
            )
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
        # If framerate is 30, take every other frame.
        if framerate == 30:
            # This must be a 30Hz video, not 15Hz. Take every other GT frame.
            print(f"halve gt for {vid_id}. Len = {len(activity_gts)}")
            activity_gts = [a for ind, a in enumerate(activity_gts) if ind%2==0]
            print(f"new len = {len(activity_gts)}")
        # ...and cut out the first 25 frames.
        activity_gts = activity_gts[25:]


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

        # activity_gt_maxes[i] set to max step so far at index i.
        activity_gt_maxes = activity_gts.copy()
        for ind in range(1, len(activity_gts)):
            activity_gt_maxes[ind] = max(activity_gts[:ind])

        _TP, _FP, _FN = step_predictor.save_TP_FP_FN_per_class(activity_gt_maxes)
        if 'TP' in locals():
            TP += _TP
            FP += _FP
            FN += _FN
        else:
            TP = _TP
            FP = _FP
            FN = _FN
        class_F1s, mean_F1 = compute_class_f1s_and_mean_f1(TP,FP,FN)
        print(f"class-wise F1s: {class_F1s}\nmean F1: {mean_F1}")

        mean_F1s = np.append(mean_F1s, mean_F1)

        pred_history = step_predictor.get_single_tracker_pred_history()
        smds, smds_normd = get_start_moment_distances(pred_history, activity_gt_maxes)

        avg_smd = np.append(avg_smd, np.mean(smds))
        avg_smd_normd = np.append(avg_smd_normd, np.mean(smds_normd))

        print(f"smds (# frames): {smds}, normalized:{smds_normd}")
        try:
            print(f"avg frame-wise smd:{avg_smd[-1]}, normalized: {avg_smd_normd[-1]}")
        except:
            import ipdb; ipdb.set_trace()

        _ = step_predictor.plot_gt_vs_predicted_one_recipe(
            activity_gts,
            recipe_type,
            fname_suffix=f"{str(vid_id)}_granular_{extra_output_suffix}",
            granular_or_broad="granular",
        )
    print("########## OVERALL")
    print(f"Overall average smd: {np.mean(avg_smd)}. Normalized: {np.mean(avg_smd_normd)}")
    print(f"overall mean F1: {np.mean(mean_F1s)}")

def get_start_moment_distances(pred_history, activity_gt_maxes):
    """
    Get the distance between ground truth & predictions of the starting frame of
    each step.

    Outputs:
    - smds: list of length equal to number 
    """
    num_classes = max(activity_gt_maxes)
    vid_length = len(activity_gt_maxes)
    smds = np.zeros(num_classes)
    smds_normd = np.zeros(num_classes)
    for i in range(num_classes):
        if i+1 in pred_history:
            smds[i] = abs(np.where(pred_history == i+1)[0][0] - activity_gt_maxes.index(i+1))
            smds_normd[i] = smds[i] / vid_length
        else:
            smds[i] = vid_length
            smds_normd[i] = 1
    return smds, smds_normd


def compute_class_f1s_and_mean_f1(TP,FP,FN):
    F1s = np.zeros(len(TP))
    # class-wise F1s:
    for i in range(len(TP)):
        F1s[i] = 2*TP[i] / (2*TP[i] + FP[i] + FN[i])
    # mean F1
    mean_F1 = 2*np.sum(TP[:-1]) / (2*np.sum(TP[:-1]) + np.sum(FP[:-1]) + np.sum(FN[:-1]))
    return F1s, mean_F1

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
