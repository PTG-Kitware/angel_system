"""
This script will take a kwcoco file that was output from the TCN node (for example)
and output the belief file that is used by the BBN eval_kit. The belief file is a CSV.
"""
from pathlib import Path

import click
import kwcoco
import numpy as np
import yaml

from angel_system.global_step_prediction.belief_file import BeliefFile
from angel_system.global_step_prediction.global_step_predictor import (
    GlobalStepPredictor,
)

# TODO: make these options in the future?
threshold_multiplier_weak = 1.0
threshold_frame_count = 3
threshold_frame_count_weak = 8
deactivate_thresh_frame_count = 8

def get_belief_file(
    coco_ds: kwcoco.CocoDataset,
    medical_task="r18",
    code_dir=Path("."),
    out_file=Path("./belief_file.csv"),
    model_file=Path("./model_files/task_monitor/global_step_predictor_act_avgs_R18.npy"),
) -> None:
    """
    Run the inference and create the belief file.
    """

    # path to the medical activity labels
    act_path = code_dir / "config/activity_labels/medical" / f"{medical_task}.yaml"

    # load the steps from the activity config file
    with open(act_path, "r") as stream:
        config = yaml.safe_load(stream)
        labels = []
        for lbl in config["labels"]:
            id = float(lbl["id"])  # using float based on the belief file format
            if id > 0:  # skip the background label - not used in belief format
                labels.append(id)
        print(f"Labels: {labels}")

    start_time = 0  # start of the video

    # setup the belief file
    print(f"setting up output: {out_file}")
    belief = BeliefFile(out_file, medical_task.upper(), labels, start_time)

    # setup the global step predictor
    gsp = GlobalStepPredictor(
        threshold_multiplier_weak=threshold_multiplier_weak,
        threshold_frame_count=threshold_frame_count,
        threshold_frame_count_weak=threshold_frame_count_weak,
        deactivate_thresh_frame_count=deactivate_thresh_frame_count,
        recipe_types=[f"{medical_task}"],
        activity_config_fpath=act_path.as_posix(),
        recipe_config_dict={
            f"{medical_task}": code_dir
            / "config/tasks/medical"
            / f"{medical_task}.yaml"
        },
    )
    # load the model
    gsp.get_average_TP_activations_from_file(model_file)

    all_vid_ids = np.unique(np.asarray(coco_ds.images().lookup("video_id")))
    for vid_id in all_vid_ids:
        print(f"vid_id {vid_id}===========================")

        image_ids = coco_ds.index.vidid_to_gids[vid_id]
        annots_images = coco_ds.subset(gids=image_ids, copy=True)

        # All N activity confs x each video frame
        activity_confs = annots_images.annots().get("prob")

        # get the frame_index from the images
        ftimes = annots_images.images().lookup("frame_index")
        #print(ftimes)

        step_mode = "granular"
        for i, conf_array in enumerate(activity_confs):
            current_time = ftimes[i]  # get the time from the image's frame_index

            if current_time > 0:  # skip any 0 index frames
                tracker_dict_list = gsp.process_new_confidences(np.array([conf_array]))
                for task in tracker_dict_list:
                    current_step_id = task[f"current_{step_mode}_step"]

                    # If we are on the last step and it is not active, mark it as done
                    if (
                        current_step_id == task[f"total_num_{step_mode}_steps"] - 1
                        and not task["active"]
                    ):
                        belief.final_step_done()

                    print(f"Updating based on: {current_time}")
                    belief.update_values(current_step_id, conf_array, current_time)

    print(f"finished writing belief file: {out_file}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "medical_task",
    type=str,
)
@click.argument(
    "coco_file",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
    default="./stuff/r18_bench1_activity_predictions.kwcoco",
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
    default="./belief_file.csv",
    help="The path to where to save the output file",
)
def run_expirement(
    medical_task: str,
    coco_file: Path,
    code_dir: Path,
    out_file: Path,
) -> None:
    """
    Creates the belief file.
    """

    print(f"Running medical task: {medical_task}")
    print(f"coco_file = {coco_file}")

    get_belief_file(
        kwcoco.CocoDataset(coco_file),
        medical_task=medical_task,
        code_dir=code_dir,
        out_file=out_file,
    )


if __name__ == "__main__":
    run_expirement()
