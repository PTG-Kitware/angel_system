import os
import argparse

from pathlib import Path

from angel_system.data.medical.load_bbn_data import (
    bbn_yolomodel_dataloader,
    save_as_kwcoco,
)


def main():
    all_skills = ["M2"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/data/PTG/medical/bbn_data/Release_v0.5/",
        help=f"Path to the dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.56",
        help=f"BBN dataset version",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/data/PTG/medical/object_anns",
        help=f"BBN dataset version",
    )
    args = parser.parse_args()

    root_dir_version = f"{args.root_dir}/{args.version}"
    # Should be M1 folder, M2 folder, etc
    subfolders = os.listdir(root_dir_version)
    for task_name in subfolders:
        print(task_name)

        task_id = task_name.split("_")[0].lower()
        output_dir = f"{args.output_root}/{task_id}/{args.version}"
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        for split in ["train", "test"]:
            classes, gt_bboxes = bbn_yolomodel_dataloader(
                root_dir=root_dir_version, skill=task_name, split=split
            )

            out = f"{output_dir}/{task_name}_YoloModel_LO_{split}.mscoco.json"
            save_as_kwcoco(classes, gt_bboxes, save_fn=out)


if __name__ == "__main__":
    main()
