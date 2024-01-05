import argparse

from pathlib import Path

from angel_system.data.load_bbn_medical_data import data_loader, save_as_kwcoco
from angel_system.data.data_paths import grab_data


def main():
    all_skills = ["M2"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skill",
        type=str,
        default="M2",
        help=f"Title of skill to run. Options are {all_skills}",
    )

    args = parser.parse_args()
    if args.skill not in all_skills:
        print(f"Must select one of: {all_skills}")
        return

    (
        ptg_root,
        data_dir,
        activity_config_fn,
        activity_gt_dir,
        ros_bags_dir,
        training_split,
        obj_dets_dir,
        obj_config,
    ) = grab_data(args.recipe, "gyges")

    dive_f = f"{obj_dets_dir}/berkeley/dive"
    dst_dir = f"{data_dir}/images/{args.recipe}/berkeley/"
    Path(dst_dir).mkdir(parents=True, exist_ok=True)

    output_dir = f"{obj_dets_dir}/berkeley/"

    for split in ["train", "test"]:
        classes, gt_bboxes = data_loader(split)

        out = f"{root_dir}/M2_Tourniquet/YoloModel/M2_YoloModel_LO_{split}.mscoco.json"
        save_as_kwcoco(classes, gt_bboxes, save_fn=out)


if __name__ == "__main__":
    main()