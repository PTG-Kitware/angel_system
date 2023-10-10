import argparse

from pathlib import Path

from angel_system.data.common.kwcoco_utils import dive_csv_to_kwcoco
from angel_system.data.data_paths import grab_data


def main():
    all_recipes = ["coffee", "tea", "oatmeal", "pinwheel", "dessertquesadilla"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recipe",
        type=str,
        default="tea",
        help=f"Title of recipe to run. Options are {all_recipes}",
    )

    args = parser.parse_args()
    if args.recipe not in all_recipes:
        print(f"Must select one of: {all_recipes}")
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

    dive_csv_to_kwcoco(dive_f, obj_config, ros_bags_dir, dst_dir, output_dir)


if __name__ == "__main__":
    main()
