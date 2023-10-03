import argparse

from data.common.kwcoco_utils import dive_csv_to_kwcoco


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

    dive_f = f"/data/PTG/cooking/annotations/{args.recipe}/berkeley/dive"
    obj_config = f"config/object_labels/recipe_{args.recipe}.yaml"
    data_dir = f"/data/PTG/cooking/ros_bags/{args.recipe}/{args.recipe}_extracted/"
    dst_dir = f"/data/PTG/cooking/images/{args.recipe}/berkeley/"
    
    dive_csv_to_kwcoco(dive_f, obj_config, data_dir, dst_dir, args.output_dir)

if __name__ == "__main__":
    main()