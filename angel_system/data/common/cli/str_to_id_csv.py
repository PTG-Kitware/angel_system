import warnings
import glob
import yaml
import csv
import argparse

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str,
)


def str_to_id(activity_config_fn, activity_gt_dir):
    """Replace the full string activity label with the 
    activity id specified by ``activity_config_fn`` in the 
    csv file(s) inside ``activity_gt_dir``
    """
    with open(activity_config_fn, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]
    activity_version = activity_config["version"]

    print(activity_labels)
    for csv_file in glob.glob(f"{activity_gt_dir}/*.csv"):
        drop_rows = []
        with open(csv_file, "r", newline='') as csvfile_in, open(f"{csv_file[:-4]}_activity_labels_v_{activity_version}.csv",  "w", newline='') as csvfile_out:
            spamreader = csv.reader(csvfile_in, delimiter=',')
            spamwriter = csv.writer(csvfile_out, delimiter=',')

            for row in spamreader:
                if "#" in row[0]:
                    spamwriter.writerow(row)
                    continue

                label = row[-2]

                activity = [
                    x
                    for x in activity_labels
                    if sanitize_str(x["full_str"]) == sanitize_str(label)
                ]
                if not activity:
                    warnings.warn(
                        f"Label: {label} is not in the activity labels config, ignoring"
                    )
                    continue
                else:
                    activity = activity[0]
            
                    row[-2] = str(activity["id"])
                    spamwriter.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activity_gt_dir",
        type=str,
        default="coffee_labels/",
        help="Path to a folder containing actiivty annotations in csv format",
    )
    parser.add_argument(
        "--activity_config_fn",
        type=str,
        default="config/activity_configs/recipe_coffee.yaml",
        help="Path to the activity config file",
    )

    args = parser.parse_args()

    str_to_id(args.activity_config_fn, args.activity_gt_dir)

if __name__ == "__main__":
    main()
