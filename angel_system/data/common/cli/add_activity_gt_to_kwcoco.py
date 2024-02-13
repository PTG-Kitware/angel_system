import argparse

from angel_system.data.common.kwcoco_utils import add_activity_gt_to_kwcoco


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset",
        type=str,
        default="obj_annotations.mscoco.json",
        help="kwcoco filename",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="medical",
        help="Task name",
    )

    args = parser.parse_args()

    add_activity_gt_to_kwcoco(args.task, args.dset)


if __name__ == "__main__":
    main()
