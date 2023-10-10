import argparse

from pathlib import Path

from angel_system.data.common.kwcoco_utils import visualize_kwcoco_by_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset",
        type=str,
        default="obj_annotations.mscoco.json",
        help="kwcoco filename",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="obj_annotations.mscoco.json",
        help="Path to save output images",
    )

    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    visualize_kwcoco_by_label(args.dset, args.save_dir)

if __name__ == "__main__":
    main()