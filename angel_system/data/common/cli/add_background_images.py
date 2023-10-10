import argparse

from data.common.kwcoco_utils import add_background_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset",
        type=str,
        default="test.mscoco.json",
        help="",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="",
    )

    args = parser.parse_args()

    add_background_images(args.dset, args.images_dir)


if __name__ == "__main__":
    main()
