#!/usr/bin/env python3
"""
Convert a video (mp4) or a series of images into a ROS bag.

Example running (inside ROS environment):
ros2 run angel_utils convert_video_to_ros_bag.py \
  --video-fn video.mp4 \
  --output-bag-folder ros_bags/new_bag
"""
import argparse
from glob import glob
from pathlib import Path
import PIL.Image
import time
from typing import Generator
from typing import Iterable
from typing import Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
import rclpy.time
import rclpy.duration
from rclpy.serialization import serialize_message
import rosbag2_py
from sensor_msgs.msg import Image


def iter_video_frames(video_filepath: str):
    """
    Iterate output frames via OpenCV's VideoCapture functionality.

    Mock's starting time as "right now".

    :param video_filepath: Filepath to the video file.

    :return: Iterator returning a tuple of:
        0) BGR format image pixel matrices.
        1) Relative time in seconds of this frame since the beginning of the
           sequence.
    """
    cap = cv2.VideoCapture(video_filepath)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_secs = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        yield frame, frame_secs
    cap.release()


def iter_file_frames(filepath_glob: str, frame_rate: float):
    """
    Iterate output frames found from the input glob pattern.

    Using PIL.Image for image loading.

    Mock's starting time as "right now".

    :param filepath_glob: Filepath glob pattern.
    :param frame_rate: FPS in float Hz.

    :return: Iterator returning a tuple of:
        0) BGR format image pixel matrices.
        1) Relative time in seconds of this frame since the beginning of the
           sequence.
    """
    sec_per_frame = 1.0 / frame_rate
    cur_ts = 0
    for filepath in sorted(glob(filepath_glob)):
        img = PIL.Image.open(filepath)
        if img.mode != "RGB":
            img = img.convert("RGB")
        yield np.asarray(img)[:, :, ::-1], cur_ts
        cur_ts += sec_per_frame


def convert_video_to_bag(
    frame_iter: Iterable[Tuple[np.ndarray, float]],
    output_bag_folder,
    output_image_topic="/kitware/PVFramesBGR",
    downsample_rate=None,
):
    """Convert a mp4 video to a ros bag

    :param frame_iter: Iterable of frame matrices.
    :param output_bag_folder: Path to the folder that will be created to contain the ROS bag
    :param output_image_topic: ROS topic to publish the images to in the ROS bag. Must include the namespace
    """
    # Create output bag
    storage_options = rosbag2_py.StorageOptions(
        uri=output_bag_folder, storage_id="sqlite3"
    )
    serialization_fmt = "cdr"
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_fmt,
        output_serialization_format=serialization_fmt,
    )

    bag_writer = rosbag2_py.SequentialWriter()
    bag_writer.open(storage_options, converter_options)
    bag_writer.create_topic(
        rosbag2_py.TopicMetadata(
            name=output_image_topic,
            type="sensor_msgs/msg/Image",
            serialization_format=serialization_fmt,
        )
    )

    bridge = CvBridge()

    # Starting at this so our first increment starts us at frame ID 0.
    frame_id = -1
    for frame, frame_rel_ts in frame_iter:
        frame_id += 1
        # Only proceed if we don't have a down-sample rate specified or if the
        # current frame aligns with the down-sample rate.
        if downsample_rate is not None and frame_id % downsample_rate != 0:
            continue
        print(f"==== FRAME {frame_id} ====")

        # Create image message
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        # split the frame timestamp into sec and nsec
        seconds = frame_rel_ts
        nsec = int((seconds - int(seconds)) * 1_000_000_000)
        seconds = int(seconds)
        image_msg.header.stamp.sec = seconds
        image_msg.header.stamp.nanosec = nsec
        print(f"timestamp: {image_msg.header.stamp}")

        image_msg.header.frame_id = "PVFramesBGR"

        # Write to bag
        try:
            bag_writer.write(
                output_image_topic,
                serialize_message(image_msg),
                # Time position of this message in nanoseconds (integer).
                int(frame_rel_ts * 1e9),
            )
        except Exception as err:
            # Truncating the error message because it printed out the whole image_msg input
            print("error", type(err), str(err)[:400])
            exit(1)

    del bag_writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-fn",
        type=str,
        help=f"Use video mode, and is the path to an mp4 video file.",
    )
    parser.add_argument(
        "-i",
        "--image-file-glob",
        type=str,
        help=(
            "Use image glob mode, and is the glob for images to pick up an "
            "use in lexicographic order."
        ),
    )
    parser.add_argument(
        "--output-bag-folder",
        type=str,
        default="video123",
        help=f"Path to the folder that will be created to contain the ROS bag",
    )
    parser.add_argument(
        "--output-image-topic",
        type=str,
        default="/kitware/PVFramesBGR",
        help=f"ROS topic to publish the images to in the ROS bag. Must include the namespace",
    )
    parser.add_argument(
        "--downsample-rate",
        type=int,
        default=None,
        help=(
            "Only capture every N frames from the input video into the "
            "output bag. E.g. a value of 2 here will mean that only every "
            "other frame is written to the bag, turning an M Hz video info "
            "an M/2 Hz bag."
        ),
    )
    parser.add_argument(
        "--file-frame-rate",
        type=float,
        help=(
            "Frame-rate of the input imagery provided via the "
            "-i/--image-file-glob input mode. This option is only considered "
            "when that input mode is specified, otherwise this value is "
            "ignored. This must be a value greater than zero."
        ),
    )

    args = parser.parse_args()

    # Can only be in one "mode" at a time.
    if None not in [args.video_fn, args.image_file_glob]:
        print(
            "ERROR: Both input selection modes provided. Providing both is "
            "ambiguous."
        )
        exit(1)
    # Need to provide at least *one* of the input selection modes...
    if not any([args.video_fn, args.image_file_glob]):
        print("ERROR: No input selection mode options provided.")
        exit(1)
    # Down-sample value, if given, must be 2 or more.
    if args.downsample_rate is not None and args.downsample_rate < 2:
        print("ERROR: Down-sample rate must be a positive value >1")
        exit(1)

    if args.video_fn is not None:
        frame_iter = iter_video_frames(args.video_fn)
    elif args.image_file_glob is not None:
        # Frame-rate if given must be a positive value.
        if args.file_frame_rate is None:
            print("ERROR: Input frame-rate required.")
            exit(1)
        elif args.file_frame_rate <= 0:
            print("ERROR: Provided input image file frame-rate was not positive.")
            exit(1)
        frame_iter = iter_file_frames(args.image_file_glob, args.file_frame_rate)
    else:
        raise RuntimeError("How did we get here?")

    convert_video_to_bag(
        frame_iter,
        args.output_bag_folder,
        args.output_image_topic,
        args.downsample_rate,
    )


if __name__ == "__main__":
    main()
