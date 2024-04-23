import cv2
import argparse
import rclpy
import rclpy.time
import rclpy.duration
import time

import rosbag2_py
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.serialization import serialize_message


def convert_video_to_bag(
    video_fn, output_bag_folder, output_image_topic="/kitware/image_0"
):
    """Convert a mp4 video to a ros bag

    :param video_fn: Path to an mp4 video file
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

    # Read video
    cam = cv2.VideoCapture(video_fn)

    frame_id = 0
    time_nanosec = time.time_ns()
    start_ts = rclpy.time.Time(nanoseconds=time_nanosec)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        print(f"==== FRAME {frame_id} ====")
        # Create timestamp
        frame_secs = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000

        frame_ts = start_ts + rclpy.duration.Duration(seconds=frame_secs)
        frame_ts_msg = frame_ts.to_msg()
        print("timestamp", frame_ts)

        # Create image message
        image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = frame_ts_msg
        image_msg.header.frame_id = "PVFramesBGR"

        # Write to bag
        try:
            bag_writer.write(
                output_image_topic,
                serialize_message(image_msg),
                frame_ts.nanoseconds,
            )
        except Exception as err:
            # Truncating the error message because it printed out the whole image_msg input
            print("error", type(err), str(err)[:400])
            exit(1)

        frame_id += 1

    cam.release()
    del bag_writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-fn",
        type=str,
        default="video123.mp4",
        help=f"Path to an mp4 video file",
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
        default="/kitware/image_0",
        help=f"ROS topic to publish the images to in the ROS bag. Must include the namespace",
    )

    args = parser.parse_args()

    convert_video_to_bag(args.video_fn, args.output_bag_folder, args.output_image_topic)


if __name__ == "__main__":
    main()
