import os
import subprocess
import argparse
import libtmux
import time
from pathlib import Path
import signal


def run(args):
    tmux_script = "record_ros_bag_activity_only"
    model_name = os.path.basename(args.activity_model).split(".")[0]

    server = libtmux.Server()
    env = os.environ.copy()

    for bag in args.bags:
        done = False
        print(f"Bag: {bag}")

        #env["ROS_NAMESPACE"] = "/kitware"
        env["ROS_NAMESPACE"] = "-/debug"
        env["HL2_IP"] = "192.168.1.101"
        env["CONFIG_DIR"] = os.getenv("ANGEL_WORKSPACE_DIR") + "/src/angel_system_nodes/configs"
        env["MODEL_DIR"] = os.getenv("ANGEL_WORKSPACE_DIR") + "/model_files"

        env["PARAM_ROS_BAG_DIR"] = f"{args.bags_dir}/{bag}"
        env["PARAM_ACTIVITY_MODEL"] = args.activity_model
        output_bag = f"{bag}_{model_name}"
        env["PARAM_ROS_BAG_OUT"] = output_bag

        p1 = subprocess.Popen(["rqt", 
            "-s", "rqt_image_view/ImageView", 
            "--args", env["ROS_NAMESPACE"]+"/PVFramesRGB",
            "--ros-args", 
            "-p", "_image_transport:=raw"], env=env)
        p2 = subprocess.Popen(["ros2", "run", "angel_datahub", "ImageConverter", "--ros-args",
            "-r", env["ROS_NAMESPACE"],
            "-p", "topic_input_images:=PVFramesNV12",
            "-p", "topic_output_images:=PVFramesRGB"], env=env)
        p3 = subprocess.Popen(["ros2", "run", "angel_system_nodes", "berkeley_object_detector", "--ros-args",
            "-r", env["ROS_NAMESPACE"],
            "-p", "image_topic:=PVFramesRGB",
            "-p", "det_topic:=ObjectDetections2d",
            "-p", "det_conf_threshold:=0.4",
            "-p", "model_config:=${ANGEL_WORKSPACE_DIR}/angel_system/berkeley/configs/MC50-InstanceSegmentation/cooking/coffee/stage1/mask_rcnn_R_101_FPN_1x_demo.yaml",
            "-p", "cuda_device_id:=0"], env=env)
        p4 = subprocess.Popen(["ros2", "run", "angel_debug", "Simple2dDetectionOverlay", "--ros-args",
            "-r", env["ROS_NAMESPACE"],
            "-p", "topic_input_images:=PVFramesRGB",
            "-p", "topic_input_det_2d:=ObjectDetections2d",
            "-p", "topic_output_images:=pv_image_detections_2d",
            "-p", "filter_top_k:=5"], env=env)
        p5 = subprocess.Popen(["ros2", "run", "image_transport", "republish", "raw", "compressed", "--ros-args",
            "-r", env["ROS_NAMESPACE"],
            "--remap", "in:=pv_image_detections_2d",
            "--remap", "out/compressed:=pv_image_detections_2d/compressed"], env=env)
        p6 = subprocess.Popen(["ros2", "run", "angel_system_nodes", "activity_from_obj_dets_classifier", "--ros-args",
            "-r", env["ROS_NAMESPACE"],
            "-p", "det_topic:=ObjectDetections2d",
            "-p", "act_topic:=ActivityDetections",
            "-p", "classifier_file:=${MODEL_DIR}/recipe_m2_apply_tourniquet_v0.052.pkl"], env=env)
        p7 = subprocess.Popen(["ros2", "bag", "record",
            env["ROS_NAMESPACE"]+"/ActivityDetections",
            env["ROS_NAMESPACE"]+"/PVFramesRGB",
            "-o", env["PARAM_ROS_BAG_OUT"]], cwd="./ros_bags/", env=env)
        time.sleep(15)
        p8 = subprocess.Popen(["ros2", "bag", "play", env["PARAM_ROS_BAG_DIR"]], env=env).wait()
        if p8 == 0:
            print("'ros2 bag play' command succeeded.")
        else:
            print(f"WARNING: 'ros2 bag play' command failed. error code {p8}")

        # Kill recording process, then wait 30sec for it to fully end
        p7.send_signal(signal.SIGINT)

        # Wait some more until new bag is written
        finished_recording = False
        while not finished_recording:
            finished_recording = os.path.exists(
                f"{args.bags_dir}/{output_bag}/metadata.yaml"
            )
            time.sleep(10)

        # Kill session
        p1.kill()
        p2.kill()
        p3.kill()
        p4.kill()
        p5.kill()
        p6.kill()
        p7.kill()
        p8.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activity_model",
        dest="activity_model",
        type=str,
        default="uho_checkpoint_20221022.ckpt",
    )
    parser.add_argument(
        "--split",
        dest="split",
        type=str,
        required=False,
        help="alias either by name (alex/hannah/brian) or training split (train/val/test) to grab the bag split defined by v1.3 of the dataset",
    )
    parser.add_argument("--bags_dir", dest="bags_dir", type=Path, default="ros_bags")
    parser.add_argument(
        "--bags", dest="bags", nargs="+", type=int, default=[], help="Bag ids"
    )

    args = parser.parse_args()

    # Add pre-defined dataset splits
    if args.split:
        if args.split == "alex" or args.split == "train":
            args.bags.extend([*range(1, 23), *range(25, 41), *range(46, 51)])
        if args.split == "hannah" or args.split == "val":
            args.bags.extend([23, 24, 41, 42, 43, 44, 45])
        if args.split == "brian" or args.split == "test":
            args.bags.extend([51, 52, 53, 54])

    # Reformat bag names
    args.bags = [f"all_activities_{bag_id}" for bag_id in args.bags]

    run(args)


if __name__ == "__main__":
    main()
