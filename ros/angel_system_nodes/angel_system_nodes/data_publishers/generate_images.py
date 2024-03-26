import time
import os
from cv_bridge import CvBridge
import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from random import sample 
from glob import glob
from PIL import Image as pillow_image

def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types from directories, optionally recursively.

    Parameters:
        path (str): Root directory path.
        types (list): List of file types (extensions) to be extracted.
        recursive (bool, optional): Search for files in subsequent directories if True. Default is False.

    Returns:
        list: List of file paths with full paths.
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files

def random_image(height=720, width=1280, channels=3, images_root="/angel_workspace/model_files/sample_images/"):
    """
    Generate a new random
    :param height: Pixel height
    :param width: Pixel width
    :param channels: image channels
    :return: Numpy image matrix
    """
    # print(f"images_root: {images_root}")
    images_paths = dictionary_contents(images_root, types=["*.png"])
    # print(f"images_paths: {images_paths}")
    image_path = sample(images_paths, 1)[0]
    # print(f"image_path: {image_path}")
    
    image = np.array(pillow_image.open(image_path))
    print(image.max())

    return image
    # return np.random.randint(0, 255, (height, width, channels), np.uint8)


bridge = CvBridge()


class GenerateImages(Node):
    """
    It is the node that defines the inputs and outputs at the topic level.

    :param fps_avg_window: The number of previous frame timings to consider for
        reporting publishing rate.
    """

    def __init__(
        self,
        node_name: str = None,
        output_topic_name: str = "/image",
        fps: float = 30,
        fps_avg_window: int = 30,
        height: int = 720,
        width: int = 1280,
        rgb: bool = True,
    ):
        if node_name is None:
            node_name = self.__class__.__name__
        super().__init__(node_name)
        self._output_topic_name = output_topic_name
        self._fps = fps
        self._fps_window = fps_avg_window
        self._img_shape = (height, width) + ((3,) if rgb else ())
        # Tracking the number of images we have published.
        self._img_count = 0

        self._frame_time_q = []  # FIFO queue in spirit
        self._last_pub_time = None

        self.pub_generated_image = self.create_publisher(
            Image,
            # CompressedImage,
            self._output_topic_name,
            10,  # TODO: Learn QoS meanings
        )

        timer_period = 1.0 / self._fps
        self.timer = self.create_timer(timer_period, self.generate_publish_image)

    def generate_publish_image(self) -> None:
        """
        Generate a new random image and publish it to the set topic.
        """
        log = self.get_logger()
        new_img = random_image(*self._img_shape)
        img_msg = bridge.cv2_to_imgmsg(new_img, encoding="rgb8")
        # img_msg = bridge.cv2_to_compressed_imgmsg(new_img, dst_format='jpg')
        self.pub_generated_image.publish(img_msg)
        time_since_last_pub = time.time()

        # Manage publish rate measurement
        window_size = self._fps_window
        frame_time_q = self._frame_time_q
        last_pub_time = self._last_pub_time
        if last_pub_time is not None:
            if len(frame_time_q) == window_size:
                frame_time_q.pop(0)
            frame_time_q.append(time_since_last_pub - last_pub_time)
            avg_time = np.mean(frame_time_q)
            avg_fps = f"{1 / avg_time:.2f}"
        else:
            avg_fps = "None"

        log.info(
            f"[{self._output_topic_name}] "
            f"Published image #{self._img_count} with shape {new_img.shape} "
            f"(fps: {avg_fps})"
        )

        self._last_pub_time = time_since_last_pub
        self._img_count += 1


def main(args=None):
    # Not using the default main builder due to non-standard main functionality
    # desired by this.

    # Hard code some parameters until we (re)learn parameterization.
    rclpy.init(args=args)
    log = rclpy.logging.get_logger("main")

    # gen_images = GenerateImages()
    # rclpy.spin(gen_images)
    # gen_images.destroy_node()

    num_nodes = 4

    executor = rclpy.executors.SingleThreadedExecutor()
    # executor = rclpy.executors.MultiThreadedExecutor()
    node_list = []
    for i in range(num_nodes):
        node = GenerateImages(
            node_name=f"generator_{i}",
            output_topic_name=f"/image_{i}",
        )
        node_list.append(node)
        executor.add_node(node)

    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        log.warn("Interrupt/shutdown signal received.")
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        for node in node_list:
            executor.remove_node(node)
            node.destroy_node()

        log.info("Final try-shutdown")
        rclpy.try_shutdown()
