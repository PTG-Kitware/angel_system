import json
import time

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from smqtk_core.configuration import from_config_dict
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects

from angel_msgs.msg import ObjectDetection2dSet
from angel_utils.conversion import from_detect_image_objects_result


BRIDGE = CvBridge()


class DescriptorGenerator(Node):

    def __init__(self):
        torch_device: str = "cpu",
        super().__init__(self.__class__.__name__)

        self._image_topic = self.declare_parameter("image_topic", "PVFrames").get_parameter_value().string_value
        self._det_topic = self.declare_parameter("det_topic", "ObjectDescriptors").get_parameter_value().string_value
        self._torch_device = self.declare_parameter("torch_device", "cpu").get_parameter_value().string_value
        self._detector_config = self.declare_parameter("detector_config", "default_object_det_config.json").get_parameter_value().string_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Torch device? {self._torch_device}")
        log.info(f"Detector config: {self._detector_config}")

        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )

        # TODO: change this
        self._publisher = self.create_publisher(
            ObjectDetection2dSet,
            self._det_topic,
            1
        )


    def get_model(self) -> "torch.nn.Module":
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            model = models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                progress=False
            )
            model = model.eval()

            # Transfer the model to the requested device
            if self._torch_device != 'cpu':
                if torch.cuda.is_available():
                    model_device = torch.device(device=self._torch_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError(
                        "Use of CUDA requested but not available."
                    )
            else:
                model_device = torch.device(self._torch_device)

            self._model = model
            self._model_device = model_device

        return model

    def listener_callback(self, image):
        log = self.get_logger()
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif time.time() - self._prev_time > 1:
            log.info(f"Frames rcvd: {self._frames_recvd}")
            self._frames_recvd = 0
            self._prev_time = time.time()

        model = self.get_model()

        # convert ROS Image message to CV2
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")

        # Send to Detector
        output = model(rgb_image)
        log.info(f"output: {output}")



def main():
    rclpy.init()

    descriptor_generator = DescriptorGenerator()
    rclpy.spin(descriptor_generator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    descriptor_generator.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    m
