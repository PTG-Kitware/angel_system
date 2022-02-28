import time

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN

from angel_msgs.msg import ObjectDetection


BRIDGE = CvBridge()


class ObjectDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("image_topic", "PVFrames")
        self.declare_parameter("det_topic", "ObjectDetections")
        self.declare_parameter("use_cuda", False)
        self.declare_parameter("detection_threshold", 0.8)

        self._image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self._det_topic = self.get_parameter("det_topic").get_parameter_value().string_value
        self._use_cuda = self.get_parameter("use_cuda").get_parameter_value().bool_value
        self._detection_threshold = self.get_parameter("detection_threshold").get_parameter_value().double_value

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Use cuda? {self._use_cuda}")
        log.info(f"Detection threshold: {self._detection_threshold}")

        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            100
        )

        self._publisher = self.create_publisher(
            ObjectDetection,
            self._det_topic,
            10
        )

        self._frames_recvd = 0
        self._prev_time = -1

        # instantiate detector
        self._detector = ResNetFRCNN(img_batch_size=1, use_cuda=self._use_cuda)
        log.info(f"Ready to detect using {self._detector}")

    def listener_callback(self, image):
        log = self.get_logger()
        self._frames_recvd += 1
        if self._prev_time == -1:
            self._prev_time = time.time()
        elif (time.time() - self._prev_time > 1):
            log.info(f"Frames rcvd: {self._frames_recvd}")
            self._frames_recvd = 0
            self._prev_time = time.time()

        # convert NV12 image to RGB
        try:
            yuv_image = np.frombuffer(image.data, np.uint8).reshape(image.height*3//2, image.width)
            rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_NV12)
        except ValueError:
            rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        #print(type(rgb), rgb.size, rgb.shape)
        #print("image_np stuff", rgb.shape, rgb.dtype, image.height, image.width, rgb[0:2])

        # convert np array to tensor
        # image_tensor = transforms.ToTensor()(rgb_image)
        image_tensor = rgb_image

        # send to detector
        start = time.time()
        detections = self._detector.detect_objects([image_tensor])
        end = time.time()
        #print("Time to perform detections", end - start)

        for detection in detections:
            for i in detection:
                bounding_box = i[0]
                class_dict = i[1]
                # Previous use-case: push single highest confidence
                # detection if above set threshold
                highest_conf_item = sorted(
                    class_dict.items(), key=lambda item: item[1],
                    reverse=True)[0]
                highest_conf_label = highest_conf_item[0]
                highest_conf_value = highest_conf_item[1]
                if highest_conf_value > self._detection_threshold:
                    left, top = bounding_box.min_vertex
                    right, bottom = bounding_box.max_vertex
                    det_msg = ObjectDetection(
                        left=float(left),
                        right=float(right),
                        top=float(top),
                        bottom=float(bottom),
                        label_vec=[highest_conf_label],
                        label_confidence_vec=[highest_conf_value],
                    )
                    log.debug(f"Publishing detection msg: {det_msg}")
                    self._publisher.publish(det_msg)


def main():
    rclpy.init()

    object_detector = ObjectDetector()
    '''
    detection_publisher = DetectionPublisher()

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(object_detector)
    executor.add_node(detection_publisher)

    executor.spin()
    '''
    rclpy.spin(object_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    #executor.remove_node(object_detector)
    #executor.remove_node(detection_publisher)
    object_detector.destroy_node()
    #detection_publisher.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
