from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import torch
import torchvision
from torch.hub import load_state_dict_from_url
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    fasterrcnn_resnet50_fpn,
    model_urls
)
from torchvision.models.detection._utils import (
    overwrite_eps
)
from torchvision.models.detection.backbone_utils import (
    _validate_trainable_layers,
    resnet_fpn_backbone
)
from torchvision import transforms

from angel_msgs.msg import ObjectDescriptors


BRIDGE = CvBridge()


class ResNetDescriptors(FasterRCNN):
    """
    Custom ResNet class to override the forward method call
    of the torch `FasterRCNN` class.
    """

    def __init__(self):
        """
        Creates a model with a `resnet_fpn_backbone`.

        Copied from the torchvision.models.detection.faster_rcnn
        faster_rcnn_resnet50_fpn function.
        """
        pretrained = True
        trainable_backbone_layers = _validate_trainable_layers(
            pretrained, None, 5, 3
        )

        backbone = resnet_fpn_backbone(
            'resnet50',
            False,
            trainable_layers=trainable_backbone_layers
        )

        # COCO 2017 has 91 classes
        super().__init__(backbone=backbone, num_classes=91)

        state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
                                              progress=True)
        self.load_state_dict(state_dict)
        overwrite_eps(self, 0.0)

    def forward(self, images, targets=None):
        """
        Overrides the FasterRCNN forward call to return the output of the
        model backbone.

        Modified version of the torchvision `GeneralizedRCNN` forward call().
        """
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )

        images, targets = self.transform(images, targets)

        features = self.backbone(images.tensors)
        return features


class DescriptorGenerator(Node):
    """
    ROS node that subscribes to `Image` messages and publishes descriptors
    for those images in the form of `ObjectDescriptors` messages.
    """

    def __init__(self):
        torch_device: str = "cpu",
        super().__init__(self.__class__.__name__)

        self._image_topic = self.declare_parameter("image_topic", "PVFrames").get_parameter_value().string_value
        self._desc_topic = self.declare_parameter("descriptor_topic", "ObjectDescriptors").get_parameter_value().string_value
        self._torch_device = self.declare_parameter("torch_device", "cpu").get_parameter_value().string_value
        self._feature_layer = self.declare_parameter("feature_layer", "head").get_parameter_value().string_value

        # Currently supported layers from torch's resnet_fpn_backbone
        allowable_layers = ['0', '1', '2', '3', 'pool', 'head']
        if self._feature_layer not in allowable_layers:
            raise ValueError(
                f"Layer {self._feature_layer} not currently supported"
            )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Torch device? {self._torch_device}")
        log.info(f"Feature layer? {self._feature_layer}")

        self._model = None

        self._subscription = self.create_subscription(
            Image,
            self._image_topic,
            self.listener_callback,
            1
        )
        self._publisher = self.create_publisher(
            ObjectDescriptors,
            self._desc_topic,
            1
        )

        self.transforms = transforms.ToTensor()

    def get_model(self) -> torch.nn.Module:
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            if self._feature_layer == "head":
                # Use default resnet from torch.models
                model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    pretrained=True,
                    progress=False,
                )
            else:
                # Use custom resnet that overrides the forward method call
                # so that we can access the output of the other layers
                model = ResNetDescriptors()

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

        model = self.get_model()

        # Convert ROS Image message to CV2
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")

        # Convert to tensor
        image_tensor = self.transforms(rgb_image)
        image_tensor = image_tensor.unsqueeze(0)

        # Send to model
        output = model(image_tensor)

        if self._feature_layer == "head":
            # Use the object scores as the features
            output = output[0]['scores']
        else:
            output = output[self._feature_layer]

        # Form into object descriptor message
        msg = ObjectDescriptors()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "Object descriptors"
        msg.source_stamp = image.header.stamp

        msg.descriptor_dims = list(output.shape)

        # Flatten the descriptors and convert to list
        msg.descriptors = output.ravel().tolist()

        # Publish
        self._publisher.publish(msg)


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
