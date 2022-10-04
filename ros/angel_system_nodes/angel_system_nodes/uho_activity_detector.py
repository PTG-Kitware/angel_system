# TODO: This is added so that the the angel_system/uho/src folder
# is found when torch attempts to load the saved model checkpoints.
# Is there a better way to do this?
import sys
sys.path.append("angel_system/uho")

from cv_bridge import CvBridge
import cv2
import message_filters as mf
import numpy as np
import rclpy
from rclpy.node import Node
import torch

from angel_msgs.msg import (
    ActivityDetection,
    HandJointPosesUpdate,
    ObjectDetection2dSet
)
from sensor_msgs.msg import Image

from angel_system.uho.src.models.components.transformer import TemTRANSModule
from angel_system.uho.src.models.components.unified_fcn import UnifiedFCNModule
from angel_system.uho.src.models.unified_ho_module import UnifiedHOModule


BRIDGE = CvBridge()


class UHOActivityDetector(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        # Declare ROS topics
        self._image_topic = (
            self.declare_parameter("image_topic", "PVFramesRGB")
            .get_parameter_value()
            .string_value
        )
        self._hand_topic = (
            self.declare_parameter("hand_pose_topic", "HandJointPoseData")
            .get_parameter_value()
            .string_value
        )
        self._obj_det_topic = (
            self.declare_parameter("obj_det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._torch_device = (
            self.declare_parameter("torch_device", "cuda")
            .get_parameter_value()
            .string_value
        )
        self._det_topic = (
            self.declare_parameter("det_topic", "ActivityDetections")
            .get_parameter_value()
            .string_value
        )
        self._frames_per_det = (
            self.declare_parameter("frames_per_det", 32)
            .get_parameter_value()
            .integer_value
        )
        self._model_checkpoint = (
            self.declare_parameter("model_checkpoint",
                                   "/angel_workspace/model_files/epoch_090.ckpt")
            .get_parameter_value()
            .string_value
        )
        self._labels_file = (
            self.declare_parameter("labels_file",
                                   "/angel_workspace/model_files/uho_epoch_090_labels.txt")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Image topic: {self._image_topic}")
        log.info(f"Hand topic: {self._hand_topic}")
        log.info(f"Object detections topic: {self._obj_det_topic}")
        log.info(f"Device? {self._torch_device}")
        log.info(f"Frames per detection: {self._frames_per_det}")
        log.info(f"Checkpoint: {self._model_checkpoint}")
        log.info(f"Labels: {self._labels_file}")

        self.subscription_list = []
        # Image subscription
        self.subscription_list.append(mf.Subscriber(self, Image, self._image_topic))
        # Hand pose subscription
        self.subscription_list.append(mf.Subscriber(self, HandJointPosesUpdate, self._hand_topic))
        self.time_sync = mf.ApproximateTimeSynchronizer(
            self.subscription_list,
            1500, # TODO: queue size
            10  # TODO: slop (delay msgs can be synchronized in seconds)
        )

        self.time_sync.registerCallback(self.multimodal_listener_callback)

        # Object detections subscriber
        self._obj_det_subscriber = self.create_subscription(
            ObjectDetection2dSet,
            self._obj_det_topic,
            self.obj_det_callback,
            1000 # TODO: queue size
        )

        self._publisher = self.create_publisher(
            ActivityDetection,
            self._det_topic,
            1
        )

        # Stores the data until we have enough to send to the detector
        self._frames = []
        self._hand_poses = dict(
            lhand=[],
            rhand=[],
        )
        self._frame_stamps = []
        self._obj_dets = []

        # TODO - parameterize this?
        self._topk = 5

        # Instantiate the activity detector models
        fcn = UnifiedFCNModule(net="resnext", num_cpts=21, obj_classes=9, verb_classes=12)
        temporal = TemTRANSModule(act_classes=27, hidden=256, dropout=0.1, depth=6)

        self._detector: UnifiedHOModule = UnifiedHOModule(
            fcn=fcn,
            temporal=temporal,
            checkpoint=self._model_checkpoint,
            device=self._torch_device,
            labels_file=self._labels_file
        )
        self._detector.eval()
        self._detector = self._detector.to(device=self._torch_device)
        log.info(f"UHO Detector initialized")

    def multimodal_listener_callback(self, image, hand_pose):
        """
        Callback function images + hand poses. Messages are synchronized with
        with the ROS time synchronizer.
        """
        # Convert ROS img msg to CV2 image and add it to the frame stack
        rgb_image = BRIDGE.imgmsg_to_cv2(image, desired_encoding="rgb8")
        rgb_image_np = np.asarray(rgb_image)

        self._frames.append(rgb_image_np)

        # Store the image timestamp
        self._frame_stamps.append(image.header.stamp)

        lhand, rhand = self.get_hand_pose_from_msg(hand_pose)
        self._hand_poses['lhand'].append(lhand)
        self._hand_poses['rhand'].append(rhand)

    def obj_det_callback(self, msg):
        """
        Callback for the object detection set message. Once there is a object
        detection message for each frame in the current set, the activity detector
        model is called and a new activity detection message is published with
        the current activity predictions.
        """
        log = self.get_logger()

        if msg.num_detections < self._topk:
            log.warn(f"Received msg with less than {self._topk} detections.")
            return

        self._obj_dets.append(msg)

        if len(self._frames) >= self._frames_per_det:
            frame_stamp_set = self._frame_stamps[:self._frames_per_det]
            ready_to_predict = False

            # If the source stamp for this detection message is after or equal to
            # the last frame stamp in the current set, then we have all of the
            # detections and can move onto processing this set of frames.
            if msg.source_stamp.sec > frame_stamp_set[-1].sec:
                #print(frame_stamp_set, msg.source_stamp.sec)
                ready_to_predict = True
            elif msg.source_stamp.sec == frame_stamp_set[-1].sec:
                if msg.source_stamp.nanosec >= frame_stamp_set[-1].nanosec:
                    #print(frame_stamp_set, msg.source_stamp.sec)
                    ready_to_predict = True
            else:
                # Keep waiting
                ready_to_predict = False

            # Need to wait until the object detector has processed all of these frames
            if not ready_to_predict:
                log.info(f"Waiting for object detection results")
                return

            frame_set = self._frames[:self._frames_per_det]
            lhand_pose_set = self._hand_poses['lhand'][:self._frames_per_det]
            rhand_pose_set = self._hand_poses['rhand'][:self._frames_per_det]

            aux_data = dict(
                lhand=lhand_pose_set,
                rhand=rhand_pose_set,
                dets=[],
                bbox=[],
            )

            # Format the object detections into descriptors and bboxes
            idxs_to_remove = []
            for idx, det in enumerate(self._obj_dets):
                #print(det.source_stamp, frame_stamp_set[0], frame_stamp_set[-1])
                if det.num_detections == 0:
                    log.info(f"no dets, det source: {det.source_stamp}")
                    continue

                # Check this detection is within the range of time for the current
                # frame set
                if det.source_stamp.sec < frame_stamp_set[0].sec:
                    # Detection is before the first frame in this set,
                    # so we can remove it
                    idxs_to_remove.append(idx)
                    continue
                elif det.source_stamp.sec == frame_stamp_set[0].sec:
                    if det.source_stamp.nanosec < frame_stamp_set[0].nanosec:
                        # Detection is before the first frame in this set,
                        # so we can remove it
                        idxs_to_remove.append(idx)
                        continue
                elif det.source_stamp.sec > frame_stamp_set[-1].sec:
                    # Detection is after the last frame in this set,
                    # so keep it for later
                    continue
                elif det.source_stamp.sec == frame_stamp_set[-1].sec:
                    if det.source_stamp.nanosec > frame_stamp_set[-1].nanosec:
                        # Detection is after the last frame in this set,
                        # so keep it for later
                        continue

                det_descriptors = (
                    torch.Tensor(det.descriptors).reshape((det.num_detections, det.descriptor_dim))
                )
                aux_data['dets'].append(det_descriptors[:self._topk])

                bboxes = [
                    torch.Tensor((det.left[i], det.top[i], det.right[i], det.bottom[i])) for i in range(self._topk)
                ]
                bboxes = torch.stack(bboxes)

                aux_data['bbox'].append(bboxes)

            # Check if we didn't get any detections in the time range of the frame set
            if len(aux_data["dets"]) == 0 or len(aux_data["bbox"]) == 0:
                aux_data["dets"] = [torch.zeros((self._topk, msg.descriptor_dim))]
                aux_data["bbox"] = [torch.zeros((self._topk, 4))]

            # Inference!
            activities_detected, labels = self._detector.forward(frame_set, aux_data)

            # Create and publish the ActivityDetection msg
            activity_msg = ActivityDetection()

            # This message time
            activity_msg.header.stamp = self.get_clock().now().to_msg()

            # Trace to the source
            activity_msg.header.frame_id = "Activity detection"
            activity_msg.source_stamp_start_frame = frame_stamp_set[0]
            activity_msg.source_stamp_end_frame = frame_stamp_set[-1]

            activity_msg.label_vec = labels
            activity_msg.conf_vec = activities_detected[0].squeeze().tolist()

            # Publish!
            self._publisher.publish(activity_msg)
            log.info(f"Activities detected: {activities_detected}")
            log.info(f"Top activity detected: {activities_detected[1]}")

            # Clear out stored frames, aux_data, and timestamps
            self._frames = self._frames[self._frames_per_det:]
            self._frame_stamps = self._frame_stamps[self._frames_per_det:]
            self._hand_poses['lhand'] = self._hand_poses['lhand'][self._frames_per_det:]
            self._hand_poses['rhand'] = self._hand_poses['rhand'][self._frames_per_det:]

            for i in sorted(idxs_to_remove, reverse=True):
                del self._obj_dets[i]

    def get_hand_pose_from_msg(self, msg):
        """
        Formats the hand pose information from the ROS hand pose message
        into the format required by activity detector model.
        """
        hand_joints = [{"joint": m.joint,
                        "position": [ m.pose.position.x,
                                      m.pose.position.y,
                                      m.pose.position.z]}
                      for m in msg.joints]

        # Rejecting joints not in OpenPose hand skeleton format
        reject_joint_list = ['ThumbMetacarpalJoint',
                            'IndexMetacarpal',
                            'MiddleMetacarpal',
                            'RingMetacarpal',
                            'PinkyMetacarpal']
        joint_pos = []
        for j in hand_joints:
            if j["joint"] not in reject_joint_list:
                joint_pos.append(j["position"])
        joint_pos = np.array(joint_pos).flatten()

        if msg.hand == 'Right':
            rhand = joint_pos
            lhand = np.zeros_like(joint_pos)
        elif msg.hand == 'Left':
            lhand = joint_pos
            rhand = np.zeros_like(joint_pos)
        else:
            lhand = np.zeros_like(joint_pos)
            rhand = np.zeros_like(joint_pos)

        return lhand, rhand


def main():
    rclpy.init()

    detector = UHOActivityDetector()

    rclpy.spin(detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detector.destroy_node()

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()