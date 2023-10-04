import json
import time
from typing import Dict

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ObjectDetection3dSet,
    HeadsetPoseData,
)
from angel_utils import (
    declare_and_get_parameters,
)
from angel_utils.conversion import (
    time_to_int,
    to_confidence_matrix,
)
from hl2ss.viewer import hl2ss
from hl2ss.viewer import hl2ss_3dcv

import os


calibration_path = '/angel_workspace/calibration/'
max_depth = 5

# ROS topic definitions
PARAM_PV_HEAD_POSE_TOPIC = "head_pose_topic"
PARAM_DEPTH_HEAD_POSE_TOPIC = "depth_head_pose_topic"
PARAM_OBJECT_DET_2D_TOPIC = "det_topic"
PARAM_OBJECT_DET_3D_TOPIC = "det_3d_topic"
PARAM_DEPTH_IMAGE_TOPIC = "depth_image_topic"


# HL2SS HELPER FUNCTIONS THAT SHOULD EVENTUALLY MOVE SOMEWHERE ELSE ***********
def to_homogeneous(array):
    return np.concatenate((array, np.ones(array.shape[0:-1] + (1,), dtype=array.dtype)), axis=-1)


def compute_norm(array):
    return np.linalg.norm(array, axis=-1)


def rm_depth_compute_rays(uv2xy, depth_scale):
    xy1 = to_homogeneous(uv2xy)
    scale = compute_norm(xy1) * depth_scale
    return (xy1, scale)


def create_pv_intrinsics(focal_length, principal_point):
    return np.array([
            [-focal_length[0], 0, 0, 0],
            [0, focal_length[1], 0, 0],
            [principal_point[0], principal_point[1], 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32
    )


def rm_depth_normalize(depth, scale):
    return slice_to_block(depth / scale)


def slice_to_block(slice):
    return slice[:, :, np.newaxis]


def rm_depth_to_points(rays, depth):
    return rays * depth


def transform(points, transform4x4):
    return points @ transform4x4[:3, :3] + transform4x4[3, :3].reshape(([1] * (len(points.shape) - 1)).append(3))


def project(points, projection4x4):
    return to_inhomogeneous(transform(points, projection4x4))


def get_homogeneous_component(array):
    return array[..., -1, np.newaxis]


def get_inhomogeneous_component(array):
    return array[..., 0:-1]


def to_homogeneous(array):
    return np.concatenate((array, np.ones(array.shape[0:-1] + (1,), dtype=array.dtype)), axis=-1)


def to_inhomogeneous(array):
    return get_inhomogeneous_component(array) / get_homogeneous_component(array)


def block_to_list(points):
    return points.reshape((-1, points.shape[-1]))
# HL2SS HELPER FUNCTIONS THAT SHOULD EVENTUALLY MOVE SOMEWHERE ELSE ***********


class Convert2dDetsTo3dDets(Node):
    """
    Node that will be used to convert 2d object detections to 3d object
    detections.
    """
    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_DEPTH_IMAGE_TOPIC,),
                (PARAM_PV_HEAD_POSE_TOPIC,),
                (PARAM_DEPTH_HEAD_POSE_TOPIC,),
                (PARAM_OBJECT_DET_2D_TOPIC,),
                (PARAM_OBJECT_DET_3D_TOPIC,),
            ],
        )
        self._depth_image_topic = param_values[PARAM_DEPTH_IMAGE_TOPIC]
        self._pv_head_pose_topic = param_values[PARAM_PV_HEAD_POSE_TOPIC]
        self._depth_head_pose_topic = param_values[PARAM_DEPTH_HEAD_POSE_TOPIC]
        self._det_topic = param_values[PARAM_OBJECT_DET_2D_TOPIC]
        self._det_3d_topic = param_values[PARAM_OBJECT_DET_3D_TOPIC]

        # Stores PV camera pose matrices
        self.pv_poses = []
        # Stores depth camera pose matrices
        self.depth_poses = []
        # Stores received depth images
        self.depth_images = []

        # Get RM Depth Long Throw calibration
        self.calibration_lt = hl2ss_3dcv._load_calibration_rm_depth_longthrow(
            calibration_path + "/rm_depth_longthrow"
        )
        uv2xy = self.calibration_lt.uv2xy
        self.xy1, self.scale = rm_depth_compute_rays(uv2xy, self.calibration_lt.scale)

        # Create subscribers
        self._depth_image_subscription = self.create_subscription(
            Image,
            self._depth_image_topic,
            self.depth_image_callback,
            100
        )
        self._pv_head_pose_subscription = self.create_subscription(
            HeadsetPoseData,
            self._pv_head_pose_topic,
            self.pv_head_pose_callback,
            100
        )
        self._depth_head_pose_subscription = self.create_subscription(
            HeadsetPoseData,
            self._depth_head_pose_topic,
            self.depth_head_pose_callback,
            100
        )
        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.detection_callback,
            100
        )
        # Create publisher
        self._object_3d_publisher = self.create_publisher(
            ObjectDetection3dSet, self._det_3d_topic, 1
        )

    def pv_head_pose_callback(self, pose):
        self.pv_poses.append(pose)

    def depth_head_pose_callback(self, pose):
        self.depth_poses.append(pose)

    def depth_image_callback(self, image):
        self.depth_images.append(image)

    def detection_callback(self, detection):
        """
        General outline:
            - Get detection message
            - Extract out 2d points and timestamps
            - Get closest depth frame
            - Create point cloud
            - Get PV intrinsics and extrinsics matrices
            - Get PV pose
            - Create world-to-image matrix
            - Create image-to-world matrix (inverse of above)
            - Project points from image to world space
        """
        log = self.get_logger()
        if detection.num_detections == 0:
            log.debug("No detections for this image")
            return

        # Find the corresponding PV camera pose for this detection
        for i in reversed(range(len(self.pv_poses))):
            if detection.source_stamp == self.pv_poses[i].header.stamp:
                world_matrix_1d = self.pv_poses[i].world_matrix
                projection_matrix_1d = self.pv_poses[i].projection_matrix

                log.debug(
                    f"time stamps: {self.pv_poses[i].header.stamp} {detection.source_stamp}"
                )
                # Clear out our pose list
                # Assuming we won't get image frame detections out of order
                self.pv_poses = self.pv_poses[i:]
                break

        # Find the closest depth camera image
        min_diff = None
        for d in self.depth_images:
            diff = abs(time_to_int(d.header.stamp) - time_to_int(detection.source_stamp))
            if min_diff is None or diff < min_diff:
                min_diff = diff
                depth_image = d

        # Get the depth pose for this depth image
        for i in reversed(range(len(self.depth_poses))):
            if depth_image.header.stamp == self.depth_poses[i].header.stamp:
                depth_pose = np.array(self.depth_poses[i].world_matrix).reshape((4, 4))

                # Clear out our pose list
                # Assuming we won't get image frame detections out of order
                self.depth_poses = self.depth_poses[i:]
                break

        print(f"Detection source stamp {detection.source_stamp}")
        print(f"PV frame world matrix {world_matrix_1d}")
        print(f"Detection projection matrix {projection_matrix_1d}")

        # Clear the stored depth images
        self.depth_images = []

        # Update PV intrinsics/pose
        pv_pose = np.array(world_matrix_1d).reshape((4, 4))
        pv_intrinsics = np.array(projection_matrix_1d).reshape((4, 4))
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        pv_intrinsics, pv_extrinsics, _ = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Preprocess frames
        depth_payload = np.frombuffer(depth_image.data, dtype="uint16")
        depth_payload = depth_payload.reshape((depth_image.height, depth_image.width))
        depth = rm_depth_normalize(depth_payload, self.scale)
        depth[depth > max_depth] = 0  # Depth buffer image (288, 320, 1). (H, W)

        # Build pointcloud
        points = rm_depth_to_points(depth, self.xy1)  # depth * xy1

        # Create depth camera to world transformation matrix
        depth_to_world = (
            hl2ss_3dcv.camera_to_rignode(self.calibration_lt.extrinsics) @
            hl2ss_3dcv.reference_to_world(depth_pose)
        )
        # Transform depth camera coordinates to world coordinate system
        # This is the 3D world coordinate of every pixel in the depth image
        # (I THINK)
        points = transform(points, depth_to_world)
        #print("world points", points.shape)

        # Create world coordinate system to image space transformation matrix
        world_to_image = (
            hl2ss_3dcv.world_to_reference(pv_pose) @
            hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @
            hl2ss_3dcv.camera_to_image(pv_intrinsics)
        )

        # Project 3D world coordinates of depth camera to image space
        # This is the image space pixel coordinate of every point in the depth
        # image
        pixels = project(points, world_to_image)

        map_u = pixels[:, :, 0]  # X coordinates
        map_v = pixels[:, :, 1]  # Y coordinates

        image_to_world = np.linalg.inv(world_to_image)

        # Start creating the 3D detection set message
        det_3d_set_msg = ObjectDetection3dSet()
        det_3d_set_msg.header.stamp = self.get_clock().now().to_msg()
        det_3d_set_msg.header.frame_id = detection.header.frame_id
        det_3d_set_msg.source_stamp = detection.source_stamp

        # Convert the 2d detections to 3d
        det_conf_mat = to_confidence_matrix(detection)
        for i in range(detection.num_detections):
            object_type = sorted(zip(det_conf_mat[i], detection.label_vec))[-1][1]
            print(object_type)

            # Get pixel positions of detected object box and form into box corners
            min_vertex0 = detection.left[i]
            min_vertex1 = detection.top[i]
            max_vertex0 = detection.right[i]
            max_vertex1 = detection.bottom[i]
            pixel_coordinates = [
                [int(min_vertex0), int(min_vertex1)],
                [int(min_vertex0), int(max_vertex1)],
                [int(max_vertex0), int(max_vertex1)],
                [int(max_vertex0), int(min_vertex1)],
            ]

            fx = pv_intrinsics[0][0]
            fy = pv_intrinsics[1][1]
            cx = pv_intrinsics[2][0]
            cy = pv_intrinsics[2][1]

            map_to_3d_successful = True
            world_coordinates = []
            for p in pixel_coordinates:
                x = ux = p[0]
                y = uy = p[1]
                print(f"Attempting to map pixel {x}, {y}")

                # https://github.com/jdibenes/hl2ss/issues/21
                # Hack method to map the PV pixel in a depth image
                # 1. Create a same-size empty image;
                frame = np.zeros((760, 429))  # PV camera width and height
                # 2. Set the target pixel (x,y) a non-zero value, e.g. 255;
                frame[x][y] = 255.0
                nonzero = np.nonzero(frame)
                # 3. Remap
                frame_mapped = cv2.remap(frame, map_u, map_v, cv2.INTER_CUBIC)
                # 4. Find the non-zero value, get its index (x', y');
                nonzero_x, nonzero_y = np.nonzero(frame_mapped)
                if len(nonzero_x) == 0 or len(nonzero_y) == 0:
                    # No corresponding depth point found
                    map_to_3d_successful = False
                    break

                # If this works, then we have the index of the depth camera
                # coordinates corresponding to our PV camera coordinates. Then,
                # we can lookup the the world coordinates of this in the
                # `points` array.

                # 5. Get depth info at the position of (x', y').
                point_depth = depth[nonzero_x[0], nonzero_y[0]]
                Z = point_depth[0]
                print("Computed depth for this pixel : ", Z)

                # World point could be this (leaning towards this)
                print("Computed world point for pixel : ", points[nonzero_x[0], nonzero_y[0]])

                # or this
                X_3D = Z * (x - cx) / fx
                Y_3D = Z * (y - cy) / fy
                Z_3D = Z
                print(X_3D, Y_3D, Z_3D)

                # TODO: unsure if this is needed
                # or this
                image_points = np.array([[[X_3D, Y_3D, Z_3D]]])
                world_points2 = transform(image_points, image_to_world)
                print(world_points2)

                world_coordinates.append(points[nonzero_x[0], nonzero_y[0]])

            if not map_to_3d_successful:
                # TODO: another option is if we do get a valid depth from one
                # of the four corners, just use the depth for that one that is
                # valid for the others
                print(f"failed to map detection at {i}")
                continue

            print(f"Detection {i} mapped successfully!")
            # Since we were able to find this object's 3D position,
            # add it to 3d detection message
            det_3d_set_msg.object_labels.append(object_type)
            det_3d_set_msg.num_objects += 1

            for p in range(4):
                point_3d = Point()

                log.info(f"Scene position {world_coordinates[p]}")
                point_3d.x = float(world_coordinates[p][0])
                point_3d.y = float(world_coordinates[p][1])
                point_3d.z = float(world_coordinates[p][2])

                if p == 0:
                    det_3d_set_msg.left.append(point_3d)
                elif p == 1:
                    det_3d_set_msg.top.append(point_3d)
                elif p == 2:
                    det_3d_set_msg.right.append(point_3d)
                elif p == 3:
                    det_3d_set_msg.bottom.append(point_3d)

        # Publish the 3d object detection message
        self._object_3d_publisher.publish(det_3d_set_msg)

def main():
    rclpy.init()

    conversion_node = Convert2dDetsTo3dDets()
    rclpy.spin(conversion_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    conversion_node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
