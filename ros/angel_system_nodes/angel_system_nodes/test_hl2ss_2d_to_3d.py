# 3D segmentation from 2d segmentation using mmdetection

import multiprocessing as mp
import numpy as np
from hl2ss.viewer import hl2ss
from hl2ss.viewer import hl2ss_mp
from hl2ss.viewer import hl2ss_3dcv

import json
import time
from typing import Dict

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from smqtk_core.configuration import from_config_dict

from angel_msgs.msg import (
    ObjectDetection2dSet,
    ObjectDetection3dSet,
    SpatialMesh,
    HeadsetPoseData,
)


import os
import open3d as o3d

#from mmdet.apis import inference_detector, init_detector
#from mmdet.evaluation import INSTANCE_OFFSET


host = '10.0.0.170'
calibration_path = '/angel_workspace/calibration/'

device = 'cpu'
score_thr = 0.3
wait_time = 1
buffer_length = 5
max_depth = 5


def hl2ss_main():
    enable = True

    # Start PV Subsystem ------------------------------------------------------
    """
    hl2ss.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)
    """

    # Get RM Depth Long Throw calibration -------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_lt = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, calibration_path)

    uv2xy = calibration_lt.uv2xy

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

    pv_width = 640
    pv_height = 360
    pv_framerate = 15
    PV_BITRATE = 1 * 1024 * 1024

    xy1, scale = rm_depth_compute_rays(uv2xy, calibration_lt.scale)
    print("depth ray done, xy1 shape", xy1.shape)

    # Create visualizers ------------------------------------------------------
    #cv2.namedWindow('Detections')
    print("created window")

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    main_pcd = o3d.geometry.PointCloud()
    first_geometry = True

    # Start PV and RM Depth Long Throw streams --------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(
        hl2ss.StreamPort.PERSONAL_VIDEO,
        hl2ss.rx_decoded_pv(
            host,
            hl2ss.StreamPort.PERSONAL_VIDEO,
            hl2ss.ChunkSize.PERSONAL_VIDEO,
            hl2ss.StreamMode.MODE_1,
            pv_width,
            pv_height,
            pv_framerate,
            hl2ss.VideoProfile.H265_MAIN,
            PV_BITRATE,
            format="bgr24",
        )
    )
    producer.configure(
        hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
        hl2ss.rx_rm_depth_longthrow(
            host,
            hl2ss.StreamPort.RM_DEPTH_LONGTHROW,
            chunk=hl2ss.ChunkSize.RM_DEPTH_LONGTHROW,
            mode=hl2ss.StreamMode.MODE_1,
            png_filter=hl2ss.PngFilterMode.Paeth,
        )
    )
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, buffer_length * pv_framerate)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_LONGTHROW, buffer_length * hl2ss.Parameters_RM_DEPTH_LONGTHROW.FPS)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    manager = mp.Manager()
    consumer = hl2ss_mp.consumer()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_depth = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_LONGTHROW, manager, None)
    sink_pv.get_attach_response()
    sink_depth.get_attach_response()

    # Init Detector -----------------------------------------------------------
    #model = init_detector(config, checkpoint, device=device)

    # Main loop ---------------------------------------------------------------
    while (enable):
        # Get RM Depth Long Throw frame and nearest (in time) PV frame --------
        data_depth = sink_depth.get_most_recent_frame()
        if ((data_depth is None) or (not hl2ss.is_valid_pose(data_depth.pose))):
            continue

        _, data_pv = sink_pv.get_nearest(data_depth.timestamp)
        if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
            continue

        print(data_pv.payload.focal_length, data_pv.payload.principal_point)
        data_depth_decoded = hl2ss.decode_rm_depth_longthrow(data_depth.payload)
        print(data_depth_decoded)

        # Update PV intrinsics ------------------------------------------------
        # PV intrinsics may change between frames due to autofocus
        pv_intrinsics = create_pv_intrinsics(
            data_pv.payload.focal_length,
            data_pv.payload.principal_point
        )
        pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        pv_intrinsics, pv_extrinsics, _ = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        # Preprocess frames ---------------------------------------------------
        depth = rm_depth_normalize(data_depth_decoded.depth, scale)
        depth[depth > max_depth] = 0
        print(depth.shape)

        # Inference -----------------------------------------------------------
        frame = data_pv.payload.image
        print(frame.shape)
        #result = inference_detector(model, frame)
        #mask = result.pred_panoptic_seg.sem_seg.numpy()
        #print(mask.shape)

        cv2.imwrite("pv_frame.png", frame)
        cv2.imwrite("depth_frame.png", depth)

        # Build pointcloud ----------------------------------------------------
        points = rm_depth_to_points(depth, xy1)
        depth_to_world = hl2ss_3dcv.camera_to_rignode(calibration_lt.extrinsics) @ hl2ss_3dcv.reference_to_world(data_depth.pose)
        print("Depth to world", depth_to_world)
        points = transform(points, depth_to_world)
        print("world points", points.shape)

        # Project pointcloud image --------------------------------------------
        world_to_image = hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)
        print("world to image", world_to_image)
        pixels = project(points, world_to_image)
        print("pixels", pixels.shape)

        map_u = pixels[:, :, 0]
        map_v = pixels[:, :, 1]
        print("map u", map_u.shape)
        print("map v", map_v.shape)

        # Get 3D points labels and colors -------------------------------------
        #labels = cv2.remap(mask, map_u, map_v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=model.num_classes)
        #cv2.imwrite("results_mapped.png", labels)
        rgb = cv2.remap(frame, map_u, map_v, cv2.INTER_NEAREST)
        cv2.imwrite("pv_frame_mapped.png", rgb)
        print("rgb", rgb.shape)

        image_to_world = np.linalg.inv(world_to_image)
        print("image_to_world", image_to_world)

        # Workspace to convert 2d to 3d
        '''
        fx = data_pv.payload.focal_length[0]
        fy = data_pv.payload.focal_length[1]
        cx = data_pv.payload.principal_point[0]
        cy = data_pv.payload.principal_point[1]
        '''

        twod_points = [[200, 200], [205, 200], [200, 205], [205, 205]]
        world_points = []
        color_points = []
        for p in twod_points:
            print("2d point", p)
            '''
            Z = depth[p[0], p[1]]
            Z2 = points[p[0], p[1]]
            ux = map_u[p[0], p[1]]
            uy = map_v[p[0], p[1]]
            print("depth", Z)
            print("depth v2", Z2)
            print("ux", ux)
            print("uy", uy)
            X = Z / fx * (ux - cx)
            Y = Z / fy * (uy - cy)
            d_point = [X[0], Y[0], Z[0]]
            print("3d point", d_point)
            world_points.append(d_point)
            color_points.append([255, 0, 0])
            '''
            image_points = np.array([[[p[0], p[1], 1]]])
            point_matrix = np.array([[p[0]], [p[1]], [1], [1]])
            world_points1 = np.matmul(image_to_world, point_matrix)[:3]
            world_points2 = transform(image_points, image_to_world)
            print(world_points1, world_points2)

        points = block_to_list(points)
        #labels = labels.reshape((-1,))
        rgb = block_to_list(rgb)

        # Get class colors ----------------------------------------------------
        #kinds = labels % INSTANCE_OFFSET
        #instances = labels // INSTANCE_OFFSET

        #class_colors = np.array([list(PALETTE[kind]) for kind in kinds], dtype=np.uint8)

        # Get final color -----------------------------------------------------
        #colors = (0.5 * class_colors) + (0.5 * rgb)

        # Remove invalid points -----------------------------------------------
        #select = depth.reshape((-1,)) > 0

        #points = points[select, :]
        #colors = colors[select, :] / 255
        print("points", points.shape)
        print("colors", colors.shape)

        # Visualize results ---------------------------------------------------
        #detections = model.show_result(frame, result, score_thr=score_thr, mask_color=PALETTE)
        #mmcv.imshow(detections, 'Detections', wait_time)

        test_point = o3d.geometry.PointCloud()
        test_point.points = o3d.utility.Vector3dVector(world_points)
        test_point.points = o3d.utility.Vector3dVector(color_points)

        #main_pcd.points = o3d.utility.Vector3dVector(points)
        #main_pcd.colors = o3d.utility.Vector3dVector(colors)

        if (first_geometry):
            vis.add_geometry(main_pcd)
            vis.add_geometry(test_point)
            first_geometry = False
        else:
            vis.update_geometry(main_pcd)

        vis.poll_events()
        vis.update_renderer()

        break

    # Stop PV and RM Depth Long Throw streams ---------------------------------
    sink_pv.detach()
    sink_depth.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_LONGTHROW)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)


class HL2SSTester(Node):
    """
    Node that will be used to convert 2d object detections to 3d object
    detections.
    """
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._det_topic = (
            self.declare_parameter("det_topic", "ObjectDetections")
            .get_parameter_value()
            .string_value
        )
        self._headset_pose_topic = (
            self.declare_parameter("headset_pose_topic", "HeadsetPoseData")
            .get_parameter_value()
            .string_value
        )
        self._det_3d_topic = (
            self.declare_parameter("det_3d_topic", "ObjectDetections3d")
            .get_parameter_value()
            .string_value
        )

        log = self.get_logger()
        log.info(f"Detection topic: {self._det_topic}")
        log.info(f"Detection3d topic: {self._det_3d_topic}")
        log.info(f"Pose topic: {self._headset_pose_topic}")

        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet, self._det_topic, self.detection_callback, 100
        )

        self._headset_pose_subscription = self.create_subscription(
            HeadsetPoseData, self._headset_pose_topic, self.headset_pose_callback, 100
        )

        self._object_3d_publisher = self.create_publisher(
            ObjectDetection3dSet, self._det_3d_topic, 1
        )

        # TODO: Need to create PV image subscriber
        # TODO: Need to create depth image subscriber
        # TODO: Need to create object detection subscriber
        #hl2ss_main()

    def headset_pose_callback(self, pose):
        log = self.get_logger()
        self.poses.append(pose)

    def detection_callback(self, detection):
        log = self.get_logger()

        log.info("got detection")

        if detection.num_detections == 0:
            log.debug("No detections for this image")
            return

        # TODO: do 2d to 3d stuff here

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

def main():
    rclpy.init()

    activity_detector = HL2SSTester()

    rclpy.spin(activity_detector)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    activity_detector.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
