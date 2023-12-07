import time

import numpy as np
import rclpy
from rclpy.node import Node
from angel_msgs.msg import (
    ObjectDetection2dSet,
    ObjectDetection3dSet,
    SpatialMesh,
    HeadsetPoseData,
)
from angel_msgs.srv import QueryImageSize
from angel_utils import make_default_main
from angel_utils.conversion import to_confidence_matrix
from geometry_msgs.msg import Point

import trimesh
import trimesh.viewer

# NOTE: These values were extracted from the projection matrix provided by the
# Unity main camera with the Windows MR plugin (now deprecated).
# For some unknown reason, the values provided by the Open XR plugin are
# different and do not provide the correct results. In the future, we should
# figure out why they are different or extract the focal length values from the
# MediaFrameReader which provides the frames, instead of the Unity main camera.
FOCAL_LENGTH_X = 1.6304
FOCAL_LENGTH_Y = 2.5084


class SpatialMapSubscriber(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        self._spatial_map_topic = (
            self.declare_parameter("spatial_map_topic", "SpatialMapData")
            .get_parameter_value()
            .string_value
        )
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
        log.info(f"Spatial map topic: {self._spatial_map_topic}")
        log.info(f"Detection topic: {self._det_topic}")
        log.info(f"Detection3d topic: {self._det_3d_topic}")
        log.info(f"Pose topic: {self._headset_pose_topic}")

        self._spatial_mesh_subscription = self.create_subscription(
            SpatialMesh, self._spatial_map_topic, self.spatial_map_callback, 100
        )

        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet, self._det_topic, self.detection_callback, 100
        )

        self._headset_pose_subscription = self.create_subscription(
            HeadsetPoseData, self._headset_pose_topic, self.headset_pose_callback, 100
        )

        self._object_3d_publisher = self.create_publisher(
            ObjectDetection3dSet, self._det_3d_topic, 1
        )

        self.prev_time = -1
        self.meshes = {}

        self.scene = trimesh.Scene()

        self.poses = []

        # setup the image size query client and make sure the service is running
        self.image_size_client = self.create_client(QueryImageSize, "query_image_size")
        while not self.image_size_client.wait_for_service(timeout_sec=1.0):
            log.info("Waiting for image size service...")

        # send image size queries unti we get a valid response
        r = self.send_image_size_request()
        log.info(f"Image size response {r}")
        while r.image_width <= 0 or r.image_height <= 0:
            log.warn(
                "Invalid image dimensions."
                + " Make sure the image converter node is running and receiving frames"
            )

            time.sleep(1)
            r = self.send_image_size_request()

        log.info(
            f"Received valid image dimensions. Current size: {r.image_width}x{r.image_height}"
        )
        self.image_height = r.image_height
        self.image_width = r.image_width

    def send_image_size_request(self):
        """
        Sends a `QueryImageSize` request message to the QueryImageSize
        service running in the object detector node.
        """
        req = QueryImageSize.Request()
        future = self.image_size_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def spatial_map_callback(self, msg):
        log = self.get_logger()

        # extract the vertices into np arrays
        vertices = np.array([[v.x, v.y, v.z] for v in msg.mesh.vertices])

        # extract the triangles into np arrays
        triangles = np.array(
            [
                [t.vertex_indices[0], t.vertex_indices[1], t.vertex_indices[2]]
                for t in msg.mesh.triangles
            ]
        )

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        self.meshes[msg.mesh_id] = mesh
        self.scene.add_geometry(mesh)

        # self.scene.show()

    def headset_pose_callback(self, pose):
        log = self.get_logger()
        log.debug(f"pose stamp: {pose.header.stamp}")
        self.poses.append(pose)

    def detection_callback(self, detection):
        log = self.get_logger()

        if detection.num_detections == 0:
            log.debug("No detections for this image")
            return

        # locate the headset poses for this detection using
        # the detection source timestamp
        world_matrix_1d = None
        projection_matrix_1d = None

        # print(detection.source_stamp)

        # TODO: maybe implement a more efficient binary search here
        for i in reversed(range(len(self.poses))):
            if detection.source_stamp == self.poses[i].header.stamp:
                world_matrix_1d = self.poses[i].world_matrix
                projection_matrix_1d = self.poses[i].projection_matrix

                log.debug(
                    f"time stamps: {self.poses[i].header.stamp} {detection.source_stamp}"
                )

                # can clear out our pose list now assuming we won't get
                # image frame detections out of order
                self.poses = self.poses[i:]
                break

        if world_matrix_1d == None or projection_matrix_1d == None:
            log.info(f"Did not get world or projection matrix.")
            return

        # get world matrix from detection
        world_matrix_2d = self.convert_1d_4x4_to_2d_matrix(world_matrix_1d)
        log.info(f"world matrix {world_matrix_2d}")

        # get position of the camera at the time of the frame
        camera_origin = self.get_world_position(
            world_matrix_2d, np.array([0.0, 0.0, 0.0])
        ).reshape((1, 3))
        log.info(f"origin {camera_origin}")

        # compute the x, y, and z axes for this image
        x_dir = self.get_world_position(
            world_matrix_2d, np.array([1.0, 0.0, 0.0])
        ).flatten()

        y_dir = self.get_world_position(
            world_matrix_2d, np.array([0.0, 1.0, 0.0])
        ).flatten()

        z_dir = self.get_world_position(
            world_matrix_2d, np.array([0.0, 0.0, 1.0])
        ).flatten()

        # plot the x, y, and z axes on the scene
        vs = np.array([camera_origin[0], x_dir])
        el = trimesh.path.entities.Line([0, 1])
        path = trimesh.path.Path3D(
            entities=[el], vertices=vs, colors=np.array([255, 0, 0, 255]).reshape(1, 4)
        )
        self.scene.add_geometry(path)

        vs = np.array([camera_origin[0], y_dir])
        el = trimesh.path.entities.Line([0, 1])
        path = trimesh.path.Path3D(
            entities=[el], vertices=vs, colors=np.array([0, 255, 0, 255]).reshape(1, 4)
        )
        self.scene.add_geometry(path)

        vs = np.array([camera_origin[0], z_dir])
        el = trimesh.path.entities.Line([0, 1])
        path = trimesh.path.Path3D(
            entities=[el], vertices=vs, colors=np.array([0, 0, 255, 255]).reshape(1, 4)
        )
        self.scene.add_geometry(path)
        # self.scene.show()

        # get projection matrix from detection
        projection_matrix_2d = self.convert_1d_4x4_to_2d_matrix(projection_matrix_1d)

        # start creating the 3D detection set message
        det_3d_set_msg = ObjectDetection3dSet()
        det_3d_set_msg.header.stamp = self.get_clock().now().to_msg()
        det_3d_set_msg.header.frame_id = detection.header.frame_id
        det_3d_set_msg.source_stamp = detection.source_stamp

        det_conf_mat = to_confidence_matrix(detection)
        for i in range(detection.num_detections):
            object_type = sorted(zip(det_conf_mat[i], detection.label_vec))[-1][1]

            # get pixel positions of detected object box and form into box corners
            min_vertex0 = detection.left[i]
            min_vertex1 = detection.top[i]
            max_vertex0 = detection.right[i]
            max_vertex1 = detection.bottom[i]
            image_min_vertex0 = 0
            image_min_vertex1 = 0
            image_max_vertex0 = self.image_width
            image_max_vertex1 = self.image_height

            corners_screen_pos = [
                [min_vertex0, min_vertex1],
                [min_vertex0, max_vertex1],
                [max_vertex0, max_vertex1],
                [max_vertex0, min_vertex1],
            ]

            # convert detection screen pixel coordinates to world coordinates
            corners_world_pos = []
            for p in corners_screen_pos:
                point_3d = self.convert_pixel_coord_to_world_coord(
                    world_matrix_2d, projection_matrix_2d, p, camera_origin
                )
                log.info(f"point 3d: {point_3d}")
                if point_3d is None:
                    log.info(f"No point found!")
                    # self.scene.show()
                    return
                corners_world_pos.append(point_3d)

            log.info(f"Drawing box for {object_type}")
            vs = np.array(
                [
                    corners_world_pos[0],
                    corners_world_pos[1],
                    corners_world_pos[2],
                    corners_world_pos[3],
                ]
            )
            el = trimesh.path.entities.Line([0, 1, 2, 3, 0])
            path = trimesh.path.Path3D(entities=[el], vertices=vs)
            self.scene.add_geometry(path)

            # draw the image edges
            bounds_screen_pos = [
                [image_min_vertex0, image_min_vertex1],
                [image_min_vertex0, image_max_vertex1],
                [image_max_vertex0, image_max_vertex1],
                [image_max_vertex0, image_min_vertex1],
            ]
            image_corners_world_pos = []
            for p in bounds_screen_pos:
                scaled_point = self.scale_pixel_coordinates(p)

                # see note with these constants for why we are not using the
                # focal length values from the projection matrix
                focal_length_x = FOCAL_LENGTH_X
                focal_length_y = FOCAL_LENGTH_Y

                center_x = projection_matrix_2d[0][2]
                center_y = projection_matrix_2d[1][2]

                norm_factor = projection_matrix_2d[2][2]
                center_x = center_x / norm_factor  # 0
                center_y = center_y / norm_factor  # 0

                # convert to camera space
                # NOTE: The negative sign in the z-direction is to convert
                # between the left-handed Unity coordinates and the right-handed
                # trimesh scene coordinates.
                dir_ray = np.array(
                    [
                        (scaled_point[0] - center_x) / focal_length_x,
                        (scaled_point[1] - center_y) / focal_length_y,
                        -1.0 / norm_factor,
                    ]
                ).reshape((1, 3))

                # project camera space onto world position
                direction = self.get_world_position(
                    world_matrix_2d, dir_ray[0]
                ).reshape((1, 3))

                image_corners_world_pos.append(direction[0])

            vs = np.array(
                [
                    image_corners_world_pos[0],
                    image_corners_world_pos[1],
                    image_corners_world_pos[2],
                    image_corners_world_pos[3],
                ]
            )
            el = trimesh.path.entities.Line([0, 1, 2, 3, 0])
            path = trimesh.path.Path3D(
                entities=[el],
                vertices=vs,
                colors=np.array([255, 0, 255, 255]).reshape(1, 4),
            )
            self.scene.add_geometry(path)

            # since we were able to find this object's 3D position,
            # add it to 3d detection message
            det_3d_set_msg.object_labels.append(object_type)
            det_3d_set_msg.num_objects += 1

            for p in range(4):
                point_3d = Point()

                log.info(f"scene position {corners_world_pos[p]}")
                point_3d.x = corners_world_pos[p][0]
                point_3d.y = corners_world_pos[p][1]
                point_3d.z = corners_world_pos[p][2]

                if p == 0:
                    det_3d_set_msg.left.append(point_3d)
                elif p == 1:
                    det_3d_set_msg.top.append(point_3d)
                elif p == 2:
                    det_3d_set_msg.right.append(point_3d)
                elif p == 3:
                    det_3d_set_msg.bottom.append(point_3d)

        # form and publish the 3d object detection message
        self._object_3d_publisher.publish(det_3d_set_msg)

        # uncomment this to visualize the scene
        # self.scene.show()

    def show_plot(self):
        try:
            self.scene.show()
        except:
            pass

    def cast_ray(self, origin, direction):
        intersection_points = []
        for key, m in self.meshes.items():
            ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(m)
            try:
                intersection = ray_intersector.intersects_location(origin, direction)
            except:
                continue

            if (len(intersection[0])) != 0:
                for i in intersection[0]:
                    intersection_points.append(i)

        return intersection_points

    def get_world_position(self, world_matrix, point):
        point_matrix = np.array([[point[0]], [point[1]], [point[2]], [1]])
        return np.matmul(world_matrix, point_matrix)[:3]

    def convert_1d_4x4_to_2d_matrix(self, matrix_1d):
        matrix_2d = [[], [], [], []]
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                matrix_2d[row].append(matrix_1d[idx])

        return matrix_2d

    def convert_pixel_coord_to_world_coord(
        self, world_matrix_2d, projection_matrix, p, camera_origin
    ):
        """
        Adapted from https://github.com/VulcanTechnologies/HoloLensCameraStream
        """
        log = self.get_logger()

        scaled_point = self.scale_pixel_coordinates(p)

        # see note with these constants for why we are not using the
        # focal length values from the projection matrix
        focal_length_x = FOCAL_LENGTH_X
        focal_length_y = FOCAL_LENGTH_Y

        center_x = projection_matrix[0][2]  # 0
        center_y = projection_matrix[1][2]  # 0

        norm_factor = projection_matrix[2][2]  # -1.004
        center_x = center_x / norm_factor
        center_y = center_y / norm_factor

        # convert coords to camera space
        # NOTE: The negative sign in the z-direction is to convert
        # between the left-handed Unity coordinates and the right-handed
        # trimesh scene coordinates.
        dir_ray = np.array(
            [
                (scaled_point[0] - center_x) / focal_length_x,
                (scaled_point[1] - center_y) / focal_length_y,
                -1.0 / norm_factor,
            ]
        ).reshape((1, 3))
        log.debug(f"dir_ray {dir_ray}")

        # project camera space onto world position
        direction = self.get_world_position(world_matrix_2d, dir_ray[0]).reshape((1, 3))

        vs = np.array([camera_origin[0], direction[0]])
        el = trimesh.path.entities.Line([0, 1])
        path = trimesh.path.Path3D(
            entities=[el],
            vertices=vs,
            colors=np.array([255, 255, 0, 255]).reshape(1, 4),
        )
        self.scene.add_geometry(path)

        # self.scene.show()
        log.info(f"direction: {direction}")
        log.debug(f"origin : {camera_origin}")

        intersecting_points = self.cast_ray(camera_origin, direction - camera_origin)
        if intersecting_points is None:
            log.info("No intersecting meshes found!")
            return None

        closest_point = None
        try:
            log.debug(f"Points found {intersecting_points}")

            # if there is more than one point found, use the closest one to the camera
            min_distance = -1
            for point in intersecting_points:
                # calculate distance between this point and the camera
                distance = (
                    (camera_origin[0][0] - point[0]) ** 2
                    + (camera_origin[0][1] - point[1]) ** 2
                    + (camera_origin[0][2] - point[2]) ** 2
                ) ** 0.5
                log.debug(f"Point {p}: distance = {distance}")
                if min_distance == -1:
                    min_distance = distance
                    closest_point = point
                elif distance < min_distance:
                    min_distance = distance
                    closest_point = point

            # closest_point[2] = -closest_point[2]
            log.debug(f"Closest point = {closest_point}")
        except Exception as e:
            log.info(str(e))

        return closest_point

    def scale_pixel_coordinates(self, p):
        """
        Adapted from https://github.com/VulcanTechnologies/HoloLensCameraStream
        """
        log = self.get_logger()

        scaled_point = p

        half_width = self.image_width / 2.0
        half_height = self.image_height / 2.0

        # translate registration to image center
        scaled_point[0] -= half_width
        scaled_point[1] -= half_height

        # scale pixel coords to percentage coords (-1 to 1)
        scaled_point[0] = scaled_point[0] / half_width
        scaled_point[1] = scaled_point[1] / half_height * -1.0

        return scaled_point


main = make_default_main(SpatialMapSubscriber)


if __name__ == "__main__":
    main()
