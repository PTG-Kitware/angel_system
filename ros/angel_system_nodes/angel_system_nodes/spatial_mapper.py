import math
import queue
from threading import (
    Event,
    Lock,
    Thread,
)
import time

import numpy as np
import rclpy
from rclpy.node import Node
from angel_msgs.msg import (
    ObjectDetection2dSet,
    ObjectDetection3dSet,
    SpatialMeshes,
    HeadsetPoseData
)
from angel_msgs.srv import QueryImageSize
from angel_utils import declare_and_get_parameters
from angel_utils.conversion import to_confidence_matrix
from geometry_msgs.msg import Point

import open3d as o3d


# NOTE: These values were extracted from the projection matrix provided by the
# Unity main camera with the Windows MR plugin (now deprecated).
# For some unknown reason, the values provided by the Open XR plugin are
# different and do not provide the correct results. In the future, we should
# figure out why they are different or extract the focal length values from the
# MediaFrameReader which provides the frames, instead of the Unity main camera.
FOCAL_LENGTH_X = 1.6304
FOCAL_LENGTH_Y = 2.5084

PARAM_SM_TOPIC = "sm_topic"
PARAM_DET_TOPIC = "det_topic"
PARAM_HEAD_POSE_TOPIC = "headset_pose_topic"
PARAM_DET_3D_TOPIC = "det_3d_topic"


class SpatialMapSubscriber(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()

        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_SM_TOPIC,),
                (PARAM_DET_TOPIC,),
                (PARAM_HEAD_POSE_TOPIC,),
                (PARAM_DET_3D_TOPIC,),
            ]
        )

        self._spatial_map_topic = param_values[PARAM_SM_TOPIC]
        self._det_topic = param_values[PARAM_DET_TOPIC]
        self._headset_pose_topic = param_values[PARAM_HEAD_POSE_TOPIC]
        self._det_3d_topic = param_values[PARAM_DET_3D_TOPIC]

        # Spatial meshes are added to this scene
        self.o3d_scene = o3d.t.geometry.RaycastingScene()

        self.poses = []
        self.dets = queue.Queue()
        self.mesh_l = Lock()

        # Setup the image size query client and make sure the service is running
        self.image_size_client = self.create_client(QueryImageSize, 'query_image_size')
        while not self.image_size_client.wait_for_service(timeout_sec=1.0):
            log.info("Waiting for image size service...")

        # Send image size queries unti we get a valid response
        r = self.send_image_size_request()
        log.info(f"Image size response {r}")
        while (r.image_width <= 0 or r.image_height <= 0):
            log.warn("Invalid image dimensions."
                     + " Make sure the image converter node is running and receiving frames")

            time.sleep(1)
            r = self.send_image_size_request()
        log.info(
            f"Received valid image dimensions. Current size: {r.image_width}x{r.image_height}"
        )
        self.image_height = r.image_height
        self.image_width = r.image_width

        # Start the runtime thread
        log.info("Starting mapper thread...")
        # switch for runtime loop
        self._rt_active = Event()
        self._rt_active.set()
        # seconds to occasionally time out of the wait condition for the loop
        # to check if it is supposed to still be alive.
        self._rt_active_heartbeat = 0.1
        # Event to notify runtime it should try processing now.
        self._rt_awake_evt = Event()
        self._rt_thread = Thread(
            target=self.thread_mapper_runtime,
            name="mapper_runtime"
        )
        self._rt_thread.daemon = True
        self._rt_thread.start()
        log.info("Starting mapper thread... Done")

        # Create ROS pubs/subs
        self._spatial_mesh_subscription = self.create_subscription(
            SpatialMeshes,
            self._spatial_map_topic,
            self.spatial_map_callback,
            1,
        )
        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.detection_callback,
            1,
        )
        self._headset_pose_subscription = self.create_subscription(
            HeadsetPoseData,
            self._headset_pose_topic,
            self.headset_pose_callback,
            1,
        )
        self._object_3d_publisher = self.create_publisher(
            ObjectDetection3dSet,
            self._det_3d_topic,
            1,
        )

    def thread_mapper_runtime(self):
        """
        Function that projects 2D object detections to their 3D positions.
        """
        log = self.get_logger()
        log.info("Mapper loop starting")

        while self._rt_active.wait(0):  # will quickly return false if cleared.
            if self._rt_awake_evt.wait(self._rt_active_heartbeat):
                log.info("Mapper loop awakened")
                # reset the flag for the next go-around
                self._rt_awake_evt.clear()

                detection = self.dets.get()

                # locate the headset poses for this detection using
                # the detection source timestamp
                world_matrix_1d = None
                projection_matrix_1d = None

                # TODO: maybe implement a more efficient binary search here
                for i in reversed(range(len(self.poses))):
                    if detection.source_stamp == self.poses[i].header.stamp:
                        world_matrix_1d = np.array(self.poses[i].world_matrix)
                        projection_matrix_1d = np.array(self.poses[i].projection_matrix)

                        log.debug(
                            f"time stamps: {self.poses[i].header.stamp} {detection.source_stamp}"
                        )

                        # can clear out our pose list now assuming we won't get
                        # image frame detections out of order
                        self.poses = self.poses[i:]
                        break

                if world_matrix_1d is None or projection_matrix_1d is None:
                    log.info("Did not get world or projection matrix.")
                    continue
                should_break = False

                # get world matrix from detection
                world_matrix_2d = world_matrix_1d.reshape(4, 4)
                log.debug(f"world matrix {world_matrix_2d}")

                # get position of the camera at the time of the frame
                camera_origin = self.get_world_position(world_matrix_2d,
                                                        np.array([0.0, 0.0, 0.0])).reshape((1, 3))
                log.debug(f"origin {camera_origin}")

                # get projection matrix from detection
                projection_matrix_2d = projection_matrix_1d.reshape(4, 4)

                # start creating the 3D detection set message
                det_3d_set_msg = ObjectDetection3dSet()
                det_3d_set_msg.header.stamp = self.get_clock().now().to_msg()
                det_3d_set_msg.header.frame_id = detection.header.frame_id
                det_3d_set_msg.source_stamp = detection.source_stamp

                log.debug("Processing detections matrix")
                det_conf_mat = to_confidence_matrix(detection)
                for i in range(detection.num_detections):
                    object_type = sorted(zip(det_conf_mat[i], detection.label_vec))[-1][1]

                    # get pixel positions of detected object box and form into box corners
                    min_vertex0 = detection.left[i]
                    min_vertex1 = detection.top[i]
                    max_vertex0 = detection.right[i]
                    max_vertex1 = detection.bottom[i]

                    corners_screen_pos = [[min_vertex0, min_vertex1], [min_vertex0, max_vertex1],
                                          [max_vertex0, max_vertex1], [max_vertex0, min_vertex1]]

                    # convert detection screen pixel coordinates to world coordinates
                    corners_world_pos = []
                    all_points_found = True
                    for p in corners_screen_pos:
                        point_3d = self.convert_pixel_coord_to_world_coord(world_matrix_2d,
                                                                           projection_matrix_2d,
                                                                           p, camera_origin)
                        log.debug(f"point 3d: {point_3d}")
                        if point_3d is None:
                            log.debug("No point found!")
                            all_points_found = False
                            break

                        corners_world_pos.append(point_3d)

                    if not all_points_found:
                        # Skip this detection, not all 3D points found
                        continue

                    # Since we were able to find this object's 3D position,
                    # add it to 3d detection message
                    det_3d_set_msg.object_labels.append(object_type)
                    det_3d_set_msg.num_objects += 1

                    for p in range(4):
                        point_3d = Point()

                        log.debug(f"scene position {corners_world_pos[p]}")
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

                        if object_type == "hand":
                            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.02)
                            mesh_sphere.compute_vertex_normals()
                            mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

                            mesh_sphere = mesh_sphere.translate(corners_world_pos[p])

                            self.meshes.append(mesh_sphere)
                            should_break = True

                    if object_type == "hand":
                        xs = sorted([
                            det_3d_set_msg.right[-1].x,
                            det_3d_set_msg.left[-1].x,
                            det_3d_set_msg.top[-1].x,
                            det_3d_set_msg.bottom[-1].x
                        ])
                        ys = sorted([
                            det_3d_set_msg.right[-1].y,
                            det_3d_set_msg.left[-1].y,
                            det_3d_set_msg.top[-1].y,
                            det_3d_set_msg.bottom[-1].y,
                        ])
                        zs = sorted([
                            det_3d_set_msg.right[-1].z,
                            det_3d_set_msg.left[-1].z,
                            det_3d_set_msg.top[-1].z,
                            det_3d_set_msg.bottom[-1].z
                        ])

                        width = xs[-1] - xs[0]  # width
                        height = ys[-1] - ys[0]  # height
                        depth = zs[-1] - zs[0]  # depth

                        center_x = xs[0] + (0.5 * width)
                        center_y = ys[0] + (0.5 * height)
                        center_z = zs[0] + (0.5 * depth)
                        print("CENTER", center_x, center_y, center_z)

                        # Draw the center sphere
                        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=.02)
                        mesh_sphere.compute_vertex_normals()
                        mesh_sphere.paint_uniform_color([0.7, 0.1, 0.7])
                        mesh_sphere = mesh_sphere.translate([center_x, center_y, center_z])
                        self.meshes.append(mesh_sphere)

                # Form and publish the 3d object detection message
                self._object_3d_publisher.publish(det_3d_set_msg)
                log.info(f"Published 3d detections with {det_3d_set_msg.num_objects} dets")

                if should_break:
                    o3d.visualization.draw_geometries(self.meshes, mesh_show_back_face=False)
                    break

        log.info("Runtime function end.")

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
        """
        Callback function for the spatial meshes subscriber. Extracts the
        meshes and adds them to the Open3d scene.
        """
        log = self.get_logger()

        self.meshes = []
        with self.mesh_l:
            # Clear the old meshes
            self.o3d_scene = o3d.t.geometry.RaycastingScene()

            for m in msg.meshes:
                # extract the vertices into np arrays
                vertices = np.array([[v.x, v.y, v.z] for v in m.mesh.vertices])

                # extract the triangles into np arrays
                triangles = np.array([[t.vertex_indices[0],
                                       t.vertex_indices[1],
                                       t.vertex_indices[2]] for t in m.mesh.triangles])

                open3d_mesh = o3d.geometry.TriangleMesh()
                open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                open3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

                self.meshes.append(open3d_mesh)

                self.o3d_scene.add_triangles(
                    o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
                )

        log.info("Received updated meshes")

    def headset_pose_callback(self, pose):
        """
        Callback function for headset pose messages. Appends the pose msg to
        list of stored poses.
        """
        self.poses.append(pose)

    def detection_callback(self, detection):
        """
        Callback function for detection messages. Places the detection into the
        detection queue and returns.
        """
        log = self.get_logger()
        if detection.num_detections == 0:
            log.debug("No detections for this image")
            return

        self.dets.put(detection)
        log.info("Queued detection")

        # Awaken the mapper thread
        self._rt_awake_evt.set()

    def cast_ray(self, origin, direction):
        """
        Casts a ray from the given `origin` point in the direction of `direction`.

        Returns the closest intersecting point, if there is an intersection
        with any of the meshes in the scene, otherwise returns None.
        """
        with self.mesh_l:
            origin_np = np.array(origin).squeeze()
            dir_np = np.array(direction).squeeze()

            # Define the ray
            rays = o3d.core.Tensor([[origin[0][0], origin[0][1], origin[0][2],
                                    direction[0][0], direction[0][1], direction[0][2]]],
                                   dtype=o3d.core.Dtype.Float32)

            # Cast ray and get the intersection distance
            ans = self.o3d_scene.cast_rays(rays)
            distance = ans["t_hit"][0].numpy()

            if distance != math.inf:
                # Compute intersection point from distance and direction ray
                intersecting_point = origin_np + distance * dir_np
            else:
                # Infinite distance returned if no intersection
                intersecting_point = None

            return intersecting_point

    def get_world_position(self, world_matrix, point):
        """
        Converts the position of a point in camera space to its position in
        world space using the camera-to-world matrix.
        """
        point_matrix = np.array([point[0], point[1], point[2], 1])
        return np.matmul(point_matrix, world_matrix)[:3]

    def convert_pixel_coord_to_world_coord(self, world_matrix_2d,
                                           projection_matrix, p, camera_origin):
        """
        Converts a point, p, in pixel coordinates to its 3d world position using
        the camera to world matrix, camera project matrix, and camera origin.

        If no world point is found, None is returned.

        Adapted from https://github.com/VulcanTechnologies/HoloLensCameraStream
        """
        log = self.get_logger()
        scaled_point = self.scale_pixel_coordinates(p)

        # see note with these constants for why we are not using the
        # focal length values from the projection matrix
        focal_length_x = FOCAL_LENGTH_X
        focal_length_y = FOCAL_LENGTH_Y

        center_x = projection_matrix[0][2] # 0
        center_y = projection_matrix[1][2] # 0

        norm_factor = projection_matrix[2][2] # -1.004
        center_x = center_x / norm_factor
        center_y = center_y / norm_factor

        # convert coords to camera space
        # NOTE: The negative sign in the z-direction is to convert
        # between the left-handed Unity coordinates and the right-handed
        # scene coordinates.
        dir_ray = np.array([(scaled_point[0] - center_x) / focal_length_x,
                            (scaled_point[1] - center_y) / focal_length_y,
                            -1.0 / norm_factor]).reshape((1, 3))
        log.debug(f"dir_ray {dir_ray}")

        # project camera space onto world position
        direction = self.get_world_position(world_matrix_2d, dir_ray[0]).reshape((1, 3))

        log.debug(f"direction: {direction}")
        log.debug(f"origin : {camera_origin}")

        return self.cast_ray(camera_origin, direction - camera_origin)

    def scale_pixel_coordinates(self, p):
        """
        Converts a set of coordinates in pixel space [image width pixels, image
        height pixels] to the range [-1, 1].

        Adapted from https://github.com/VulcanTechnologies/HoloLensCameraStream
        """
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

    def stop_runtime(self) -> None:
        """
        Indicate that the runtime loop should cease.
        """
        self._rt_active.clear()


def main():
    rclpy.init()

    spatial_mapper = SpatialMapSubscriber()

    try:
        rclpy.spin(spatial_mapper)
    except KeyboardInterrupt:
        spatial_mapper.stop_runtime()
        spatial_mapper.get_logger().info("Keyboard interrupt, shutting down.\n")

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    spatial_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
