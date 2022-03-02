import queue
import socket
import struct
import sys
import time
import threading
import pickle

import numpy as np
import rclpy
from rclpy.node import Node
from angel_msgs.msg import ObjectDetection2dSet, SpatialMesh, HeadsetPoseData

import trimesh
import trimesh.viewer


class SpatialMapSubscriber(Node):

    def __init__(self):
        super().__init__(self.__class__.__name__)

        self.declare_parameter("spatial_map_topic", "SpatialMapData")
        self.declare_parameter("det_topic", "ObjectDetections")
        self.declare_parameter("pose_topic", "HeadsetPoseData")

        self._spatial_map_topic = self.get_parameter("spatial_map_topic").get_parameter_value().string_value
        self._det_topic = self.get_parameter("det_topic").get_parameter_value().string_value
        self._pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value

        log = self.get_logger()
        log.info(f"Spatial map topic: {self._spatial_map_topic}")
        log.info(f"Detection topic: {self._det_topic}")
        log.info(f"Pose topic: {self._pose_topic}")

        self.subscription = self.create_subscription(
            SpatialMesh,
            self._spatial_map_topic,
            self.spatial_map_callback,
            100)

        self._detection_subscription = self.create_subscription(
            ObjectDetection2dSet,
            self._det_topic,
            self.detection_callback,
            100
        )

        self._pose_subscription = self.create_subscription(
            HeadsetPoseData,
            self._pose_topic,
            self.headset_pose_callback,
            100
        )

        self.frames_recvd = 0
        self.prev_time = -1
        self.meshes = {}

        self.scene = trimesh.Scene()

        self.poses = []


    def spatial_map_callback(self, msg):
        log = self.get_logger()
        self.frames_recvd += 1

        #print(len(msg.data), msg.data[0:12])

        mesh_id = msg.mesh_id
        num_vertices = len(msg.mesh.vertices)
        num_triangles = len(msg.mesh.triangles)
        #print(mesh_id, num_vertices, num_triangles)
        #print(msg.mesh.vertices)
        #print(msg.mesh.triangles)

        # extract the vertices into np arrays
        vertices = np.array([])

        for v in msg.mesh.vertices:
            if vertices.size == 0:
                vertices = np.array([v.x, v.y, v.z])
            else:
                vertices = np.vstack([vertices, np.array([v.x, v.y, v.z])])

        # extract the triangles into np arrays
        triangles = np.array([])
        for t in msg.mesh.triangles:
            if triangles.size == 0:
                triangles = np.array([t.vertex_indices[0],
                                      t.vertex_indices[1],
                                      t.vertex_indices[2]
                                     ])
            else:
                triangles = np.vstack([triangles,
                                       np.array([t.vertex_indices[0],
                                                 t.vertex_indices[1],
                                                 t.vertex_indices[2]
                                                ])])

        if msg.removal:
            log.debug("Got a removal!")
            # TODO: remove from the spatial map

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        self.meshes[msg.mesh_id] = mesh
        self.scene.add_geometry(mesh)


    def headset_pose_callback(self, pose):
        log = self.get_logger()

        #log.info(f"World matrix: {pose.world_matrix}")
        #log.info(f"Projection matrix: {pose.projection_matrix}")
        #log.info(f"pose stamp: {pose.header.stamp}")

        self.poses.append(pose)


    def detection_callback(self, detection):
        log = self.get_logger()
        image_width = 1280.0
        image_height = 720.0

        #log.info(f"Num detections: {detection.num_detections}")
        log.info(f"image stamp: {detection.source_stamp}")
        #log.info(f"Frame id: {detection.header.frame_id}")
        #log.info(f"labels: {len(detection.label_vec)}")

        #for i in range(len(detection.label_vec)):
        #    log.info(f"object: {detection.label_vec[i]}, {detection.label_confidences[i]}")

        world_matrix_1d = None
        projection_matrix_1d = None
        
        for i in range(len(self.poses)):
            if detection.source_stamp == self.poses[i].header.stamp:
                log.info("Found our pose!!")
                world_matrix_1d = self.poses[i].world_matrix
                projection_matrix_1d = self.poses[i].projection_matrix

                # can clear out our pose list now assuming we won't get
                # image frame detections out of order
                self.poses = self.poses[i:]
                break

        if world_matrix_1d == None or projection_matrix_1d == None:
            log.info("Did not get world or projection matrix")
            return

        # get world matrix from detection
        world_matrix_2d = [[], [], [], []]
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                world_matrix_2d[row].append(world_matrix_1d[idx])

        # negate z component of world matrix
        world_matrix_2d[0][2] = -world_matrix_2d[0][2]
        world_matrix_2d[1][2] = -world_matrix_2d[1][2]
        world_matrix_2d[2][2] = -world_matrix_2d[2][2]
        print(world_matrix_2d)

        # get projection matrix from detection
        projection_matrix_2d = [[], [], [], []]
        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                projection_matrix_2d[row].append(projection_matrix_1d[idx])

        # get pixel positions of detected object box and form into box corners
        object_type = detection.label_vec
        min_vertex0 = detection.left
        min_vertex1 = detection.top
        max_vertex0 = detection.right
        max_vertex1 = detection.bottom

        #print(object_type, min_vertex0, min_vertex1, max_vertex0, max_vertex1)
        corners_screen_pos = [[min_vertex0, min_vertex1], [min_vertex0, max_vertex1],
                              [max_vertex0, max_vertex1], [max_vertex0, min_vertex1]]

        # get the inverse of the projection matrix
        projection_inv = np.linalg.inv(projection_matrix_2d)
        #print(projection_inv)

        # get position of the camera at the time of the frame
        camera_origin = self.get_world_position(world_matrix_2d,
                                                np.array([0.0, 0.0, 0.0]))
        camera_origin = camera_origin.reshape((1, 3))
        log.debug(f"origin: {camera_origin} {camera_origin.shape}")

        # convert detection screen pixel coordinates to world pos
        corners_world_pos = []
        for p in corners_screen_pos:
            # scale by image width and height and convert to -1:1 coordinates
            image_pos_zero_to_one = np.array([p[0] / image_width, 1 - (p[1] / image_height)])
            image_pos_zero_to_one = (image_pos_zero_to_one * 2) - np.array([1, 1])
            #print("object position in -1 : 1", image_pos_zero_to_one)

            # convert screen point to camera point
            image_pos_projected = self.get_world_position(projection_inv,
                                                          np.array([image_pos_zero_to_one[0],
                                                                    image_pos_zero_to_one[1],
                                                                    1]))
            #print("object position in camera space 1", image_pos_projected)

            # convert camera position to world position
            world_space_box_pos = self.get_world_position(world_matrix_2d,
                                                          np.array([image_pos_projected[0][0],
                                                                    image_pos_projected[1][0],
                                                                    1]))
            world_space_box_pos = world_space_box_pos.reshape((1, 3))
            #print("object position in world space ", world_space_box_pos, world_space_box_pos.shape)

            '''
            # draw debug vector
            vs = np.array([camera_origin[0], world_space_box_pos[0]])
            el = trimesh.path.entities.Line([0, 1])
            path = trimesh.path.Path3D(entities=[el], vertices=vs, colors=np.array([255, 0, 0, 255]).reshape(1, 4))
            self.scene.add_geometry(path)
            '''

            # cast ray from camera origin to the object
            intersecting_points = self.cast_ray(camera_origin, world_space_box_pos - camera_origin)

            if intersecting_points is None:
                log.info("No intersecting meshes found!")
                return

            try:
                log.debug(f"Points found {intersecting_points}, {object_type}")

                # if there is more than one point found, use the closest one to the camera
                min_distance = -1
                closest_point = None
                for p in intersecting_points:
                    # calculate distance between this point and the camera
                    distance = ((camera_origin[0][0] - p[0]) ** 2 +
                                (camera_origin[0][1] - p[1]) ** 2 +
                                (camera_origin[0][2] - p[2]) ** 2) ** 0.5
                    log.debug(f"Point {p}: distance = {distance}")
                    if min_distance == -1:
                        min_distance = distance
                        closest_point = p
                    elif distance < min_distance:
                        min_distance = distance
                        closest_point = p

                log.debug(f"Closest point = {closest_point}")
                corners_world_pos.append(closest_point)
            except Exception as e:
                print(e)
                pass

        vs = np.array([corners_world_pos[0], corners_world_pos[1],
                       corners_world_pos[2], corners_world_pos[3]])
        el = trimesh.path.entities.Line([0, 1, 2, 3, 0])
        path = trimesh.path.Path3D(entities=[el], vertices=vs)
        self.scene.add_geometry(path)
        self.show_plot()

    def show_plot(self):
        try:
            self.scene.show()
        except:
            pass

    def cast_ray(self, origin, direction):
        intersection_point = None
        for key, m in self.meshes.items():
            ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(m)
            intersection = ray_intersector.intersects_location(origin, direction)

            if (len(intersection[0])) != 0:
                intersection_point = intersection[0]
                #print("point", intersection_point)

        return intersection_point


    def get_world_position(self, world_matrix, point):
        point_matrix = np.array([[point[0]], [point[1]], [point[2]], [1]])
        return np.matmul(world_matrix, point_matrix)[:3]


def main():
    rclpy.init()

    spatial_map_subscriber = SpatialMapSubscriber()
    rclpy.spin(spatial_map_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    spatial_map_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
