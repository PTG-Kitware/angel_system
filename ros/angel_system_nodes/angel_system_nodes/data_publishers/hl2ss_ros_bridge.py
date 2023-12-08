import time
from threading import Event, Thread

from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from geometry_msgs.msg import (
    Point,
    Pose,
    Quaternion,
)
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from shape_msgs.msg import (
    Mesh,
    MeshTriangle,
)

from angel_msgs.msg import (
    HandJointPose,
    HandJointPosesUpdate,
    HeadsetAudioData,
    HeadsetPoseData,
    SpatialMesh,
)
from angel_utils import declare_and_get_parameters, RateTracker
from angel_utils import make_default_main
from angel_utils.hand import JOINT_LIST
from hl2ss.viewer import hl2ss


BRIDGE = CvBridge()

# Encoded stream average bits per second
# Must be > 0
# Value copied from hl2ss/viewer/cient_pv.py example
PV_BITRATE = 5 * 1024 * 1024

PARAM_PV_IMAGES_TOPIC = "image_topic"  # for publishing image data.
PARAM_PV_IMAGES_TS_TOPIC = "image_ts_topic"  # for image timestamp publishing only.
PARAM_HAND_POSE_TOPIC = "hand_pose_topic"
PARAM_AUDIO_TOPIC = "audio_topic"
PARAM_SM_TOPIC = "sm_topic"
PARAM_HEAD_POSE_TOPIC = "head_pose_topic"
PARAM_IP_ADDR = "ip_addr"
PARAM_PV_WIDTH = "pv_width"
PARAM_PV_HEIGHT = "pv_height"
PARAM_PV_FRAMERATE = "pv_framerate"
PARAM_SM_FREQ = "sm_freq"
PARAM_RM_DEPTH_AHAT_TOPIC = "rm_depth_AHAT"

# Pass string for any of the ROS topic params to disable that stream
DISABLE_TOPIC_STR = "disable"


class HL2SSROSBridge(Node):
    """
    ROS node that uses HL2SS client/server library to convert HL2SS data to
    ROS messages used throughout the ANGEL system.
    """

    def __init__(self):
        super().__init__(self.__class__.__name__)
        log = self.get_logger()
        param_values = declare_and_get_parameters(
            self,
            [
                (PARAM_PV_IMAGES_TOPIC,),
                (PARAM_PV_IMAGES_TS_TOPIC,),
                (PARAM_HAND_POSE_TOPIC,),
                (PARAM_AUDIO_TOPIC,),
                (PARAM_SM_TOPIC,),
                (PARAM_HEAD_POSE_TOPIC,),
                (PARAM_IP_ADDR,),
                (PARAM_PV_WIDTH,),
                (PARAM_PV_HEIGHT,),
                (PARAM_PV_FRAMERATE,),
                (PARAM_SM_FREQ,),
                (PARAM_RM_DEPTH_AHAT_TOPIC,),
            ],
        )

        self._image_topic = param_values[PARAM_PV_IMAGES_TOPIC]
        self._image_ts_topic = param_values[PARAM_PV_IMAGES_TS_TOPIC]
        self._hand_pose_topic = param_values[PARAM_HAND_POSE_TOPIC]
        self._audio_topic = param_values[PARAM_AUDIO_TOPIC]
        self._sm_topic = param_values[PARAM_SM_TOPIC]
        self._head_pose_topic = param_values[PARAM_HEAD_POSE_TOPIC]
        self.ip_addr = param_values[PARAM_IP_ADDR]
        self.pv_width = param_values[PARAM_PV_WIDTH]
        self.pv_height = param_values[PARAM_PV_HEIGHT]
        self.pv_framerate = param_values[PARAM_PV_FRAMERATE]
        self.sm_freq = param_values[PARAM_SM_FREQ]
        self._rm_depth_AHAT_topic = param_values[PARAM_RM_DEPTH_AHAT_TOPIC]

        # Define HL2SS server ports
        self.pv_port = hl2ss.StreamPort.PERSONAL_VIDEO
        self.si_port = hl2ss.StreamPort.SPATIAL_INPUT
        self.audio_port = hl2ss.StreamPort.MICROPHONE
        self.sm_port = hl2ss.IPCPort.SPATIAL_MAPPING
        self.rm_depth_AHAT_port = hl2ss.StreamPort.RM_DEPTH_AHAT

        self._head_pose_topic_enabled = False
        if self._head_pose_topic != DISABLE_TOPIC_STR:
            self._head_pose_topic_enabled = True
            # Establishing head-pose publisher before starting the _pv_thread
            # to avoid a race condition (this publisher can be used in that
            # thread
            log.info("Creating head pose publisher")
            self.ros_head_pose_publisher = self.create_publisher(
                HeadsetPoseData, self._head_pose_topic, 1
            )

            # Check to make sure image topic is valid, otherwise the image thread
            # will not be running, which is where head pose data is fetched.
            if self._image_topic == DISABLE_TOPIC_STR:
                log.warn(
                    "Warning! Image topic is not configured, so head pose data will not be published."
                )
        if self._image_topic != DISABLE_TOPIC_STR:
            # Create frame publisher
            self.ros_frame_publisher = self.create_publisher(
                Image, self._image_topic, 1
            )
            self.ros_frame_ts_publisher = self.create_publisher(
                Time, self._image_ts_topic, 1
            )
            self.connect_hl2ss_pv()
            log.info("PV client connected!")

            # Start the frame publishing thread
            self._pv_active = Event()
            self._pv_active.set()
            self._pv_rate_tracker = RateTracker()
            self._pv_thread = Thread(target=self.pv_publisher, name="publish_pv")
            self._pv_thread.daemon = True
            self._pv_thread.start()
        if self._hand_pose_topic != DISABLE_TOPIC_STR:
            # Create the hand joint pose publisher
            self.ros_hand_publisher = self.create_publisher(
                HandJointPosesUpdate, self._hand_pose_topic, 1
            )
            self.connect_hl2ss_si()
            log.info("SI client connected!")

            # Start the hand tracking data thread
            self._si_active = Event()
            self._si_active.set()
            self._si_rate_tracker = RateTracker()
            self._si_thread = Thread(target=self.si_publisher, name="publish_si")
            self._si_thread.daemon = True
            self._si_thread.start()
        if self._audio_topic != DISABLE_TOPIC_STR:
            # Create the audio publisher
            self.ros_audio_publisher = self.create_publisher(
                HeadsetAudioData, self._audio_topic, 1
            )
            self.connect_hl2ss_audio()
            log.info("Audio client connected!")

            # Start the audio data thread
            self._audio_active = Event()
            self._audio_active.set()
            self._audio_rate_tracker = RateTracker()
            self._audio_thread = Thread(
                target=self.audio_publisher, name="publish_audio"
            )
            self._audio_thread.daemon = True
            self._audio_thread.start()
        if self._sm_topic != DISABLE_TOPIC_STR:
            # Create the spatial map publisher
            self.ros_sm_publisher = self.create_publisher(
                SpatialMesh, self._sm_topic, 1
            )
            self.connect_hl2ss_sm()
            log.info("SM client connected!")

            # Start the spatial mapping thread
            self._sm_active = Event()
            self._sm_active.set()
            self._sm_rate_tracker = RateTracker()
            self._sm_thread = Thread(target=self.sm_publisher, name="publish_sm")
            self._sm_thread.daemon = True
            self._sm_thread.start()
        if self._rm_depth_AHAT_topic != DISABLE_TOPIC_STR:
            # Create frame publisher
            self.ros_depth_ahat_publisher = self.create_publisher(
                Image, self._rm_depth_AHAT_topic, 1
            )
            self.connect_hl2ss_rm_depth_ahat()
            log.info("RM Depth AHAT client connected!")

            # Start the frame publishing thread
            self._rm_depth_ahat_active = Event()
            self._rm_depth_ahat_active.set()
            self._rm_depth_ahat_rate_tracker = RateTracker()
            self._rm_depth_ahat_thread = Thread(
                target=self.rm_depth_ahat_publisher, name="publish_rm_depth_ahat"
            )
            self._rm_depth_ahat_thread.daemon = True
            self._rm_depth_ahat_thread.start()

        log.info("Initialization complete.")

    def connect_hl2ss_pv(self) -> None:
        """
        Creates the HL2SS PV client and connects it to the server on the headset.
        """
        # Operating mode
        # 0: video
        # 1: video + camera pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_1

        # Video encoding profile
        profile = hl2ss.VideoProfile.H265_MAIN

        # Decoded format
        decoded_format = "bgr24"

        hl2ss.start_subsystem_pv(self.ip_addr, self.pv_port)

        # Get the camera parameters for this configuration
        pv_cam_params = hl2ss.download_calibration_pv(
            self.ip_addr, self.pv_port, self.pv_width, self.pv_height, self.pv_framerate
        )
        self.camera_intrinsics = [float(x) for x in pv_cam_params.intrinsics.flatten()]

        self.hl2ss_pv_client = hl2ss.rx_decoded_pv(
            self.ip_addr,
            self.pv_port,
            hl2ss.ChunkSize.PERSONAL_VIDEO,
            mode,
            self.pv_width,
            self.pv_height,
            self.pv_framerate,
            profile,
            PV_BITRATE,
            decoded_format,
        )
        self.hl2ss_pv_client.open()

    def connect_hl2ss_si(self) -> None:
        """
        Creates the HL2SS Spatial Input (SI) client and connects it to the
        server on the headset.
        """
        self.hl2ss_si_client = hl2ss.rx_si(
            self.ip_addr, self.si_port, hl2ss.ChunkSize.SPATIAL_INPUT
        )
        self.hl2ss_si_client.open()

    def connect_hl2ss_audio(self) -> None:
        """
        Creates the HL2SS audio client and connects it to the
        server on the headset.
        """
        # AAC 24000 bytes/s per channel
        profile = hl2ss.AudioProfile.AAC_24000

        self.hl2ss_audio_client = hl2ss.rx_decoded_microphone(
            self.ip_addr, self.audio_port, hl2ss.ChunkSize.MICROPHONE, profile
        )
        self.hl2ss_audio_client.open()

    def connect_hl2ss_sm(self) -> None:
        """
        Creates the HL2SS Spatial Mapping (SM) client and connects it to the
        server on the headset.
        """
        self.hl2ss_sm_client = hl2ss.ipc_sm(self.ip_addr, self.sm_port)
        self.hl2ss_sm_client.open()

    def connect_hl2ss_rm_depth_ahat(self) -> None:
        """
        Creates the HL2SS RM Depth AHAT client and connects it to the server on the headset.
        """
        # Operating mode
        # 0: video
        # 1: video + rig pose
        # 2: query calibration (single transfer)
        mode = hl2ss.StreamMode.MODE_1

        # Video encoding profile
        profile = hl2ss.VideoProfile.H265_MAIN

        # TODO: Consider getting RM Depth AHAT camera intrinsics as well
        # client_rm_depth_ahat.py example in hl2ss repo shows that this can be done, but
        # would require some additional code changes (e.g., separate camera_intrinsics).
        # Link: https://github.com/jdibenes/hl2ss/blob/main/viewer/client_rm_depth_ahat.py

        self.hl2ss_rm_depth_ahat_client = hl2ss.rx_decoded_rm_depth_ahat(
            self.ip_addr,
            self.pv_port,
            hl2ss.ChunkSize.RM_DEPTH_AHAT,
            mode,
            profile,
            PV_BITRATE,
        )
        self.hl2ss_rm_depth_ahat_client.open()

    def shutdown_clients(self) -> None:
        """
        Shuts down the frame publishing thread and the HL2SS client.
        """
        log = self.get_logger()

        if self._image_topic != DISABLE_TOPIC_STR:
            # Stop frame publishing thread
            self._pv_active.clear()  # make RT active flag "False"
            self._pv_thread.join()
            log.info("PV thread closed")

            # Close client connections
            self.hl2ss_pv_client.close()
            hl2ss.stop_subsystem_pv(self.ip_addr, self.pv_port)
            log.info("PV client disconnected")

        if self._hand_pose_topic != DISABLE_TOPIC_STR:
            # Stop SI publishing thread
            self._si_active.clear()
            self._si_thread.join()
            log.info("SI thread closed")

            self.hl2ss_si_client.close()
            log.info("SI client disconnected")

        if self._audio_topic != DISABLE_TOPIC_STR:
            # Stop audio publishing thread
            self._audio_active.clear()
            self._audio_thread.join()
            log.info("Audio thread closed")

            self.hl2ss_audio_client.close()
            log.info("Audio client disconnected")

        if self._sm_topic != DISABLE_TOPIC_STR:
            # Stop SM publishing thread
            self._sm_active.clear()
            self._sm_thread.join()
            log.info("SM thread closed")

            self.hl2ss_sm_client.close()
            log.info("SM client disconnected")

        if self._rm_depth_AHAT_topic != DISABLE_TOPIC_STR:
            # Stop SM publishing thread
            self._rm_depth_ahat_active.clear()
            self._rm_depth_ahat_thread.join()
            log.info("RM Depth AHAT thread closed")

            self.hl2ss_rm_depth_ahat_client.close()
            log.info("RM Depth AHAT client disconnected")

    def pv_publisher(self) -> None:
        """
        Main thread that gets frames from the HL2SS PV client and publishes
        them to the image topic. For each image message published, a corresponding
        HeadsetPoseData message is published with the same header info and the world
        matrix for that image.
        """
        log = self.get_logger()

        while self._pv_active.wait(0):  # will quickly return false if cleared.
            # The data returned from HL2SS is just a numpy array of the
            # configured resolution. The payload array is in BGR 3-channel
            # format.
            data = self.hl2ss_pv_client.get_next_packet()

            try:
                stamp = self.get_clock().now().to_msg()
                image_msg = BRIDGE.cv2_to_imgmsg(data.payload.image, encoding="bgr8")
                image_msg.header.stamp = stamp
                image_msg.header.frame_id = "PVFramesBGR"
            except TypeError as e:
                log.warning(f"{e}")
                return

            # Publish the image msg
            self.ros_frame_publisher.publish(image_msg)
            self.ros_frame_ts_publisher.publish(image_msg.header.stamp)

            # Publish the corresponding headset pose msg
            world_matrix = [float(x) for x in data.pose.flatten()]

            # Cannot publish head poses if it was not enabled.
            if self._head_pose_topic_enabled:
                headset_pose_msg = HeadsetPoseData()
                # same timestamp/frame_id as image
                headset_pose_msg.header = image_msg.header
                headset_pose_msg.world_matrix = world_matrix
                headset_pose_msg.projection_matrix = self.camera_intrinsics
                self.ros_head_pose_publisher.publish(headset_pose_msg)

            self._pv_rate_tracker.tick()
            log.debug(
                f"Published image message (hz: "
                f"{self._pv_rate_tracker.get_rate_avg()})",
                throttle_duration_sec=1,
            )

    def si_publisher(self) -> None:
        """
        Thread the gets spatial input packets from the HL2SS SI client, converts
        the SI data to ROS messages, and publishes them.

        Currently only publishes hand tracking data. However, eye gaze data and
        head pose data is also available in the SI data packet.
        """
        log = self.get_logger()

        while self._si_active.wait(0):  # will quickly return false if cleared.
            data = self.hl2ss_si_client.get_next_packet()
            si_data = hl2ss.unpack_si(data.payload)

            # Publish the hand tracking data if it is valid
            if si_data.is_valid_hand_left():
                hand_msg_left = self.create_hand_pose_msg_from_si_data(si_data, "Left")
                self.ros_hand_publisher.publish(hand_msg_left)
            if si_data.is_valid_hand_right():
                hand_msg_right = self.create_hand_pose_msg_from_si_data(
                    si_data, "Right"
                )
                self.ros_hand_publisher.publish(hand_msg_right)

            self._si_rate_tracker.tick()
            log.debug(
                f"Published hand pose message (hz: "
                f"{self._si_rate_tracker.get_rate_avg()})",
                throttle_duration_sec=1,
            )

    def audio_publisher(self) -> None:
        """
        Thread the gets audio packets from the HL2SS audio client, converts
        the data to ROS HeadsetAudioData messages, and publishes them.
        """
        log = self.get_logger()

        while self._audio_active.wait(0):  # will quickly return false if cleared.
            data = self.hl2ss_audio_client.get_next_packet()

            n_channels, sample_len = data.payload.shape
            assert n_channels == hl2ss.Parameters_MICROPHONE.CHANNELS

            audio = np.zeros((data.payload.size), dtype=data.payload.dtype)
            audio[0::2] = data.payload[0, :]
            audio[1::2] = data.payload[1, :]

            sample_rate = hl2ss.Parameters_MICROPHONE.SAMPLE_RATE
            sample_duration = (1.0 / sample_rate) * sample_len

            audio_msg = HeadsetAudioData()
            audio_msg.header.stamp = self.get_clock().now().to_msg()
            audio_msg.header.frame_id = "AudioData"

            audio_msg.channels = n_channels
            audio_msg.sample_rate = sample_rate
            audio_msg.sample_duration = sample_duration
            audio_msg.data = audio.tolist()

            self.ros_audio_publisher.publish(audio_msg)

            self._audio_rate_tracker.tick()
            log.debug(
                f"Published audio message (hz: "
                f"{self._audio_rate_tracker.get_rate_avg()})",
                throttle_duration_sec=1,
            )

    def sm_publisher(self) -> None:
        """
        Thread that is responsible for fetching the spatial map data from HL2SS,
        converting the meshes to SpatialMesh ROS messages, and publishing them.

        Spatial meshes are retrieved every 5 seconds.
        """
        log = self.get_logger()

        # Maximum triangles per cubic meter
        tpcm = 1000

        # Data format
        vpf = hl2ss.SM_VertexPositionFormat.R32G32B32A32Float
        tif = hl2ss.SM_TriangleIndexFormat.R32Uint
        vnf = hl2ss.SM_VertexNormalFormat.R32G32B32A32Float

        # Include normals
        normals = True

        # Maximum number of active threads (on the HoloLens) to compute meshes
        n_threads = 2

        center = [0.0, 0.0, 0.0]  # Position of the box
        extents = [8.0, 8.0, 8.0]  # Dimensions of the box

        # Initialize observer and bounding volume
        self.hl2ss_sm_client.create_observer()
        volumes = hl2ss.sm_bounding_volume()
        volumes.add_box(center, extents)
        self.hl2ss_sm_client.set_volumes(volumes)

        while self._sm_active.wait(0):  # will quickly return false if cleared.
            ids = self.hl2ss_sm_client.get_observed_surfaces()
            tasks = hl2ss.sm_mesh_task()
            for i in ids:
                tasks.add_task(i, tpcm, vpf, tif, vnf, normals)

            meshes = self.hl2ss_sm_client.get_meshes(tasks, n_threads)
            log.debug(f"Received {len(meshes)} meshes")
            for index, mesh in meshes.items():
                id_hex = ids[index].hex()

                if mesh is None:
                    log.warning(
                        f"Task {index}: surface id {id_hex} compute mesh failed"
                    )
                    continue

                mesh.unpack(vpf, tif, vnf)

                log.debug(
                    f"Task {index}: surface id {id_hex} has {mesh.vertex_positions.shape[0]}"
                    f" vertices {mesh.triangle_indices.shape[0]}"
                    f" triangles {mesh.vertex_normals.shape[0]} normals"
                )

                mesh.vertex_positions[:, 0:3] = (
                    mesh.vertex_positions[:, 0:3] * mesh.vertex_position_scale
                )
                mesh.vertex_positions = mesh.vertex_positions @ mesh.pose
                mesh.vertex_normals = mesh.vertex_normals @ mesh.pose

                # Create the spatial mesh message for this mesh
                spatial_mesh_msg = SpatialMesh()
                spatial_mesh_msg.mesh_id = id_hex
                spatial_mesh_msg.removal = False  # Is this field even used?

                mesh_shape = Mesh()
                for ind in mesh.triangle_indices:
                    m_tri = MeshTriangle()
                    m_tri.vertex_indices = ind
                    mesh_shape.triangles.append(m_tri)
                for v in mesh.vertex_positions:
                    m_v = Point()
                    m_v.x = float(v[0])
                    m_v.y = float(v[1])
                    m_v.z = float(v[2])
                    mesh_shape.vertices.append(m_v)

                spatial_mesh_msg.mesh = mesh_shape

                # Publish!
                self.ros_sm_publisher.publish(spatial_mesh_msg)

            time.sleep(self.sm_freq)

    def rm_depth_ahat_publisher(self) -> None:
        """
        Main thread that gets depth frames from the HL2SS RM Depth AHAT client and publishes
        them to the rm_depth_ahat topic. For each image message published, a corresponding
        HeadsetPoseData message is published with the same header info and the world
        matrix for that image.
        """
        log = self.get_logger()

        # `.wait(0)` will quickly return false if cleared.
        while self._rm_depth_ahat_active.wait(0):
            data = self.hl2ss_rm_depth_ahat_client.get_next_packet()

            log.debug(
                f"Published RM Depth AHAT message. Pose at time "
                f"{data.timestamp} recorded.",
                data.pose,
            )

            # TODO: What is AB portion of data.payload (related to IR?), absolute? Albedo?
            # Should it be included here as a part of the depth message?
            try:
                # Assume BGR because AHAT shares the same video encoding profile as PV
                # Payload contains 16-bit Depth and 16-bit AB, so use BGR16
                rm_depth_ahat_msg = BRIDGE.cv2_to_imgmsg(
                    data.payload.depth, encoding="bgr16"
                )
                rm_depth_ahat_msg.header.stamp = (
                    self.get_cloc_rm_depth_audio_activek().now().to_msg()
                )
                rm_depth_ahat_msg.header.frame_id = "RMDepthAHATFrames"
            except TypeError as e:
                log.warning(f"{e}")
                return

            # Publish the RM Depth AHAT msg
            self.ros_depth_ahat_publisher.publish(rm_depth_ahat_msg)

            self._rm_depth_ahat_rate_tracker.tick()
            log.debug(
                f"Published RM Depth AHAT message (hz: "
                f"{self._rm_depth_ahat_rate_tracker.get_rate_avg()})",
                throttle_duration_sec=1,
            )

    def create_hand_pose_msg_from_si_data(
        self,
        si_data: hl2ss.unpack_si,
        hand: str,
    ) -> HandJointPosesUpdate:
        """
        Extracts the hand joint poses data from the HL2SS SI structure
        and forms a ROS HandJointPosesUpdate message.
        """
        log = self.get_logger()

        if hand == "Left":
            hand_data = si_data.get_hand_left()
        elif hand == "Right":
            hand_data = si_data.get_hand_right()
        else:
            log.warning(f"Could not process hand message with " f"handedness: {hand}")
            return

        joint_poses = []
        for j in range(0, hl2ss.SI_HandJointKind.TOTAL):
            pose = hand_data.get_joint_pose(j)

            # Extract the position
            position = Point()
            position.x = float(pose.position[0])
            position.y = float(pose.position[1])
            position.z = float(pose.position[2])

            # Extract the orientation
            orientation = Quaternion()
            orientation.x = float(pose.orientation[0])
            orientation.y = float(pose.orientation[1])
            orientation.z = float(pose.orientation[2])
            orientation.w = float(pose.orientation[3])

            # Form the geometry pose message
            pose_msg = Pose()
            pose_msg.position = position
            pose_msg.orientation = orientation

            # Create the hand joint pose message
            joint_pose_msg = HandJointPose()
            joint_pose_msg.joint = JOINT_LIST[j]
            joint_pose_msg.pose = pose_msg
            joint_poses.append(joint_pose_msg)

        # Create the top level hand joint poses update message
        hand_msg = HandJointPosesUpdate()
        hand_msg.header.stamp = self.get_clock().now().to_msg()
        hand_msg.header.frame_id = "HandJointPosesUpdate"
        hand_msg.hand = hand
        hand_msg.joints = joint_poses

        return hand_msg

    def destroy_node(self):
        """
        Clean up resources.
        """
        self.shutdown_clients()


main = make_default_main(HL2SSROSBridge)


if __name__ == "__main__":
    main()
