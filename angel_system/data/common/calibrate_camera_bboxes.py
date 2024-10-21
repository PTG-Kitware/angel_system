import configparser
import kwimage

import numpy as np


class CameraCalibration:
    def __init__(
        self,
        src_calibration_fp="config/camera_intrinsics/hololens2.conf",
        src_calibration_key="RIGHT_CAM_HD",
        dst_calibration_fp="config/camera_intrinsics/zed.conf",
        dst_calibration_key="RIGHT_CAM_HD"
    ):
        # Read the camera intrinsics
        configparser = configparser.ConfigParser()
        src_calibration_config = configparser.read(src_calibration_fp)
        dst_calibration_config = configparser.read(dst_calibration_fp)

        src_intrinsics = src_calibration_config[src_calibration_key]
        dst_intrinsics = dst_calibration_config[dst_calibration_key]

        self.k_src = np.array(
            [src_intrinsics["fx"], src_intrinsics["s"],  src_intrinsics["cx"]],
            [0,                    src_intrinsics["fy"], src_intrinsics["cy"]],
            [0,                    0,                    1]
        )
        self.k_dst = np.array(
            [dst_intrinsics["fx"], dst_intrinsics["s"],  dst_intrinsics["cx"]],
            [0,                    dst_intrinsics["fy"], dst_intrinsics["cy"]],
            [0,                    0,                    1]
        )

        self.k = np.matmul(self.k_dst, np.linalg.inv(self.k_src))
        
    def calibrate_bbox(self, bbox: kwimage.Box):
        # Make sure the bbox is tlbr
        bbox = bbox.toformat("tlbr")
        tl = bbox.data[:1]
        br = bbox.data[1:]

        # Convert coordinates from src to dst camera
        tl_dst = numpy.matmul(self.k, tl)
        br_dst = numpy.matmul(self.k, br)

        tl_final = tl_dst / tl_dst[2]
        br_final = br_dst / br_dst[2]

        return tl_final + br_final
