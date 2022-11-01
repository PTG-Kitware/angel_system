from array import array
from typing import List

import numpy as np
import numpy.typing as npt

# NOTE: These values were extracted from the projection matrix provided by the
# Unity main camera with the Windows MR plugin (now deprecated).
# For some unknown reason, the values provided by the Open XR plugin are
# different and do not provide the correct results. In the future, we should
# figure out why they are different or extract the focal length values from the
# MediaFrameReader which provides the frames, instead of the Unity main camera.
FOCAL_LENGTH_X = 1.6304
FOCAL_LENGTH_Y = 2.5084

PROJECTION_MATRIX = np.asarray([
    [FOCAL_LENGTH_X, 0.0, 0.0, 0.0],
    [0.0, FOCAL_LENGTH_Y, 0.0, 0.0],
    [0.0, 0.0, -1.0020020008087158, -0.20020020008087158],
    [0.0, 0.0, -1.0, 0.0]
])


def convert_1d_4x4_to_2d_matrix(matrix_1d: array) -> npt.NDArray:
    """
    Convert 1D array of length 16 to 2D 4x4 matrix.

    :param matrix_1d: 1D array of length 16.

    :returns: A 2D array of shape 4x4.
    """
    return np.asarray(matrix_1d).reshape(4, 4)


def project_3d_pos_to_2d_image(
    position: npt.NDArray,
    inverse_world_mat: npt.NDArray,
    projection_mat: npt.NDArray,
) -> List[float]:
    """
    Projects the 3d position vector into 2d image space using the given world
    to camera and camera projection matrices.

    The image coordinates returned are in the range [-1, 1]. Values outside of
    this range are clipped.

    :param position: 1D array representing the 3D position of the object in
        world space.
    :param inverse_world_mat: 2D matrix that is the inverse of the
        camera-to-world matrix (a.k.a the world-to-camera matrix). This is
        provided by the HoloLens RGB camera for each frame.
    :param projection_mat: 2D matrix to use to convert from camera space to
        image space. This is provided by the HoloLens. This is provided by
        the HoloLens RGB camera for each frame.

    :returns: The points on the 2D image plane as a list of [x, y, 1.0]
    """
    # Convert from world space to camera space
    x = np.matmul(inverse_world_mat, position)
    # Convert from camera space to image space
    image = np.matmul(projection_mat, x)

    return [image[0], image[1], 1.0]
