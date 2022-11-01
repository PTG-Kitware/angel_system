def convert_1d_4x4_to_2d_matrix(matrix_1d):
    matrix_2d = [[], [], [], []]
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            matrix_2d[row].append(matrix_1d[idx])

    return matrix_2d


def project_3d_pos_to_2d_image(
    position,
    inverse_world_mat,
    projection_mat
):
    """
    Projects the 3d position vector into 2d image space using the given world
    to camera and camera projection matrices.

    The image coordinates returned are in the range [-1, 1]. Values outside of
    this range are clipped.
    """
    # Convert from world space to camera space
    x = np.matmul(inverse_world_mat, position)
    # Convert from camera space to image space
    image = np.matmul(projection_mat, x)
    #print("image coords", image)

    # Normalize
    image_scaled = image / image[3]
    #print("image scaled coords", image_scaled)

    image_scaled_x = image_scaled[0]
    image_scaled_y = image_scaled[1]

    '''

    # Limit to -1 and 1
    if image_scaled_x > 1:
        image_scaled_x = 1
    elif image_scaled_x < -1:
        image_scaled_x = -1
    if image_scaled_y > 1:
        image_scaled_y = 1
    elif image_scaled_y < -1:
        image_scaled_y = -1
    '''


    '''
    # Convert to screen coordinates
    projected_point_x = image_scaled[0] * half_width + half_width
    projected_point_y = image_scaled[1] * half_height + half_height
    #print("pixel coords", projected_point_x, projected_point_y)
    '''

    return [image_scaled_x, image_scaled_y, 1.0]
