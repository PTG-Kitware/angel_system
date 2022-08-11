import numpy as np


def conf_func(dist, alpha, dth):
    dist = np.sqrt(np.sum((dist) ** 2, axis=-1))
    mask = dist < dth
    conf = np.exp(alpha * (1 - dist / dth))
    conf = mask * conf
    mean_conf = np.mean(conf, axis=-1)
    return mean_conf


def compute_corner_confidence(
    cp_pred_np: np.ndarray,
    obj_pose: np.ndarray,
    l_hand: np.ndarray,
    r_hand: np.ndarray,
    alpha: float = 2.0,
    dth=[75, 7.5],
):
    cp_gt = np.stack([obj_pose, l_hand, r_hand])
    cp_gt = cp_gt.reshape(cp_gt.shape[:-1] + (-1, 3))
    cp_pred_np = cp_pred_np.reshape(cp_pred_np.shape[:-1] + (-1, 3))
    dist = (
        cp_gt
        - cp_pred_np[
            :,
            :,
        ]
    )

    c_uv = conf_func(dist[..., :2], alpha, dth[0])
    z_mask = dist[..., -1] < dth[-1]
    c_z = np.mean(z_mask * np.abs(dist[..., -1]), axis=-1)
    conf = 0.5 * (c_uv + c_z)

    return conf


def convert2grid():
    pass
