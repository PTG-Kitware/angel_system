from typing import Dict

import numpy as np


def obj_det2d_set_to_feature(
    label_vec,
    left,
    right,
    top,
    bottom,
    label_confidences,
    descriptors,
    obj_obj_contact_state,
    obj_obj_contact_conf,
    obj_hand_contact_state,
    obj_hand_contact_conf,
    label_to_ind: Dict[str, int],
    version: int = 1,
):
    """Convert ObjectDetection2dSet fields into a feature vector.

    :param label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.

    :param version:
        Version of the feature conversion approach.
    """
    num_act = len(label_to_ind)
    num_dets = len(label_vec)

    if version == 1:
        feature_vec = np.zeros(num_act)

        for i in range(num_dets):
            ind = label_to_ind[label_vec[i]]
            feature_vec[ind] = np.maximum(feature_vec[ind], label_confidences[i])
    elif version == 2:
        # This version path is not yet defined.
        raise NotImplementedError(
            "Version 2 is expected in the future but is " "not currently defined."
        )
        # feature_vec = np.zeros((3, num_act))
        # feature_vec[:, ind] = np.maximum(feature_vec[:, ind],
        #                                  [label_confidences[i],
        #                                   obj_obj_contact_state[i]*(obj_obj_contact_conf[i]*2-1),
        #                                   obj_hand_contact_state[i]*(obj_hand_contact_conf[i]*2-1)])
    else:
        raise NotImplementedError(f"Unhandled version '{version}'")

    return feature_vec
