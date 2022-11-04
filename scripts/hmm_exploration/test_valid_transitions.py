import numpy as np
import pandas as pd
import glob
import os
import copy
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import  accuracy_score
import scipy
import yaml
from scipy.optimize import minimize, root_scalar

from angel_system.activity_hmm.core import ActivityHMM, ActivityHMMRos, \
    get_skip_score, score_raw_detections, load_and_discretize_data
from angel_system.impls.detect_activities.swinb.swinb_detect_activities import SwinBTransformer
from angel_system.ptg_eval.common.load_data import time_from_name
from angel_system.ptg_eval.activity_classification.visualization import EvalVisualization
from angel_system.ptg_eval.activity_classification.compute_scores import EvalMetrics

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
        AutoMinorLocator
    HAS_MATLOTLIB = True
except ModuleNotFoundError:
    HAS_MATLOTLIB = False


dt = 0.5
num_activities = 6
num_steps = 10
class_str = ['Background']
for i in range(num_steps - 1):
    class_str.append(str(i + 1))

med_class_duration = [5]*num_steps
class_mean_conf = class_std_conf = np.ones((num_steps, num_activities))





num_steps_can_jump_fwd = 0
num_steps_can_jump_bck = 0
model = ActivityHMM(dt, class_str, med_class_duration,
             num_steps_can_jump_fwd, num_steps_can_jump_bck,
             class_mean_conf, class_std_conf)

# In first state
assert np.all(model.valid_trans[0] == [True, True, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[1] == model.valid_trans[0])
assert np.all(model.valid_trans[2] == [False,  False, True, True, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[3] == model.valid_trans[2])

assert np.all(model.valid_trans[16] == [False,  False, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, True, True])
assert np.all(model.valid_trans[17] == model.valid_trans[16])



num_steps_can_jump_fwd = 1
num_steps_can_jump_bck = 0
model = ActivityHMM(dt, class_str, med_class_duration,
             num_steps_can_jump_fwd, num_steps_can_jump_bck,
             class_mean_conf, class_std_conf)

# In first state
assert np.all(model.valid_trans[0] == [True, True, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[1] == [True, True, True, True, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[2] == [False,  False, True, True, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[3] == [False, False, True, True, True, True,
                                       False, False, False, False, False,
                                       False, False, False, False, False,
                                       False, False])

assert np.all(model.valid_trans[16] == [False,  False, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, True, True])
assert np.all(model.valid_trans[17] == model.valid_trans[16])


num_steps_can_jump_fwd = 0
num_steps_can_jump_bck = 1
model = ActivityHMM(dt, class_str, med_class_duration,
             num_steps_can_jump_fwd, num_steps_can_jump_bck,
             class_mean_conf, class_std_conf)

# In first state
assert np.all(model.valid_trans[0] == [True, True, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[1] == model.valid_trans[0])
assert np.all(model.valid_trans[2] == [False,  False, True, True, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, False, False])
assert np.all(model.valid_trans[3] == [True, True, True, True, False, False,
                                       False, False, False, False, False,
                                       False, False, False, False, False,
                                       False, False])

assert np.all(model.valid_trans[16] == [False,  False, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, False, False, True, True])
assert np.all(model.valid_trans[17] == [False,  False, False, False, False, False, False,
                                       False, False, False, False, False, False,
                                       False, True, True, True, True])
