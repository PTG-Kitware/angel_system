import numpy as np
import glob
import os

import angel_system
from angel_system.activity_hmm.core import ActivityHMMRos


# ----------------------------------------------------------------------------
base_path = os.path.split(os.path.abspath(angel_system.__file__))[0]
config_fname = base_path + '/../config/tasks/task_steps_config-recipe_coffee.yaml'

print(f'Loading HMM with recipe {config_fname}')
live_model = ActivityHMMRos(config_fname)

curr_step = 1
start_time = 0
end_time = 1

for _ in range(2):
    conf_vec = np.zeros(len(live_model.model.class_str));
    conf_vec[curr_step] = 1
    print('Sending confidence vector with all zeros except for step', curr_step)
    live_model.add_activity_classification(live_model.model.class_str,
                                           conf_vec, start_time, end_time)
    curr_step += 1
    start_time += 1.25
    end_time += 1.25

    print('\'get_current_state\' yields:', live_model.get_current_state())

print('Calling revert_to_step(1)')
live_model.revert_to_step(1)
print('\'get_current_state\' yields:', live_model.get_current_state())
