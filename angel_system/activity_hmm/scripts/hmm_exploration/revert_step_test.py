import numpy as np
import glob
import os

import angel_system
from angel_system.activity_hmm.core import ActivityHMMRos

# os.chdir("/home/local/KHQ/matt.brown/libraries/angel_system")


# ----------------------------------------------------------------------------
base_path = os.path.split(os.path.abspath(angel_system.__file__))[0]
config_fname = (
    base_path + "/../config/tasks/task_steps_config-recipe_coffee_trimmed.yaml"
)

print(f"Loading HMM with recipe {config_fname}")
live_model = ActivityHMMRos(config_fname)

curr_step = 1
start_time = 0
end_time = 1
conf_means = live_model.get_hmm_mean_and_std()[0]

for _ in range(1, live_model.num_steps):
    conf_vec = conf_means[curr_step]
    print("Sending confidence vector with all zeros except for step", curr_step)
    live_model.add_activity_classification(
        range(live_model.num_activities), conf_vec, start_time, end_time
    )

    ret = live_model.analyze_current_state()
    times, state_sequence, step_finished_conf = ret

    assert state_sequence[-1] == curr_step

    print(
        "'get_current_state' yields:",
        live_model.class_str.index(live_model.get_current_state()),
    )

    curr_step += 1
    start_time += 1
    end_time += 1

for _ in range(1, live_model.num_steps):
    curr_step -= 1
    print(f"Calling revert_to_step({curr_step})")
    live_model.revert_to_step(curr_step)
    print("'get_current_state' yields:", live_model.get_current_state())
