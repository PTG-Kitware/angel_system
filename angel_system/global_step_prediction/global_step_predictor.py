from pathlib import Path

import yaml
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


class GlobalStepPredictor:
    def __init__(
        self,
        max_step_jump=2,
        threshold_multiplier=0.8,
        threshold_multiplier_weak=0.05,
        threshold_frame_count=8,  # full rate = 8, half rate = 4
        threshold_frame_count_weak=2,
        deactivate_thresh_mult=0.3,
        deactivate_thresh_frame_count=20,  # full rate = 20, half rate = 10
        deactivate_weak_thresh_mult=1.0,
        deactivate_thresh_frame_count_weak=2,
        recipe_types=[],
        recipe_config_dict={},
        background_threshold=0.3,
        activity_config_fpath="config/activity_labels/all_recipe_labels.yaml",
    ):
        """
        GlobalStepPredctor: based on a TCN activity classifier's activity classification
        outputs + a set of recipes, track what step a user is on for multiple recipes.
        """
        # TODO: make use of angel_system.data.config_structs instead of
        #       manually loading and accessing by string keys.
        
        print(f"activity_config_fpath: {activity_config_fpath}")
        with open(activity_config_fpath, "r") as stream:
            self.activity_config = yaml.safe_load(stream)
        num_activity_classes = len(self.activity_config["labels"])

        # maximum number of steps that can be "jumped" to.
        # i.e. if max_step_jump is 1, from step 2, you can only jump to 3.
        self.max_step_jump = max_step_jump
        # Earlier, we've caluclated the average activation for each activity,
        # when GT says that activity was happening.
        # Suppose that avg activation was 0.9. And threshold_multiplier = 0.8.
        # Then, the threshold to "activate" this step is 0.72.
        # activation must be > 0.72 for (int(threshold_frame_count)) frames.
        self.threshold_multiplier = threshold_multiplier
        # If max_step jump is 2, you could jump 2 frames, e.g. from 2 to 4.
        # The "weak threshold multiplier" is the minimum activation of the
        # "next step" needed to jump "two steps.".
        # Example: the tracker is on step 2. If step 3 "weakly activates", the
        # tracker still reports that it's on step 2, but it's now able to jump
        # to step 4 if step 4 activates.
        self.threshold_multiplier_weak = threshold_multiplier_weak
        self.threshold_frame_count = threshold_frame_count
        self.threshold_frame_count_weak = threshold_frame_count_weak

        self.deactivate_weak_thresh_mult = deactivate_weak_thresh_mult
        self.deactivate_thresh_frame_count_weak = deactivate_thresh_frame_count_weak

        # Threshold & nuber of consecutive below-threshold frames required
        # to deactivate an activity instance.
        self.deactivate_thresh_mult = deactivate_thresh_mult
        self.deactivate_thresh_frame_count = deactivate_thresh_frame_count

        # Glabally track current frame. When a new tracker is added,
        # The tracker dict "starting_frame_index" will indicate
        # how many frames were loaded before that. The first trackers
        # all start at frame 30, since the TCN takes in 30 frames.
        self.current_frame = 30

        self.activity_conf_history = np.empty((0, num_activity_classes))

        self.recipe_types = recipe_types
        # TODO: Expect use of angel_system.data.config_structs instead of
        #       a raw dictionary.
        self.recipe_configs = recipe_config_dict

        # Array of tracker dicts
        self.trackers = []
        for recipe in recipe_types:
            self.initialize_new_recipe_tracker(recipe)

        # activated_activites: shape = (number of activity indexes) x 2
        # - [ [ activity_index_0, num. activating/deactivating frames ],
        #       ...
        #   ]
        # When the left column is 0, the right column indicates the number of
        # previous frames in a row that have been "activated".
        # When the left column is 1, the right column indicates the number of
        # previous frames in a row that have been "DEactivated".
        max_activity_id_per_recipe = np.array(
            [
                np.max(np.array(tracker["granular_step_to_activity_id"]))
                for tracker in self.trackers
            ]
        )

        self.activated_activities = np.zeros(
            (
                np.max(max_activity_id_per_recipe) + 1,
                2,
            )
        )
        self.weak_activated_activities = np.zeros(
            (
                np.max(max_activity_id_per_recipe) + 1,
                2,
            )
        )

        self.gt_activities_order_from_each_config = {
            _recipe: self.get_activity_order_from_config(self.recipe_configs[_recipe])
            for _recipe in self.recipe_configs
        }
        # tracker resets: list of recipe name strings, for each reset we've had.
        # Example: ["tea", "coffee"]
        self.tracker_resets = []

    def get_activity_order_from_config(self, config_fn):
        """
        Get the order of activity_ids (mapping to granular step
        number) based on a recipe config
        """
        # TODO: make use of angel_system.data.config_structs instead of
        #       manually loading and accessing by string keys.
        with open(config_fn, "r") as stream:
            config = yaml.safe_load(stream)
        broad_steps = config["labels"]
        if broad_steps[0]["id"] == 1:
            config["labels"].insert(
                0,
                {
                    "id": 0,
                    "activity_ids": [0],
                    "label": "background",
                    "full_str": "background",
                },
            )
        return self.get_activity_per_granular_step(broad_steps)

    def compute_average_TP_activations(self, coco):
        # For each activity, given the Ground Truth-specified
        # frame subset where that activity is happening, get the
        # average activation of that class.

        all_activity_ids = np.unique(np.asarray(coco.images().lookup("activity_gt")))
        all_vid_ids = np.unique(np.asarray(coco.images().lookup("video_id")))
        print(
            f"Computing average true positive activations for {len(all_vid_ids)} videos."
        )

        # Don't use len() here... There might be skipped indexes.
        avg_probs = np.zeros(max(all_activity_ids) + 1)

        for activity_id in all_activity_ids:
            # image_ids = coco.index.vidid_to_gids[vid_id]
            image_ids = []
            for i in range(len(all_vid_ids)):
                image_ids.extend(
                    [
                        img["id"]
                        for img in coco.videos(video_ids=all_vid_ids).images[i].objs
                        if img["activity_gt"] == activity_id
                    ]
                )
            sub_dset = coco.subset(gids=image_ids, copy=True)
            probs_for_true_inds = np.asarray(sub_dset.images().lookup("activity_conf"))[
                :, activity_id
            ]
            avg_prob = np.mean(probs_for_true_inds)
            avg_probs[activity_id] = avg_prob

        self.avg_probs = avg_probs
        return self.avg_probs

    def get_average_TP_activations_from_file(self, fpath):
        self.avg_probs = np.load(fpath)

    def get_average_TP_activations_from_array(self, avg_probs):
        self.avg_probs = avg_probs

    def initialize_new_recipe_tracker(self, recipe, config_fn=None):
        """
        tracker dict fields:
            {
            "recipe":"coffee"
                - all options: ["coffee", "tea", "dessert_quesadilla",
                  "oatmeal", "pinwheel"]
            "current_step": 5
                - int type.
                - 0 = background.
            "total_num_steps": 23
                - int type.
            "can_skip": False
                - boolean type.
                - Switched to "True" when the "weak threshold" has been met.
                  i.e. the "weak threshold" has been achieved for
                  "threshold_frame_count_weak" number of frames.
                  When this happens, the tracker can "skip" a step.
            "step_to_activity_id": [0,1,2,3,24,4,...]
                - array type.
                - steps_to_activities[current_step] is the current activity index
                  the tracker is on. (step index != activity index).
            "step_to_activity_desc": ["background", ...]
            "starting_frame_index": 30
                - index videos frame at which point this tracker instance was initialized.
            "prediction_history": [0,0,0,...,1,1,1,...23,23]
            "active": True
                - When true, keep checking to update this recipe.
                - When false, the recipe is complete... stop adding to its history
                  or checking for a next step.

        """
        tracker_dict = {}
        print(f"self.recipe_configs[recipe]: {self.recipe_configs}")
        config_fn = self.recipe_configs[recipe]

        # Read in task config
        # TODO: make use of angel_system.data.config_structs instead of
        #       manually loading and accessing by string keys.
        with open(config_fn, "r") as stream:
            config = yaml.safe_load(stream)
        labels = [self.sanitize_str(l["full_str"]) for l in config["labels"]]
        broad_steps = config["labels"]
        if broad_steps[0]["id"] == 1:
            config["labels"].insert(
                0,
                {
                    "id": 0,
                    "activity_ids": [0],
                    "label": "background",
                    "full_str": "background",
                },
            )

        tracker_dict[
            "last_granular_step_per_broad_step"
        ] = self.get_last_granular_step_per_broad_step(broad_steps)
        tracker_dict["recipe"] = recipe

        tracker_dict["current_broad_step"] = 0
        tracker_dict["current_granular_step"] = 0

        tracker_dict["total_num_broad_steps"] = len(broad_steps)
        tracker_dict["total_num_granular_steps"] = np.sum(
            [len(self.get_unique(step["activity_ids"])) for step in config["labels"]]
        )

        tracker_dict["skipped_granular_steps"] = []

        # Activity ids
        tracker_dict["broad_step_to_activity_ids"] = [
            self.get_unique(step["activity_ids"]) for step in broad_steps
        ]
        tracker_dict[
            "granular_step_to_activity_id"
        ] = self.get_activity_per_granular_step(broad_steps)

        # Labels
        tracker_dict["broad_step_to_label"] = [step["label"] for step in broad_steps]

        # Full strings
        tracker_dict["broad_step_to_full_str"] = [
            step["full_str"] for step in broad_steps
        ]
        tracker_dict["granular_step_to_full_str"] = [
            self.get_activity_str_from_id(act_id)
            for act_id in tracker_dict["granular_step_to_activity_id"]
        ]

        # Prediction History
        tracker_dict["broad_step_prediction_history"] = np.array([])
        tracker_dict["granular_step_prediction_history"] = np.array([])

        tracker_dict["active"] = True
        tracker_dict["broad_steps"] = broad_steps
        tracker_dict["can_skip"] = False

        self.trackers.append(tracker_dict)

    def get_activity_per_granular_step(self, broad_steps):
        activity_id_per_granular_step = []
        for broad_step in broad_steps:
            for activity_id in self.get_unique(broad_step["activity_ids"]):
                activity_id_per_granular_step.append(activity_id)
        return activity_id_per_granular_step

    def increment_granular_step(self, tracker_ind):
        """
        Increment a tracker's granular step, and also update the tracker's
        broad step.
        (Many broad steps contain several granular steps, so a granular step
        increment may or may not entail a broad step increment.)
        """
        tracker = self.trackers[tracker_ind]
        current_granular_step = tracker["current_granular_step"]
        num_granular_steps = tracker["total_num_granular_steps"] - 1

        if current_granular_step < num_granular_steps:
            self.trackers[tracker_ind]["current_granular_step"] += 1
            self.trackers[tracker_ind][
                "current_broad_step"
            ] = self.granular_to_broad_step(tracker, current_granular_step)
        elif current_granular_step == num_granular_steps and tracker["active"] == True:
            self.trackers[tracker_ind]["active"] = False
        else:
            raise Exception(
                f"Tried to increment tracker #{tracker_ind}: "
                f"{tracker['recipe']} past last step."
            )

        self.conditionally_reset_irrational_trackers(tracker)
        return self.trackers

    def decrement_granular_step(self, tracker_ind):
        """
        Decrement a tracker's granular step, and also update the tracker's
        broad step.
        (Many broad steps contain several granular steps, so a granular step
        increment may or may not entail a broad step increment.)
        """
        tracker = self.trackers[tracker_ind]

        num_granular_steps = tracker["total_num_granular_steps"] - 1
        current_granular_step = tracker["current_granular_step"]

        if (
            current_granular_step == num_granular_steps
            and not self.trackers[tracker_ind]["active"]
        ):
            self.trackers[tracker_ind]["active"] = True
            return self.trackers

        if current_granular_step > 0:
            self.trackers[tracker_ind]["current_granular_step"] -= 1
            self.trackers[tracker_ind][
                "current_broad_step"
            ] = self.granular_to_broad_step(tracker, current_granular_step)
        else:
            raise Exception(
                f"Tried to decrement tracker #{tracker_ind}: "
                f"{tracker['recipe']} already on step 0."
            )
        return self.trackers

    def reset_one_tracker(self, tracker_ind):
        """
        Set a tracker's granular & broad steps to 0.

        NOTE: you can still see any nonzero steps in the prediction_history.

        Also, note that you'll have to process more activity confidence vectors
        to see zeros in your prediction history.
        """
        print(f"RESETTING tracker {tracker_ind}")
        self.trackers[tracker_ind]["current_broad_step"] = 0
        self.trackers[tracker_ind]["current_granular_step"] = 0
        self.tracker_resets.append(self.trackers[tracker_ind]["recipe"])

    def granular_to_broad_step(self, tracker, granular_step):
        """
        Convert granular step to broad step.
        last_granular_step_per_broad_step
        Ex: [0, 2, 5, 6, 7]
        granular_step_4
        """
        lgspbs = np.array(tracker["last_granular_step_per_broad_step"])
        return len(np.nonzero(lgspbs < granular_step)[0])

    def get_unique(self, activity_ids):
        """
        Some steps have the same activity more than once. In those cases,
        keep them in order but only have each activity once.
        Ex:
          - [3, 2, 1, 3, 2, 1] --> [3, 2, 1]
          - [5, 5, 5, 5, 5] --> [5]
        """
        indexes = np.unique(activity_ids, return_index=True)[1]
        return [activity_ids[index] for index in sorted(indexes)]

    def get_last_granular_step_per_broad_step(self, steps):
        """
        Get last substep index of each broad step.
        Example: a recipe might have 8 succinct steps, with
        multiple activities per succinct step:
        step 0: background
        step 1: activity 1, activity 2, activity 3
        step 2: activity 10, activity 4
        step 3: activity 20.

        last_granular_step_per_broad_step = [0,3,5,6].
        """
        last_granular_step_per_broad_step = []
        total_granular_steps_to_here = 0
        for step in steps:
            if step["id"] == 0:
                last_granular_step_per_broad_step.append(0)
            else:
                num_substeps = len(self.get_unique(step["activity_ids"]))
                total_granular_steps_to_here += num_substeps
                last_granular_step_per_broad_step.append(total_granular_steps_to_here)
        return last_granular_step_per_broad_step

    def process_new_confidences(self, activity_confs):
        # assert np.array(activity_confs).shape[1] <= len(self.activated_activities)

        activated_indexes = np.nonzero(self.activated_activities[:, 0] == 1)[0]
        deactivated_indexes = np.nonzero(self.activated_activities[:, 0] == 0)[0]
        assert len(activated_indexes) + len(deactivated_indexes) == len(
            self.activated_activities
        )
        weak_activated_indexes = np.nonzero(self.weak_activated_activities[:, 0] == 1)[
            0
        ]
        weak_deactivated_indexes = np.nonzero(
            self.weak_activated_activities[:, 0] == 0
        )[0]

        # activity conf = vector of all activities for one frame.
        for i, activity_conf in enumerate(activity_confs):
            # activated_confidences = classes predicted to be "happening" now.
            activated_indexes = np.nonzero(self.activated_activities[:, 0] == 1)[0]
            # deactivated_confidences = classes predicted NOT to be happening yet.

            # WEAK
            deactivated_indexes = np.nonzero(self.activated_activities[:, 0] == 0)[0]
            # activated_confidences = classes predicted to be "happening" now.
            weak_activated_indexes = np.nonzero(
                self.weak_activated_activities[:, 0] == 1
            )[0]
            # deactivated_confidences = classes predicted NOT to be happening yet.
            weak_deactivated_indexes = np.nonzero(
                self.weak_activated_activities[:, 0] == 0
            )[0]

            # STRONG
            # Check for activations > 80% * (avg act threshold)
            above_pos_threshold_indexes = np.nonzero(
                activity_conf > self.threshold_multiplier * self.avg_probs
            )[0]
            # ...and < 30% * (avg act threshold)
            below_pos_threshold_indexes = np.nonzero(
                activity_conf < self.threshold_multiplier * self.avg_probs
            )[0]
            # Check for activations > 80% * (avg act threshold)
            above_neg_threshold_indexes = np.nonzero(
                activity_conf > self.deactivate_thresh_mult * self.avg_probs
            )[0]
            # ...and < 30% * (avg act threshold)
            below_neg_threshold_indexes = np.nonzero(
                activity_conf < self.deactivate_thresh_mult * self.avg_probs
            )[0]
            # WEAK
            # Check for activations > 80% * (avg act threshold)
            above_weak_pos_threshold_indexes = np.nonzero(
                activity_conf > self.threshold_multiplier_weak * self.avg_probs
            )[0]
            # ...and < 30% * (avg act threshold)
            below_pos_weak_threshold_indexes = np.nonzero(
                activity_conf < self.threshold_multiplier_weak * self.avg_probs
            )[0]
            # Check for activations > 80% * (avg act threshold)
            above_neg_weak_threshold_indexes = np.nonzero(
                activity_conf > self.deactivate_weak_thresh_mult * self.avg_probs
            )[0]
            # ...and < 30% * (avg act threshold)
            below_neg_weak_threshold_indexes = np.nonzero(
                activity_conf < self.deactivate_weak_thresh_mult * self.avg_probs
            )[0]

            # STRONG
            indexes_to_add_pos_threshold_increment = np.intersect1d(
                np.array(deactivated_indexes), np.array(above_pos_threshold_indexes)
            )
            self.activated_activities[indexes_to_add_pos_threshold_increment, 1] += 1
            indexes_to_add_neg_threshold_increment = np.intersect1d(
                np.array(activated_indexes), np.array(below_neg_threshold_indexes)
            )
            self.activated_activities[indexes_to_add_neg_threshold_increment, 1] += 1
            # WEAK
            indexes_to_add_weak_pos_threshold_increment = np.intersect1d(
                np.array(weak_deactivated_indexes),
                np.array(above_weak_pos_threshold_indexes),
            )
            self.weak_activated_activities[
                indexes_to_add_weak_pos_threshold_increment, 1
            ] += 1
            indexes_to_add_weak_neg_threshold_increment = np.intersect1d(
                np.array(weak_activated_indexes),
                np.array(below_neg_weak_threshold_indexes),
            )
            self.weak_activated_activities[
                indexes_to_add_weak_neg_threshold_increment, 1
            ] += 1

            # Not activated AND not above activation threshold?
            # Then set the activation clock back to zero.
            indexes_to_reset_deactive = np.intersect1d(
                np.array(deactivated_indexes), np.array(below_pos_threshold_indexes)
            )
            self.activated_activities[indexes_to_reset_deactive, 1] = 0
            indexes_to_reset_active = np.intersect1d(
                np.array(activated_indexes), np.array(above_neg_threshold_indexes)
            )
            self.activated_activities[indexes_to_reset_active, 1] = 0
            # WEAK
            indexes_to_reset_weak_deactive = np.intersect1d(
                np.array(weak_deactivated_indexes),
                np.array(below_pos_weak_threshold_indexes),
            )
            self.weak_activated_activities[indexes_to_reset_deactive, 1] = 0
            indexes_to_reset_weak_active = np.intersect1d(
                np.array(weak_activated_indexes),
                np.array(above_neg_weak_threshold_indexes),
            )
            self.weak_activated_activities[indexes_to_reset_active, 1] = 0

            # Activate the inactive classes that have surpassed the
            # activation threshold for enough consecutive frames
            flipping_on_indexes = np.intersect1d(
                deactivated_indexes,
                np.nonzero(
                    self.activated_activities[:, 1] >= self.threshold_frame_count
                )[0],
            )
            self.activated_activities[flipping_on_indexes, 0] = 1
            self.activated_activities[flipping_on_indexes, 1] = 0

            # Deactivate the active classes that have surpassed the
            # deactivation threshold for enough consecutive frames
            flipping_off_indexes = np.intersect1d(
                activated_indexes,
                np.nonzero(
                    self.activated_activities[:, 1]
                    >= self.deactivate_thresh_frame_count
                )[0],
            )
            self.activated_activities[flipping_off_indexes, 0] = 0
            self.activated_activities[flipping_off_indexes, 1] = 0

            # WEAK
            # Activate the inactive classes that have surpassed the
            # activation threshold for enough consecutive frames
            flipping_on_weak_indexes = np.intersect1d(
                weak_deactivated_indexes,
                np.nonzero(
                    self.weak_activated_activities[:, 1]
                    >= self.threshold_frame_count_weak
                )[0],
            )
            self.weak_activated_activities[flipping_on_weak_indexes, 0] = 1
            self.weak_activated_activities[flipping_on_weak_indexes, 1] = 0

            # Deactivate the active classes that have surpassed the
            # deactivation threshold for enough consecutive frames
            flipping_off_weak_indexes = np.intersect1d(
                weak_activated_indexes,
                np.nonzero(
                    self.weak_activated_activities[:, 1]
                    >= self.deactivate_thresh_frame_count_weak
                )[0],
            )
            self.weak_activated_activities[flipping_off_weak_indexes, 0] = 0
            self.weak_activated_activities[flipping_off_weak_indexes, 1] = 0

            # Now, go through ACTIVE trackers and see which corresponds to "flipping_on_indexes."
            # If a tracker's last step is ACTIVE and its last step is in "flipping_off_indexes",
            # then deactivate the tracker.
            for tracker_ind, tracker in enumerate(self.trackers):
                if not tracker["active"]:
                    continue

                # TODO: For now the tracker can jump 1 or 2 steps, if base
                # jump criteria is met. Add "weak" threshold too.
                current_granular_step = tracker["current_granular_step"]
                current_activity = tracker["granular_step_to_activity_id"][
                    current_granular_step
                ]
                if current_granular_step == tracker["total_num_granular_steps"] - 1:
                    if current_activity in flipping_off_indexes:
                        tracker["active"] = False
                    continue

                next_granular_step = current_granular_step + 1
                next_next_granular_step = min(
                    next_granular_step + 1, tracker["total_num_granular_steps"] - 1
                )
                next_activity = tracker["granular_step_to_activity_id"][
                    next_granular_step
                ]
                next_next_activity = tracker["granular_step_to_activity_id"][
                    next_next_granular_step
                ]

                # TODO: prioritize a 1-step jump over a 2-step jump. Create a
                # second loop just for the 2-step jumps, after this loop has completed
                # searching for one-step jumps.
                if next_activity in flipping_on_indexes:
                    self.increment_granular_step(tracker_ind)
                    self.conditionally_reset_irrational_trackers(tracker)
                    self.trackers[tracker_ind]["can_skip"] = False
                    # Each activity activation can only be used once.
                    # Delete activity from flipping_on_indexes
                    next_act_ind = np.argwhere(flipping_on_indexes == next_activity)
                    # TODO: disabling for now. All activities will
                    # count toward any relevant trackers.
                    if False:
                        if self.should_this_activity_trigger_be_used_once(next_act_ind):
                            flipping_on_indexes = np.delete(
                                flipping_on_indexes, next_act_ind
                            )
                elif next_next_activity in flipping_on_indexes and tracker["can_skip"]:
                    # Keep track of skipped steps
                    self.add_skipped_granular_step(tracker_ind, next_granular_step)
                    # Increment the granular step twice
                    self.increment_granular_step(tracker_ind)
                    self.increment_granular_step(tracker_ind)
                    self.conditionally_reset_irrational_trackers(tracker, skip=True)
                    self.trackers[tracker_ind]["can_skip"] = False
                    # Each activity activation can only be used once.
                    # Delete activity from flipping_on_indexes
                    next_next_act_ind = np.argwhere(
                        flipping_on_indexes == next_next_activity
                    )
                    # TODO: disabling for now. All activities will
                    # count toward any relevant trackers.
                    if False:
                        if self.should_this_activity_trigger_be_used_once(
                            next_next_act_ind
                        ):
                            flipping_on_indexes = np.delete(
                                flipping_on_indexes, next_next_act_ind
                            )
                if next_activity in flipping_on_weak_indexes:
                    self.trackers[tracker_ind]["can_skip"] = True

                # TODO: Try requiring that previous step is de-activated

                # Add current preds to this tracker's prediction history
                self.record_history(
                    tracker_ind,
                    tracker["current_granular_step"],
                    tracker["current_broad_step"],
                )
        # Update the current_frame
        self.current_frame += len(activity_confs)

        self.activity_conf_history = np.append(
            self.activity_conf_history, activity_confs, axis=0
        )

        return self.trackers

    def find_trackers_by_recipe(self, recipe):
        tracker_index_list = []
        for tracker_ind, tracker in enumerate(self.trackers):
            if tracker["recipe"] == recipe:
                tracker_index_list.append(tracker_ind)
        return tracker_index_list

    def conditionally_reset_irrational_trackers(self, tracker, skip=False):
        """
        A rational recipe tracker reset.

        Assuming one pot of water with 12oz poured into it:
        - you can't make tea while coffee is between broad steps 1-7
          (Reset tea at coffee granular step 20)
        - you can't make coffee while tea is between broad steps 1-
          (Reset coffee at tea granular step 8)

        Assuming one cutting board:
        - you can't make dessert quesadilla while pinwheel is between
          broad steps 1-6
          (Reset dessert quesadilla at pinwheel granular step 10)
        - you can't make pinwheel while dessert quesadilla is between
          broad steps 1-5
          (Reset pinwheel at dessert q granular step 11)
        """
        resetter_granular_step = {
            # recipe: [ (gran index to trigger reset,
            #            recipe that should reset)
            "coffee": [17, "tea"],
            "tea": [7, "coffee"],
            # "oatmeal": [],
            "pinwheel": [10, "dessert_quesadilla"],
            "dessert_quesadilla": [11, "pinwheel"],
        }
        if not skip:
            for recipe in resetter_granular_step:
                if (
                    tracker["recipe"] == recipe
                    and tracker["current_granular_step"]
                    == resetter_granular_step[recipe][0]
                ):
                    print("reset condition hit!!")
                    for tracker_ind in self.find_trackers_by_recipe(
                        resetter_granular_step[recipe][1]
                    ):
                        if (
                            self.trackers[tracker_ind]["current_granular_step"]
                            < resetter_granular_step[
                                self.trackers[tracker_ind]["recipe"]
                            ][0]
                        ):
                            self.reset_one_tracker(tracker_ind)
        else:
            for recipe in resetter_granular_step:
                granular_steps = [
                    resetter_granular_step[recipe][0],
                    resetter_granular_step[recipe][0] + 1,
                ]
                if (
                    tracker["recipe"] == recipe
                    and tracker["current_granular_step"] in granular_steps
                ):
                    for tracker_ind in self.find_trackers_by_recipe(
                        resetter_granular_step[recipe][1]
                    ):
                        if (
                            self.trackers[tracker_ind]["current_granular_step"]
                            < resetter_granular_step[
                                self.trackers[tracker_ind]["recipe"]
                            ][0]
                        ):
                            self.reset_one_tracker(tracker_ind)

    def should_this_activity_trigger_be_used_once(self, activity_id):
        """
        Should this activity trigger be used once?
        Or should we use it to trigger any step increment it applies to?
        """
        # TODO: un-hard-code these indexes. Find them in the right configs.
        shared_activity_indexes = []
        if "tea" in self.recipe_types and "coffee" in self.recipe_types:
            # the activities in broad_step 1 are the same for tea and coffee
            shared_activity_indexes.extend([8, 1, 2])
        if (
            "pinwheel" in self.recipe_types
            and "dessert_quesadilla" in self.recipe_types
        ):
            shared_activity_indexes.extend([6])
        if activity_id in shared_activity_indexes:
            return False
        return True

    def add_skipped_granular_step(self, tracker_ind, granular_step):
        self.trackers[tracker_ind]["skipped_granular_steps"].append(granular_step)

    def get_skipped_steps_one_tracker(self, tracker_ind):
        tracker = self.trackers[tracker_ind]
        skipped_steps = []
        for granular_step in tracker["skipped_granular_steps"]:
            skipped_steps.append(
                {
                    "recipe": tracker["recipe"],
                    "granular": granular_step,
                    "part_of_broad": self.granular_to_broad_step(
                        tracker, granular_step
                    ),
                    "activity_id": self.get_activity_from_granular_step(
                        tracker, granular_step
                    )[0],
                    "activity_str": self.get_activity_from_granular_step(
                        tracker, granular_step
                    )[1],
                }
            )
        return skipped_steps

    def get_activity_from_granular_step(self, tracker, granular_step):
        activity_id = tracker["granular_step_to_activity_id"][granular_step]
        activity_str = self.get_activity_str_from_id(activity_id)

        return activity_id, activity_str

    def get_activity_str_from_id(self, activity_id):
        for activity in self.activity_config["labels"]:
            if activity["id"] == activity_id:
                return activity["full_str"]

    def get_skipped_steps_all_trackers(self):
        skipped_steps_all_trackers = []
        for tracker_ind, tracker in enumerate(self.trackers):
            skipped_steps_all_trackers.append(
                self.get_skipped_steps_one_tracker(tracker_ind)
            )
        return skipped_steps_all_trackers

    def record_history(self, tracker_ind, current_granular_step, current_broad_step):
        self.trackers[tracker_ind]["broad_step_prediction_history"] = np.append(
            self.trackers[tracker_ind]["broad_step_prediction_history"],
            current_broad_step,
        )
        self.trackers[tracker_ind]["granular_step_prediction_history"] = np.append(
            self.trackers[tracker_ind]["granular_step_prediction_history"],
            current_granular_step,
        )

    def plot_gt_vs_predicted_one_recipe(
        self,
        step_gts,  # the granular_step_gts or broad_step_gts
        recipe_type,
        fname_suffix=None,
        granular_or_broad="granular",  # "granular" or "broad"
        output_dir="outputs",
    ) -> Path:
        """
        Plot gt vs predicted class across all vid frames.

        :returns: The filesystem path written to.
        """
        assert granular_or_broad in ["granular", "broad"]
        assert recipe_type in self.recipe_types
        fig = plt.figure()
        sn.set(font_scale=1)
        step_gts = [float(i) for i in step_gts]
        plt.plot(step_gts, label=f"{granular_or_broad}_step_gt")
        for i, tracker in enumerate(self.trackers):
            step_predictions = tracker[f"{granular_or_broad}_step_prediction_history"]
            plt.plot(
                step_predictions,
                label=f"estimated_{granular_or_broad}_steps_{tracker['recipe']}_{i}",
            )

        plt.legend()
        if not fname_suffix:
            fname_suffix = f"vid{vid_id}"
        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)
        title = f"plot_pred_vs_gt_{recipe_type}_{fname_suffix}.png"
        plt.title(title)
        output_file_path = output_dir_p / title
        fig.savefig(output_file_path)
        return output_file_path

    def determine_recipe_from_gt_first_activity(self, activity_gts):
        """
        Current rough strategy: check for all of the first five
        ground truth labels from a video's ground truth being in
        the first 10 activity IDs from a config's list of
        activities.
        """
        first_five_gt_acts_from_vid = self.get_unique(activity_gts)[1:6]
        for _recipe in self.gt_activities_order_from_each_config:
            if all(
                vid_gt in self.gt_activities_order_from_each_config[_recipe]
                for vid_gt in first_five_gt_acts_from_vid
            ):
                return _recipe
        return "unknown_recipe_type"

        """
        for activity_gt in activity_gts:
            if activity_gt > 0:
                first_activity_gt = activity_gt
                break
        for tracker in self.trackers:
            if tracker["granular_step_to_activity_id"][1] == activity_gt:
                if tracker["recipe"] in ["coffee", "tea"]:
                    # Coffee and tea have the same broad step 1.
                    # TODO: For now I'm just using the specific activity configs
                    # for this TCN to discriminate tea and coffee. Do something
                    # a little more general later.
                    for activity_gt in activity_gts:
                        if activity_gt not in [0, 8, 1, 2]:
                            if activity_gt == 25:
                                return "coffee"
                            elif activity_gt in [3, 4, 5]:
                                return "tea"
                            else:
                                raise Exception(
                                    "Can't tell what recipe this should be based on the activities. First activity_id that's not 0, 8, 1, or 2 was {activity_gt}."
                                )

                if tracker["recipe"] in ["dessert_quesadilla", "pinwheel"]:
                    # Pinwheel and quesa have the same broad step 1.
                    # TODO: For now I'm just using the specific activity configs
                    # for this TCN to discriminate tea and coffee. Do something
                    # a little more general later.
                    for activity_gt in activity_gts:
                        if activity_gt not in [0, 6]:
                            if activity_gt == 9:
                                return "dessert_quesadilla"
                            elif activity_gt in [10, 13]:
                                return "pinwheel"
                            else:
                                raise Exception(
                                    "Can't tell what recipe this should be based on the activities. First activity_id that's not 0, or 6 was {activity_gt}."
                                )
                else:
                    return tracker["recipe"]
        return "unknown_recipe_type"
        """

    def plot_gt_vs_predicted_plus_activations(self, step_gts, fname_suffix=None):
        # Plot gt vs predicted class across all vid frames
        fig = plt.figure()
        sn.set(font_scale=1)
        step_gts = [float(i) for i in step_gts]
        plt.plot(step_gts, label="gt")
        for i, tracker in enumerate(self.trackers):
            step_predictions = tracker["prediction_history"]
            plt.plot(step_predictions, label=f"estimated_{tracker['recipe']}_{i}")
        starting_zero_value = 0
        for i in range(len(self.avg_probs)):
            starting_zero_value -= 2
            plot_line = np.asarray(self.activity_conf_history)[:, i]
            plt.plot(2 * plot_line + starting_zero_value, label=f"act_preds[{i}]")
            plt.plot(
                [
                    2 * int(j < self.avg_probs[i]) + starting_zero_value
                    for j in plot_line
                ],
                label=f"act_preds_threshold[{i}]",
            )

        plt.legend()
        if not fname_suffix:
            fname_suffix = f"vid{vid_id}"
        recipe_type = self.determine_recipe_from_gt_first_step(step_gts)
        title = f"plot_pred_vs_gt_{recipe_type}_{fname_suffix}.png"
        plt.title(title)
        fig.savefig(f"./outputs/{title}")
        plt.close()

    def sanitize_str(self, str_: str):
        """
        Convert string to lowercase and emove trailing whitespace and period.

        :param str_: Input text

        :return: ``str_`` converted to lowercase and stripped of trailing whitespace and period.
        :rtype: str
        """
        return str_.lower().strip(" .")

    def manually_increment_current_broad_step(self, tracker_index):
        """
        Increment to the first granular step of the next broad step.
        """
        tracker = self.trackers[tracker_index]
        num_broad_steps = tracker["total_num_broad_steps"] - 1
        current_broad_step = tracker["current_broad_step"]

        if current_broad_step < num_broad_steps:
            self.trackers[tracker_index]["current_broad_step"] += 1
            self.trackers[tracker_index]["current_granular_step"] = (
                tracker["last_granular_step_per_broad_step"][
                    tracker["current_broad_step"] - 1
                ]
                + 1
            )
        elif (
            current_broad_step == num_broad_steps
            and self.trackers[tracker_ind]["active"] == True
        ):
            self.trackers[tracker_ind]["current_granular_step"] = (
                self.trackers[tracker_ind]["total_num_granular_steps"] - 1
            )
            self.trackers[tracker_ind]["active"] = False
        else:
            raise Exception(
                f"Tried to increment tracker #{tracker_ind}: "
                f"{tracker['recipe']} past last step."
            )

        self.conditionally_reset_irrational_trackers(tracker)
        return self.trackers

    def manually_decrement_current_step(self, tracker_index):
        """
        Decrement to the first granular step of the previous broad step.
        """
        tracker = self.trackers[tracker_index]
        num_broad_steps = tracker["total_num_broad_steps"] - 1
        current_broad_step = tracker["current_broad_step"]

        if current_broad_step == num_broad_steps and not tracker["active"]:
            self.trackers[tracker_ind]["active"] = True
            return self.trackers

        if current_broad_step > 0:
            self.trackers[tracker_index]["current_broad_step"] -= 1
            self.trackers[tracker_index]["current_granular_step"] = tracker[
                "last_granular_step_per_broad_step"
            ][tracker["current_broad_step"]]
        else:
            raise Exception(
                f"Tried to decrement tracker #{tracker_ind}: "
                f"{tracker['recipe']} already on step 0."
            )

        return self.trackers

    def get_gt_steps_from_gt_activities(self, video_dset, config_fn):
        """
        Map activity IDs to granular steps and broad steps.
        Assuming one video input.

        Inputs:
        - video_dset: kwcocoDataset for a single video, with "activity_gt"
          ground truth for each video frame as a kwcoco image.
        - broad_steps: steps as derived from a config/tasks/[recipe].yaml file.
          config["labels"] example:
           'labels': [{'id': 0,
               'label': 'background',
               'full_str': 'background',
               'activity_ids': [0]},
              {'id': 1,
               'label': 'water-in-kettle',
               'full_str': 'Measure 12 ounces of cold water and transfer to a kettle.',
               'activity_ids': [8, 1, 2]},
               ...]
        - config_fn = "config/tasks/task_coffee.yaml"
        Note: In the "easy case", every activity_id maps to 'activity_ids' in just one
        "broad step" (just one element of config["labels"]. We check for this case.

        In that "easy case", every activity maps to just one "granular step", and one
        "broad step" too.

        In the "harder case", an "activity_gt" may exist in step 2, and step 6, for
        instance. I'll just incorrectly assume for now that that activity always pertains
        to the earlier step. It's just a plot, and creating a comprehensive set of
        rules to always get the activity-to-broad-step & activity-to-granular-step
        mapping perfect probably isn't worth the effort for now.

        """
        activity_gts = video_dset.images().lookup("activity_gt")
        broad_step_gts = []
        broad_step_gts_no_background = []
        granular_step_gts = []
        granular_step_gts_no_background = []
        current_step = 0

        def sanitize_str(str_: str):
            return str_.lower().strip(" .")

        # TODO: make use of angel_system.data.config_structs instead of
        #       manually loading and accessing by string keys.
        with open(config_fn, "r") as stream:
            config = yaml.safe_load(stream)
        labels = [sanitize_str(l["label"]) for l in config["labels"]]
        broad_steps = config["labels"]
        if broad_steps[0]["id"] == 1:
            config["labels"].insert(
                0,
                {
                    "id": 0,
                    "activity_ids": 0,
                    "label": "background",
                    "full_str": "background",
                },
            )
        granular_step_to_activity_id = self.get_activity_per_granular_step(broad_steps)
        lgspbs = self.get_last_granular_step_per_broad_step(broad_steps)

        def get_broad_step_from_granular_step(lgspbs, granular_step):
            lgspbs = np.array(lgspbs)
            return len(np.nonzero(lgspbs <= granular_step))

        for activity_gt in activity_gts:
            # convert activity id to step id
            granular_step_id = granular_step_to_activity_id.index(activity_gt)
            broad_step_id = get_broad_step_from_granular_step(lgspbs, granular_step_id)

            granular_step_gts.append(granular_step_id)
            broad_step_gts.append(broad_step_id)

            # A version of GT that never jumps back to 0
            if granular_step_id > 0:
                current_granular_step = granular_step_id
                current_broad_step = broad_step_id
            granular_step_gts_no_background.append(current_step)
            broad_step_gts_no_background.append(current_step)
        return (
            granular_step_gts,
            granular_step_gts_no_background,
            broad_step_gts,
            broad_step_gts_no_background,
        )


def plot_positive_GT_conf_distributions(activity_confs, activity_gt):
    """
    plot_TP_conf_distributions:
    For each activity, plot the distribution of confidences when ground
    truth indicates that activity is happening.

    i.e.: for activity x, for frames in which ground truth = x, plot
    the distribution of confidences.

    Inputs:
    activity_confs: frames x class-wise-confidences. Given a kwcoco
        dataset called "coco":
        ```
        activity_confs = torch.asarray(coco.images().lookup("activity_conf"))
        ```
        (49K x 25 for coffee val set.)
    activity_gt: frames x ground truth activity_id.
        Given a kwcoco dataset called "coco":
        ```
        activity_gt = torch.asarray(coco.images().lookup("activity_gt"))
        ```
    """

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Get data together
    true_confs = [
        float(activity_confs[i, truth_ind]) for i, truth_ind in enumerate(activity_gt)
    ]
    data = {"true_conf": true_confs, "gt": activity_gt}
    df = pd.DataFrame(data)

    false_confs = np.array(
        [
            [a for i, a in enumerate(act_conf) if i != gt]
            for act_conf, gt in zip(activity_confs, activity_gt)
        ]
    ).flatten()
    false_gt = np.array(
        [
            [gt for i, a in enumerate(act_conf) if i != gt]
            for act_conf, gt in zip(activity_confs, activity_gt)
        ]
    ).flatten()
    data_opposite = {"true_conf": false_confs, "gt": false_gt}
    df_opposite = pd.DataFrame(data_opposite)

    def plot(df):
        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-0.25, light=0.7)
        g = sns.FacetGrid(df, row="gt", hue="gt", aspect=15, height=0.5, palette=pal)

        # Draw the densities in a few steps
        g.map(
            sns.kdeplot,
            "true_conf",
            bw_adjust=0.5,
            clip_on=False,
            fill=True,
            alpha=1,
            linewidth=1.5,
        )
        g.map(sns.kdeplot, "true_conf", clip_on=False, color="w", lw=2, bw_adjust=0.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(
                0,
                0.2,
                label,
                fontweight="bold",
                color=color,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )

        g.map(label, "true_conf")

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-0.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)

        # save
        plt.savefig("./outputs/plot_positive_GT_conf_distributions.png")
        plt.close()


def bilateralFtr1D(y, sSpatial=5, sIntensity=1):
    """
    The equation of the bilateral filter is

            (       dx ^ 2       )       (         dI ^2        )
    F = exp (- ----------------- ) * exp (- ------------------- )
            (  sigma_spatial ^ 2 )       (  sigma_Intensity ^ 2 )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is a guassian filter!
        dx - The 'geometric' distance between the 'center pixel' and the pixel
         to sample
    dI - The difference between the intensity of the 'center pixel' and
         the pixel to sample
    sigma_spatial and sigma_Intesity are constants. Higher values mean
    that we 'tolerate more' higher value of the distances dx and dI.

    Dependencies: numpy, scipy.ndimage.gaussian_filter1d

    calc gaussian kernel size as: filterSize = (2 * radius) + 1; radius = floor (2 * sigma_spatial)
    y - input data
    """

    # gaussian filter and parameters
    radius = np.floor(2 * sSpatial)
    filterSize = (2 * radius) + 1
    ftrArray = np.zeros(int(filterSize))
    ftrArray[int(radius)] = 1

    # Compute the Gaussian filter part of the Bilateral filter
    gauss = ndi.gaussian_filter1d(ftrArray, sSpatial)

    # 1d data dimensions
    width = y.size

    # 1d resulting data
    ret = np.zeros(width)

    for i in range(width):
        ## To prevent accessing values outside of the array
        # The left part of the lookup area, clamped to the boundary
        xmin = max(i - radius, 1)
        # How many columns were outside the image, on the left?
        dxmin = xmin - (i - radius)

        # The right part of the lookup area, clamped to the boundary
        xmax = min(i + radius, width)
        # How many columns were outside the image, on the right?
        dxmax = (i + radius) - xmax

        # The actual range of the array we will look at
        area = y[int(xmin) : int(xmax)]

        # The center position
        center = y[i]

        # The left expression in the bilateral filter equation
        # We take only the relevant parts of the matrix of the
        # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
        # ignore the parts that are outside the image
        expS = gauss[int((1 + dxmin)) : int((filterSize - dxmax))]

        # The right expression in the bilateral filter equation
        dy = y[int(xmin) : int(xmax)] - y[i]
        dIsquare = dy * dy
        expI = np.exp(-dIsquare / (sIntensity * sIntensity))

        # The bilater filter (weights matrix)
        F = expI * expS

        # Normalized bilateral filter
        Fnormalized = F / sum(F)

        # Multiply the area by the filter
        tempY = y[int(xmin) : int(xmax)] * Fnormalized

        # The resulting pixel is the sum of all the pixels in
        # the area, according to the weights of the filter
        # ret(i,j,R) = sum (tempR(:))
        ret[i] = sum(tempY)

    return ret
