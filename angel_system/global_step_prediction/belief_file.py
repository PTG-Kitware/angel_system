"""
This class is meant to create a CSV file in the format that the BBN system expects.
"""

# Max number of seconds in the first time difference between when
# the node started and when the first update came in.
# if larger, a correction will be made to make up for start up time
START_UP_TIME_MAX = 0.6

class BState:
    """
    Enum for the state of the belief
    """
    def __init__(self):
        self.current = "current"
        self.unobserved = "unobserved"
        self.done = "done"


class BeliefFile:
    def __init__(self, filename: str, skill: str, labels: list, start_time: int):
        self.filename = filename
        self.skill = skill
        self.labels = labels
        self.running_state = {}  # keeps track of the steps
        # initialize the running states
        for label in labels:
            self.running_state[label] = BState().unobserved
        # set the first step to current
        # NOTE: the example files given had this set to current
        # from the very beginnning - an assumption we are making here, too
        self.running_state[1.0] = BState().current
        # this will be used to calculate the current time in the video
        self.start_time = start_time

        # initialize the file - in case we need to overwrite it
        with open(self.filename, 'w') as f:
            f.write("")

        # flag for handling how long it takes to start up the video
        self.first_time_diff = True

    def _add_row_to_file(self, row: str) -> None:
        # append the row to the file
        with open(self.filename, 'a') as f:
            f.write(row)

    def _add_rows(self, conf_array: list, ctime: float) -> None:
        """
        Add multiple rows to the file based on the labels
        """
        # <skill>, <step_num>, <state>, <confidence>, <timestep>
        row = self.skill

        # add the rows
        for step in self.labels:
            _row = row + f",{step},{self.running_state[step]},"
            _row = _row + f"{conf_array[int(step)]},{ctime}\n"  # _row = _row + f"{conf_array[int(step)]:0.8f},{ctime:0.8f}\n"
            self._add_row_to_file(_row)

    def final_step_done(self) -> None:
        """
        This method is called when the final step is done.
        """
        # set the final step
        self.running_state[self.labels[-1]] = BState().done

    def update_values(self, current_step: float, conf_array: list, current_time: int) -> None:
        """
        When you provide an update, this method will update internal state
        and trigger a write to the file.
        """
        curr_time = float(current_time - self.start_time) * 1e-9  # get seconds from nano

        # correction of the starting time if we notice that the first
        # time difference is too large
        if self.first_time_diff and curr_time > START_UP_TIME_MAX:
            self.first_time_diff = False
            self.start_time = current_time  # save this for the next update
            # assume 0 for now
            curr_time = 0.0

        # check the states and see if they changed
        if current_step > 0 and self.running_state[current_step] != BState().current:
            # set the current step
            self.running_state[current_step] = BState().current

            # see if the previous state was current - that means we change it to done
            prev_step = current_step - 1.0
            if prev_step > 0 and self.running_state[prev_step] == BState().current:
                self.running_state[prev_step] = BState().done

        # write the rows to the file
        self._add_rows(conf_array, curr_time)
