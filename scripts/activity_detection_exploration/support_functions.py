import dataclasses
from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea, AnchoredText
import numpy as np
from pathlib import Path
import re
from typing import Dict
from typing import List
from typing import Tuple


def sanitize_str(txt):
    txt = txt.lower()

    # remove (step 1)
    try:
        txt = txt[: txt.index("(") - 1]
    except ValueError:
        pass

    if txt[-1] == ".":
        txt = txt[:-1]

    return txt

RE_FILENAME_TIME = re.compile(r"frame_\d+_(\d+_\d+).\w+")
def time_from_name(fname):
    """
    Extract the float timestamp from the filename.
    """
    return float(RE_FILENAME_TIME.match(fname).groups()[0].replace("_", "."))


def frames_for_range(start, end):
    """
    Return frame files that occur in the [start, end) range.
    """
    print(f"Range: {start} - {end}")
    fp_in_range = []
    for img_fp in IMAGES_DIR_PATH.iterdir():
        fp_t = time_from_name(img_fp.name)
        if start <= fp_t < end:
            fp_in_range.append(
                {
                    "time": fp_t,
                    "path": img_fp,
                }
            )
    fp_in_range.sort(key=lambda e: e["time"])
    return [e["path"] for e in fp_in_range]


@dataclasses.dataclass
class SliceResult:
    index_range: Tuple[int, int]
    time_range: Tuple[float, float]
    preds: Dict[str, float]


class GlobalValues:
    """
    Container of global prediction result attributes.
    Effectively a singleton with the class-level attributes and functionality.
    """

    # Sequence of file paths in temporal order of our data-set.
    all_image_files: List[Path] = []

    # Array of float timestamps for each image in our data-set.
    all_image_times: np.ndarray = None

    # Matrix of all images as numpy matrices
    all_image_mats: np.ndarray = None

    # The [start, end) frame index ranges per slice
    slice_index_ranges: List[Tuple[int, int]] = []

    # The [start, end) frame time pairs
    slice_time_ranges: List[Tuple[float, float]] = []

    # Prediction results per slice
    slice_preds: List[Dict[str, float]] = []

    @classmethod
    def clear_slice_values(cls):
        """Clear variable states with new list instances."""
        cls.slice_index_ranges = []
        cls.slice_time_ranges = []
        cls.slice_preds = []


@dataclasses.dataclass
class SelectedSlice:
    index: int
    animation: animation.FuncAnimation

    @property
    def frame_sequence(self) -> np.ndarray:
        """matrix of image frames, shape [nFrames x H x W x C]"""
        slice_idx_range = GlobalValues.slice_index_ranges[self.index]
        slice_frames = GlobalValues.all_image_mats[
            slice_idx_range[0] : slice_idx_range[1]
        ]
        return slice_frames

    @property
    def activity_predictions(self):
        return GlobalValues.slice_preds[self.index]


def plot_activity_confidence(
    label, gt_ranges, custom_range=None, custom_range_color="red"
):
    """
    Plot activity confidences, with hover-over showing the sequence-middle frame.

    Clicks modify attributes in the object that is output with respect to what was just clicked.
    This information may be used, for example, to animate the specific window frame sequence for
    the result point that was clicked.

    :param label: String label of the activity class predictions to render.
    :param gt_ranges: A sequence of tuples indicating the starting and ending time of ground-truth
                      time ranges the label activity occurred in the image sequence.
    :param custom_range: Optional tuple indicating the starting and ending times of an additional
                         range to highlight in addition to the `gt_ranges`.
    :param custom_range_color: The color of the additional range to be drawn. If not set, we will
                               use "red".

    Learned from: http://www.andrewjanowczyk.com/image-popups-on-mouse-over-in-jupyter-notebooks/
    """
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.set_title(f'Window Confidence over time for "{label}"')
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(
        GlobalValues.slice_time_ranges[0][0] - 1,
        GlobalValues.slice_time_ranges[-1][1] + 1,
    )
    # plt.yscale("log")

    # Bar plt to show bars where the "true" time ranges are for the activity.
    xs_bars = [p[0] for p in gt_ranges]
    ys_gt_regions = [1 for _ in gt_ranges]
    bar_widths = [(p[1] - p[0]) for p in gt_ranges]
    ax.bar(xs_bars, ys_gt_regions, width=bar_widths, align="edge", color="lightgreen")

    if custom_range:
        assert len(custom_range) == 2, "Assuming only two float values for custom range"
        xs_bars2 = [custom_range[0]]
        ys_height = [1.025]  # [0.1]
        bar_widths2 = [custom_range[1] - custom_range[0]]
        ys_bottom = [0]  # [1.01]
        color = custom_range_color
        # TODO: Make this something that is added be clicking?
        ax.bar(
            xs_bars2,
            ys_height,
            width=bar_widths2,
            bottom=ys_bottom,
            align="edge",
            color=color,
            alpha=0.5,
        )

    # Line plot to show detector confidence for a window slice.
    # Plotted point at the median frame-time of the window predicted over.
    xs_slice_median_time = [
        GlobalValues.all_image_times[int(np.average(idx_rng))]
        for idx_rng in GlobalValues.slice_index_ranges
    ]
    ys_pred_conf = [one_pred[label] for one_pred in GlobalValues.slice_preds]
    err_bar_widths = [
        (t_range[1] - t_range[0]) / 2.0 for t_range in GlobalValues.slice_time_ranges
    ]
    ax.plot
    errorbar = ax.errorbar(
        xs_slice_median_time,
        ys_pred_conf,
        # xerr=err_bar_widths,
        linewidth=1,
        # elinewidth=1,
        fmt=".b-",
    )

    # Create an offset image and copy an example image into it to initialize memory.
    oi = OffsetImage(np.empty_like(GlobalValues.all_image_mats[0]), 0.25)
    ta = TextArea("")
    ab_xybox = (200.0, 100.0)
    ab = AnnotationBbox(
        oi,
        (0, 0),
        xybox=ab_xybox,
        xycoords="data",
        boxcoords="offset points",
        pad=0.3,
        arrowprops=dict(arrowstyle="->"),
    )
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def hover(event):
        line = errorbar.lines[0]
        if line.contains(event)[0]:
            # find out the index within the array from the event
            ind = line.contains(event)[1]["ind"]
            slice_idx = ind[0]  # not "really" closest, mpl func has a TODO on that
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
            hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (ab_xybox[0] * ws, ab_xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (xs_slice_median_time[slice_idx], ys_pred_conf[slice_idx])
            # set the image corresponding to that point
            img_idx = int(np.average(GlobalValues.slice_index_ranges[slice_idx]))
            # if the dataset is too large to load into memory, can instead replace this command with a
            # realtime load
            oi.set_data(GlobalValues.all_image_mats[img_idx])
        else:
            ab.set_visible(False)
        fig.canvas.draw_idle()

    selected_slice = SelectedSlice(None, None)

    def mouse_press(event):
        line = errorbar.lines[0]
        if line.contains(event)[0]:
            # press_event_output.append(event)
            # find out the index within the array from the event
            ind = line.contains(event)[1]["ind"]
            slice_idx = ind[0]  # not "really" closest, mpl func has a TODO on that
            selected_slice.index = slice_idx
            selected_slice.animation = animate_frame_sequence(selected_slice)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    fig.canvas.mpl_connect("button_press_event", mouse_press)
    plt.show()
    return selected_slice


def animate_frame_sequence(slice_selected: SelectedSlice):
    """
    Create and show animation of slice frames.

    Learned from: https://www.numfys.net/howto/animations/
    Reduction of imshow whitespace: https://stackoverflow.com/questions/37809697/remove-white-border-when-using-subplot-and-imshow-in-python-matplotlib
    """
    slice_idx_range = GlobalValues.slice_index_ranges[slice_selected.index]
    slice_frames = slice_selected.frame_sequence
    # Determine animation interval (in milliseconds) by averaging the delta between
    # frames in this slice.
    slice_frame_times = GlobalValues.all_image_times[
        slice_idx_range[0] : slice_idx_range[1]
    ]
    time_diffs = np.diff(slice_frame_times)
    interval_avg = np.round(
        1000 * np.average(list(filter(None, time_diffs))),
    )

    frame_size = slice_frames[0].shape
    fig = plt.figure(dpi=frame_size[0])
    fig.set_size_inches(1.0 * frame_size[1] / frame_size[0], 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    # TODO: Try using an existing imshow to animate into?
    imshow = ax.imshow(slice_frames[0])

    def animate(mat: np.ndarray):
        return imshow.set_data(mat)

    anim = animation.FuncAnimation(
        fig,
        animate,
        # init_func=init,
        frames=slice_frames,
        interval=interval_avg,
        # blit=True,
        cache_frame_data=False,
    )

    plt.close(anim._fig)

    # HTML(anim.to_html5_video())
    return anim
