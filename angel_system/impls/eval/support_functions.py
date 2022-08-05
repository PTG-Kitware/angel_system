import dataclasses
from matplotlib import animation
import numpy as np
from pathlib import Path
import re
from typing import Dict
from typing import List
from typing import Tuple
import PIL
import ubelt as ub
from concurrent.futures import ThreadPoolExecutor
import tqdm


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
            fp_in_range.append({
                "time": fp_t,
                "path": img_fp,
            })
    fp_in_range.sort(key=lambda e: e['time'])
    return [e['path'] for e in fp_in_range]


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
        """ Clear variable states with new list instances. """
        cls.slice_index_ranges = []
        cls.slice_time_ranges = []
        cls.slice_preds = []


@dataclasses.dataclass
class SelectedSlice:
    index: int
    animation: animation.FuncAnimation

    @property
    def frame_sequence(self) -> np.ndarray:
        """ matrix of image frames, shape [nFrames x H x W x C] """
        slice_idx_range = GlobalValues.slice_index_ranges[self.index]
        slice_frames = GlobalValues.all_image_mats[slice_idx_range[0]:slice_idx_range[1]]
        return slice_frames

    @property
    def activity_predictions(self):
        return GlobalValues.slice_preds[self.index]

def populate_global_values(images_dir):
    # Sort lexicographically to get into temporal order.
    GlobalValues.all_image_files = sorted(Path(images_dir).iterdir())
    GlobalValues.all_image_mats = []

    #with ThreadPoolExecutor() as pool:
    #    for r in tqdm.tqdm(pool.map(lambda p:np.asarray(PIL.Image.open(p)), GlobalValues.all_image_files),
    #                       total=len(GlobalValues.all_image_files), ncols=100):
    for p in ub.ProgIter(GlobalValues.all_image_files, desc='loading images'):
        r = np.asarray(PIL.Image.open(p))
        GlobalValues.all_image_mats.append(r)
    print("Done!")

    GlobalValues.all_image_mats = np.asarray(GlobalValues.all_image_mats)

    GlobalValues.all_image_times = np.asarray([
        time_from_name(p.name) for p in GlobalValues.all_image_files
    ])

    assert len(GlobalValues.all_image_files) == len(GlobalValues.all_image_mats) == len(GlobalValues.all_image_times)
