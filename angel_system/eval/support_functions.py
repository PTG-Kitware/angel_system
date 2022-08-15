import dataclasses
from matplotlib import animation
import numpy as np
from pathlib import Path
import re
from typing import Dict
from typing import List
from typing import Tuple
import PIL
from concurrent.futures import ThreadPoolExecutor
import tqdm


RE_FILENAME_TIME = re.compile(r"frame_\d+_(\d+_\d+).\w+")


def time_from_name(fname):
    """
    Extract the float timestamp from the filename.
    """
    time = RE_FILENAME_TIME.match(fname).groups()[0].split('_')
    return float(time[0]) + (float(time[1]) * 1e-9)

def frames_for_range(start, end):
    """
    Return frame files that occur in the [start, end) range.
    """
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

