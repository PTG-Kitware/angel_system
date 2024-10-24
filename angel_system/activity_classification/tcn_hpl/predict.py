from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import List
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence

import kwcoco


@dataclass
class ResultElement:
    """
    Single activity classification result element for later inclusion in an
    output COCO file.
    """

    # Class index of the single most confident activity class.
    pred_index: int
    # Vector of confidence values for all activity classes.
    conf_vec: List[float]


class ResultsCollector:
    """
    Utility class to collect TCN-HPL prediction results, encoding the ability
    to output the results to a COCO file.

    This is a beginning to generalizing results coco file aggregation and
    output, based on output from `python-tpl/TCN_HPL/tcn_hpl/models/ptg_module.py`
    in `on_validation_epoch_end` and `on_test_epoch_end` methods.

    TODO: Promote into TCN_HPL repo?
    """

    def __init__(self, output_filepath: Path, id_to_action: Mapping[int, str]):
        self._lock = RLock()  # for thread safety
        self._collection: Dict[int, ResultElement] = {}
        self._dset = dset = kwcoco.CocoDataset()
        dset.fpath = output_filepath.as_posix()
        dset.dataset["info"].append({"activity_labels": id_to_action})
        self._vid: Optional[int] = None

    def set_video(self, video_name: str) -> None:
        """
        Set the video for which we are currently collecting results for.

        :param video_name: Semantic name of the video   .
        """
        with self._lock:
            video_lookup = self._dset.index.name_to_video
            if video_name in video_lookup:
                self._vid = video_lookup[video_name]["id"]
            else:
                self._vid = self._dset.add_video(name=video_name)

    def collect(
        self,
        frame_index: int,
        activity_pred: int,
        activity_conf_vec: Sequence[float],
        name: Optional[str] = None,
        file_name: Optional[str] = None,
        activity_gt: Optional[int] = None,
    ) -> None:
        """
        See `CocoDataset.add_image` for more details.
        """
        with self._lock:
            if self._vid is None:
                raise RuntimeError(
                    "No video set before results collection. See `set_video` method."
                )
            packet = dict(
                video_id=self._vid,
                frame_index=frame_index,
                activity_pred=activity_pred,
                activity_conf=list(activity_conf_vec),
            )
            if name is not None:
                packet["name"] = name
            if file_name is not None:
                packet["file_name"] = file_name
            if activity_gt is not None:
                packet["activity_gt"] = activity_gt
            self._dset.add_image(**packet)

    def write_file(self):
        """
        Write COCO file to the set output path.
        """
        with self._lock:
            dset = self._dset
            dset.dump(dset.fpath, newlines=True)
