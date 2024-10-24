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

    def add_image(
        self,
        frame_index: int,
        name: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> int:
        """
        Add an image to the dataset. Returns the global image id.
        If the image was already added (by name or file name), returns -1.
        """
        with self._lock:
            if self._vid is None:
                raise RuntimeError(
                    "No video set before results collection. See `set_video` method."
                )

            # get the global id for the image from the frame number
            # add the image
            img = dict(
                video_id=self._vid,
                frame_index=frame_index,
            )
            if name is not None:
                img["name"] = name
            if file_name is not None:
                img["file_name"] = file_name
            # save the gid from the image to link to the annot
            try:
                gid = self._dset.add_image(**img)
            except Exception:
                return -1  # image already exists

            return gid

    def collect(
        self,
        gid: int,
        activity_pred: int,
        activity_conf_vec: Sequence[float],
    ) -> None:
        """
        See `CocoDataset.add_image` for more details.

        :param gid: Global image id.
        :param activity_pred: Predicted activity class index.
        :param activity_conf_vec: Confidence vector for all activity classes.
        """
        with self._lock:
            if self._vid is None:
                raise RuntimeError(
                    "No video set before results collection. See `set_video` method."
                )

            # add the annotation
            self._dset.add_annotation(
                image_id=gid,
                category_id=activity_pred,
                score=activity_conf_vec[activity_pred],
                prob=list(activity_conf_vec),
            )

    def write_file(self):
        """
        Write COCO file to the set output path.
        """
        with self._lock:
            dset = self._dset
            dset.dump(dset.fpath, newlines=True)
