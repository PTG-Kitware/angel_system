import importlib.util
import logging
from typing import Tuple, Iterable, Dict, Hashable, List, Union, Any
from types import MethodType
import json
import numpy as np
from joblib import load
import opensmile

class ComParEModel(SpeechEmotionModel):
    """
    Model to detect emotions from a set of input
    2D ``numpy.ndarray`` type arrays.
    """
    def __init__(self, model, scaler):
        """
        model: a serialized model. __init__ should implement loading model either using pickle or torch.
        e.g. model = "mlp_esd_compare.joblib"
        scaler: a serialized StandardScaler, e.g. "esd_compare_scaler.joblib"
        """
        self.model = load(model) 
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.scaler = load(scaler)

    def classify(
      self,
      X: Iterable[np.ndarray],
      sampling_rate: int
    ) -> int:
        """
        Classify in the given modality representation.
        :param X: ndarray of the input audio signal
        :param sample_rate: the sampling rate of X
        :return: Prediction output corresponding to class number
        """
        data = self.smile.process_signal(X, sampling_rate)
        data = data.loc[:, ~data.columns.isin(['label', 'file', 'start', 'end'])]
        data = self.scaler.transform(data)
        return self.model.predict(X)


    def __call__(
      self,
      X: Iterable[np.ndarray],
      sampling_rate: int
    ) -> int:
        """
        Calls `classify_emotion() with the given iterable modality representation.`
        """
        return self.classify(X)