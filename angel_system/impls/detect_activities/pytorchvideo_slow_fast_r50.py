import importlib.util
import logging
from typing import Tuple, Iterable, Dict, Hashable, List, Union, Any
from types import MethodType
import json

import numpy as np

import torch
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

from angel_system.interfaces.detect_activities import DetectActivities

LOG = logging.getLogger(__name__)

class PytorchVideoSlowFastR50(DetectActivities):
    """
    ``DetectActivities`` implementation using the ``pytorchvideo`` library.
    Default model is a `slowfast_r50` pretrained on the Kinetics 400 dataset

    :param use_cuda: Attempt to use a cuda device for inferences. If no
        device is found, CPU is used.
    :param cuda_device: When using CUDA use the device by the given ID. By
        default, this refers to GPU ID 0. This parameter is not used if
        `use_cuda` is false.
    """

    def __init__(
        self,
        use_cuda: bool = False,
        cuda_device: Union[int, str] = "cuda:0",
    ):
        self._use_cuda = use_cuda
        self._cuda_device = cuda_device

        # Set to None for lazy loading later.
        self._model: torch.nn.Module = None  # type: ignore
        self._model_device: torch.device = None  # type: ignore

        # Default parameters from pytorchvideo tutorial:
        # https://pytorchvideo.org/docs/tutorial_torchhub_inference
        self._side_size = 256
        self._mean = [0.45, 0.45, 0.45]
        self._std = [0.225, 0.225, 0.225]
        self._crop_size = 256
        self._num_frames = 32
        self._sampling_rate = 2
        self._frames_per_second = 30
        self._alpha = 4

    def get_model(self) -> "torch.nn.Module":
        """
        Lazy load the torch model in an idempotent manner.
        :raises RuntimeError: Use of CUDA was requested but is not available.
        """
        model = self._model
        if model is None:
            # Note: Line below is a workaround for an intermittent HTTP error
            # when downloading the model from pytorch.
            # See: https://github.com/pytorch/vision/issues/4156
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

            # Load the pretrained model
            model = torch.hub.load("facebookresearch/pytorchvideo",
                                     model="slowfast_r50", pretrained=True)
            model = model.eval()
            model_device = torch.device('cpu')
            if self._use_cuda:
                if torch.cuda.is_available():
                    model_device = torch.device(device=self._cuda_device)
                    model = model.to(device=model_device)
                else:
                    raise RuntimeError(
                        "Use of CUDA requested but not available."
                    )
            self._model = model
            self._model_device = model_device

        return model

    def detect_activities(
        self,
        frame_iter: Iterable[np.ndarray]
    ) -> Dict[str, float]:
        """
        Formats the given iterable of frames into the required input format
        for the SlowFastR50 model and then inputs them to the model for inferencing.
        """
        model = self.get_model()

        # Create tensor with shape CxTxHxW
        video_tensor = {'video': None, 'audio':torch.empty((1))}
        for frame in frame_iter:
            # Convert np.array to tensor
            frame_t = torch.from_numpy(frame)

            # Add extra dimension so shape is now 1xHxWxC
            frame_t = frame_t[None, :]
            frame_t = torch.permute(frame_t, (3, 0, 1, 2))

            try:
                video_tensor["video"] = torch.cat([video_tensor['video'], frame_t],
                                                   dim=1)
            except:
                video_tensor["video"] = frame_t

        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self._num_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(self._mean, self._std),
                    ShortSideScale(
                        size=self._side_size
                    ),
                    CenterCropVideo(self._crop_size),
                    PackPathway()
                ]
            ),
        )

        # Apply a transform to normalize the video input
        video_data = transform(video_tensor)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(self._model_device)[None, ...] for i in inputs]

        # Pass the input clip through the model
        preds = model(inputs)

        # Get the predicted classes
        post_act = torch.nn.Softmax(dim=1)
        preds: torch.Tensor = post_act(preds)[0] # shape: (400)

        # Create the label to prediction confidence map
        prediction_map = {}
        for idx, pred in enumerate(preds):
            prediction_map[KINETICS_400_LABELS[idx]] = pred.item()

        return prediction_map

    def get_config(self) -> dict:
        return {
            "use_cuda": self._use_cuda,
            "cuda_device": self._cuda_device,
        }

    @classmethod
    def is_usable(cls) -> bool:
        # check for optional dependencies
        torch_spec = importlib.util.find_spec('torch')
        torchvision_spec = importlib.util.find_spec('torchvision')
        pytorchvideo_spec = importlib.util.find_spec('pytorchvideo')
        if (
            torch_spec is not None and
            torchvision_spec is not None and
            pytorchvideo_spec is not None
        ):
            return True
        else:
            return False


KINETICS_400_LABELS = (
    'abseiling', 'air drumming', 'answering questions', 'applauding',
    'applying cream', 'archery', 'arm wrestling', 'arranging flowers',
    'assembling computer', 'auctioning', 'baby waking up', 'baking cookies',
    'balloon blowing', 'bandaging', 'barbequing', 'bartending', 'beatboxing',
    'bee keeping', 'belly dancing', 'bench pressing', 'bending back',
    'bending metal', 'biking through snow', 'blasting sand', 'blowing glass',
    'blowing leaves', 'blowing nose', 'blowing out candles', 'bobsledding',
    'bookbinding', 'bouncing on trampoline', 'bowling', 'braiding hair',
    'breading or breadcrumbing', 'breakdancing', 'brush painting', 'brushing hair',
    'brushing teeth', 'building cabinet', 'building shed', 'bungee jumping',
    'busking', 'canoeing or kayaking', 'capoeira', 'carrying baby',
    'cartwheeling', 'carving pumpkin', 'catching fish',
    'catching or throwing baseball', 'catching or throwing frisbee',
    'catching or throwing softball', 'celebrating', 'changing oil', 'changing wheel',
    'checking tires', 'cheerleading', 'chopping wood', 'clapping',
    'clay pottery making', 'clean and jerk', 'cleaning floor', 'cleaning gutters',
    'cleaning pool', 'cleaning shoes', 'cleaning toilet', 'cleaning windows',
    'climbing a rope', 'climbing ladder', 'climbing tree', 'contact juggling',
    'cooking chicken', 'cooking egg', 'cooking on campfire', 'cooking sausages',
    'counting money', 'country line dancing', 'cracking neck', 'crawling baby',
    'crossing river', 'crying', 'curling hair', 'cutting nails', 'cutting pineapple',
    'cutting watermelon', 'dancing ballet', 'dancing charleston',
    'dancing gangnam style', 'dancing macarena', 'deadlifting',
    'decorating the christmas tree', 'digging', 'dining', 'disc golfing',
    'diving cliff', 'dodgeball', 'doing aerobics', 'doing laundry', 'doing nails',
    'drawing', 'dribbling basketball', 'drinking', 'drinking beer', 'drinking shots',
    'driving car', 'driving tractor', 'drop kicking', 'drumming fingers',
    'dunking basketball', 'dying hair', 'eating burger', 'eating cake',
    'eating carrots', 'eating chips', 'eating doughnuts', 'eating hotdog',
    'eating ice cream', 'eating spaghetti', 'eating watermelon', 'egg hunting',
    'exercising arm', 'exercising with an exercise ball', 'extinguishing fire',
    'faceplanting', 'feeding birds', 'feeding fish', 'feeding goats',
    'filling eyebrows', 'finger snapping', 'fixing hair', 'flipping pancake',
    'flying kite', 'folding clothes', 'folding napkins', 'folding paper',
    'front raises', 'frying vegetables', 'garbage collecting', 'gargling',
    'getting a haircut', 'getting a tattoo', 'giving or receiving award',
    'golf chipping', 'golf driving', 'golf putting', 'grinding meat',
    'grooming dog', 'grooming horse', 'gymnastics tumbling', 'hammer throw',
    'headbanging', 'headbutting', 'high jump', 'high kick', 'hitting baseball',
    'hockey stop', 'holding snake', 'hopscotch', 'hoverboarding', 'hugging',
    'hula hooping', 'hurdling', 'hurling (sport)', 'ice climbing', 'ice fishing',
    'ice skating', 'ironing', 'javelin throw', 'jetskiing', 'jogging',
    'juggling balls', 'juggling fire', 'juggling soccer ball', 'jumping into pool',
    'jumpstyle dancing', 'kicking field goal', 'kicking soccer ball',
    'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing',
    'laying bricks', 'long jump', 'lunge', 'making a cake', 'making a sandwich',
    'making bed', 'making jewelry', 'making pizza', 'making snowman',
    'making sushi', 'making tea', 'marching', 'massaging back', 'massaging feet',
    'massaging legs', "massaging person's head", 'milking cow',
    'mopping floor', 'motorcycling', 'moving furniture', 'mowing lawn',
    'news anchoring', 'opening bottle', 'opening present', 'paragliding',
    'parasailing', 'parkour', 'passing American football (in game)',
    'passing American football (not in game)', 'peeling apples', 'peeling potatoes',
    'petting animal (not cat)', 'petting cat', 'picking fruit',
    'planting trees', 'plastering', 'playing accordion', 'playing badminton',
    'playing bagpipes', 'playing basketball', 'playing bass guitar', 'playing cards',
    'playing cello', 'playing chess', 'playing clarinet', 'playing controller',
    'playing cricket', 'playing cymbals', 'playing didgeridoo', 'playing drums',
    'playing flute', 'playing guitar', 'playing harmonica', 'playing harp',
    'playing ice hockey', 'playing keyboard', 'playing kickball', 'playing monopoly',
    'playing organ', 'playing paintball', 'playing piano',
    'playing poker', 'playing recorder', 'playing saxophone',
    'playing squash or racquetball', 'playing tennis', 'playing trombone',
    'playing trumpet', 'playing ukulele', 'playing violin', 'playing volleyball',
    'playing xylophone', 'pole vault', 'presenting weather forecast', 'pull ups',
    'pumping fist', 'pumping gas', 'punching bag', 'punching person (boxing)',
    'push up', 'pushing car', 'pushing cart', 'pushing wheelchair',
    'reading book', 'reading newspaper', 'recording music', 'riding a bike',
    'riding camel', 'riding elephant', 'riding mechanical bull',
    'riding mountain bike', 'riding mule', 'riding or walking with horse', 'riding scooter',
    'riding unicycle', 'ripping paper', 'robot dancing', 'rock climbing',
    'rock scissors paper', 'roller skating', 'running on treadmill', 'sailing',
    'salsa dancing', 'sanding floor', 'scrambling eggs', 'scuba diving',
    'setting table', 'shaking hands', 'shaking head', 'sharpening knives',
    'sharpening pencil', 'shaving head', 'shaving legs', 'shearing sheep',
    'shining shoes', 'shooting basketball', 'shooting goal (soccer)',
    'shot put', 'shoveling snow', 'shredding paper', 'shuffling cards', 'side kick',
    'sign language interpreting',  'singing', 'situp', 'skateboarding', 'ski jumping',
    'skiing (not slalom or crosscountry)', 'skiing crosscountry',
    'skiing slalom', 'skipping rope',  'skydiving', 'slacklining', 'slapping',
    'sled dog racing',  'smoking', 'smoking hookah', 'snatch weight lifting',
    'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting',
    'snowmobiling', 'somersaulting', 'spinning poi', 'spray painting',
    'spraying', 'springboard diving', 'squat', 'sticking tongue out',
    'stomping grapes', 'stretching arm', 'stretching leg', 'strumming guitar',
    'surfing crowd', 'surfing water', 'sweeping floor', 'swimming backstroke',
    'swimming breast stroke', 'swimming butterfly stroke', 'swing dancing',
    'swinging legs', 'swinging on something', 'sword fighting', 'tai chi',
    'taking a shower', 'tango dancing', 'tap dancing', 'tapping guitar', 'tapping pen',
    'tasting beer', 'tasting food', 'testifying', 'texting', 'throwing axe',
    'throwing ball', 'throwing discus', 'tickling', 'tobogganing',
    'tossing coin', 'tossing salad', 'training dog', 'trapezing',
    'trimming or shaving beard', 'trimming trees', 'triple jump', 'tying bow tie',
    'tying knot (not on a tie)', 'tying tie', 'unboxing', 'unloading truck',
    'using computer', 'using remote controller (not gaming)', 'using segway',
    'vault', 'waiting in line', 'walking the dog', 'washing dishes', 'washing feet',
    'washing hair', 'washing hands', 'water skiing', 'water sliding', 'watering plants',
    'waxing back', 'waxing chest', 'waxing eyebrows', 'waxing legs',
    'weaving basket', 'welding', 'whistling', 'windsurfing', 'wrapping present',
    'wrestling', 'writing', 'yawning', 'yoga', 'zumba'
)

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.

    Source: https://pytorchvideo.org/docs/tutorial_torchhub_inference
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // 4
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list
