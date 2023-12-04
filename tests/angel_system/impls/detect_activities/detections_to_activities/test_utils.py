import numpy as np

from tcn_hpl.data.components.augmentations import NormalizePixelPts

from angel_system.activity_classification.utils import (
    obj_det2d_set_to_feature_by_method,
)


class TestUtil_obj_det2d_set_to_feature_by_method_v5:
    """
    Tests for calling `obj_det2d_set_to_feature_by_method` in "v5" mode,
    meaning with `use_activation`, `use_hand_dist` and `use_intersection` set
    to True.
    """

    def test_hand_distance_and_norm(self):
        """
        Testing that object distance to hand is output correctly.
        """
        norm = NormalizePixelPts(1280, 720, 124, 5)

        test_input = dict(
            label_vec=[
                "coffee bag",
                "microwave (open)",
                "box of raisins",
                "hand (left)",
                "hand (left)",
            ],
            xs=[0.0, 184.00002, 222.99998, 361.0, 782.0],
            ys=[23.999998, 0.0, 584.0, 664.0, 586.0],
            ws=[248.0, 856.0, 221.00002, 177.0, 316.0],
            hs=[457.0, 714.0, 136.0, 56.0, 132.0],
            label_confidences=[
                0.93310546875,
                0.97998046875,
                0.3408203125,
                0.451171875,
                0.93310546875,
            ],
            label_to_ind={
                "banana": 58,
                "banana (peeled)": 59,
                "bowl": 80,
                "bowl + oats": 89,
                "bowl + oats + water": 91,
                "bowl + oats + water + raisins": 92,
                "bowl + water": 90,
                "box of dental floss (closed)": 111,
                "box of dental floss (open)": 110,
                "box of raisins": 86,
                "box of toothpicks (open)": 107,
                "butter knife": 57,
                "butter knife + jelly": 114,
                "butter knife + nut butter": 113,
                "butter knife + nutella": 69,
                "cinnamon (closed)": 62,
                "cinnamon (open)": 61,
                "coffee + mug": 19,
                "coffee bag": 20,
                "coffee beans + container": 21,
                "coffee beans + container + scale": 22,
                "coffee grounds + paper filter (quarter - open) + dripper": 23,
                "coffee grounds + paper filter (quarter - open) + dripper + mug": 24,
                "container": 11,
                "container + box of toothpicks (open)": 108,
                "container + raisins": 88,
                "container + scale": 25,
                "container of oats (closed)": 85,
                "container of oats (open)": 84,
                "container of oats lid": 83,
                "cutting board": 56,
                "dental floss string": 112,
                "dripper": 26,
                "dripper + mug": 27,
                "folded tortilla (semi)": 75,
                "grinder (closed)": 30,
                "grinder (open)": 29,
                "grinder lid": 28,
                "hand (left)": 0,
                "hand (right)": 1,
                "jar of honey (closed)": 49,
                "jar of honey (open)": 48,
                "jelly jar (closed)": 106,
                "jelly jar (open)": 105,
                "jelly jar lid": 104,
                "kettle (closed)": 9,
                "kettle (open)": 8,
                "kettle lid": 7,
                "kettle switch": 13,
                "measuring cup": 10,
                "microwave (closed)": 82,
                "microwave (open)": 81,
                "mug": 12,
                "mug + tea": 55,
                "mug + tea bag": 52,
                "mug + tea bag + water": 54,
                "mug + water": 53,
                "nut butter jar (closed)": 103,
                "nut butter jar (open)": 102,
                "nut butter jar lid": 101,
                "nutella jar (closed)": 68,
                "nutella jar (open)": 67,
                "nutella jar lid": 66,
                "oatmeal": 93,
                "oatmeal + cinnamon": 95,
                "oatmeal + cinnamon + honey": 100,
                "oatmeal + honey": 96,
                "oatmeal + sliced banana": 94,
                "oatmeal + sliced banana + cinnamon": 97,
                "oatmeal + sliced banana + cinnamon + honey": 99,
                "oatmeal + sliced banana + honey": 98,
                "paper filter": 31,
                "paper filter (quarter - open) + dripper": 37,
                "paper filter (quarter - open) + dripper + mug": 38,
                "paper filter (quarter)": 33,
                "paper filter (quarter) + dripper": 35,
                "paper filter (quarter) + dripper + mug": 36,
                "paper filter (semi)": 32,
                "paper filter bag": 39,
                "paper towel": 2,
                "paper towel sheet": 3,
                "pinwheel + toothpick": 120,
                "plate (empty)": 65,
                "plate + pinwheel(s) + toothpick": 121,
                "plate + tortilla wedge(s)": 77,
                "raisins": 87,
                "scale (off)": 42,
                "scale (on)": 41,
                "sliced banana": 60,
                "spoon": 47,
                "stack of paper filters": 40,
                "tablespoon": 78,
                "tablespoon + oats": 79,
                "tea bag (dry)": 50,
                "tea bag (wet)": 51,
                "thermometer (closed)": 15,
                "thermometer (open)": 14,
                "timer (off)": 17,
                "timer (on)": 16,
                "toothpick": 109,
                "tortilla": 63,
                "tortilla (partially rolled)": 122,
                "tortilla (rolled)": 118,
                "tortilla (rolled) + toothpick(s)": 119,
                "tortilla + cinnamon": 71,
                "tortilla + jelly": 116,
                "tortilla + nut butter": 115,
                "tortilla + nut butter + jelly": 117,
                "tortilla + nutella": 70,
                "tortilla + nutella + cinnamon": 73,
                "tortilla + nutella + sliced banana": 72,
                "tortilla + nutella + sliced banana + cinnamon": 74,
                "tortilla bag": 64,
                "tortilla end": 123,
                "tortilla wedge": 76,
                "trash can": 18,
                "used paper filter": 43,
                "used paper filter (quarter - open) + dripper": 44,
                "used paper filter (quarter - open) + dripper + mug": 45,
                "used paper filter (quarter) + dripper": 34,
                "used paper filter (quarter) + dripper + mug": 46,
                "water jug (closed)": 6,
                "water jug (open)": 5,
                "water jug lid": 4,
            },
            # v5 variation
            use_activation=True,
            use_hand_dist=True,
            use_intersection=True,
        )

        feature_vec = obj_det2d_set_to_feature_by_method(**test_input)
        feature_vec_normalized = feature_vec.copy()
        norm(feature_vec_normalized[None, ...])

        # Where normalization happened, nothing should be out of the [0, 1]
        # range.
        diff_indices = np.nonzero(feature_vec != feature_vec_normalized)[0]
        assert diff_indices.size == 5, "Should have 5 differences"
        assert feature_vec_normalized[diff_indices].min() >= 0.0
        assert feature_vec_normalized[diff_indices].max() <= 1.0
