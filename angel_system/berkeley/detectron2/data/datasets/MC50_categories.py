# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v1_train.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
# del x["name"]
# del x["instance_count"]
# del x["def"]
# del x["synonyms"]
# del x["frequency"]
# del x["synset"]
# LVIS_CATEGORY_IMAGE_COUNT = repr(c) + "  # noqa"
# with open("/tmp/lvis_category_image_count.py", "wt") as f:
#     f.write(f"LVIS_CATEGORY_IMAGE_COUNT = {LVIS_CATEGORY_IMAGE_COUNT}")
# Then paste the contents of that file below
# fmt: off

MC50_CATEGORIES = [{'id': 1, 'name': 'coffee + mug', 'instances_count': 85, 'def': '', 'synonyms': [], 'image_count': 85, 'frequency': '', 'synset': ''}, {'id': 2, 'name': 'coffee bag', 'instances_count': 545, 'def': '', 'synonyms': [], 'image_count': 545, 'frequency': '', 'synset': ''}, {'id': 3, 'name': 'coffee beans + container', 'instances_count': 36, 'def': '', 'synonyms': [], 'image_count': 36, 'frequency': '', 'synset': ''}, {'id': 4, 'name': 'coffee beans + container + scale', 'instances_count': 38, 'def': '', 'synonyms': [], 'image_count': 38, 'frequency': '', 'synset': ''}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone', 'instances_count': 40, 'def': '', 'synonyms': [], 'image_count': 40, 'frequency': '', 'synset': ''}, {'id': 6, 'name': 'coffee grounds + paper filter + filter cone + mug', 'instances_count': 102, 'def': '', 'synonyms': [], 'image_count': 102, 'frequency': '', 'synset': ''}, {'id': 7, 'name': 'container', 'instances_count': 653, 'def': '', 'synonyms': [], 'image_count': 653, 'frequency': '', 'synset': ''}, {'id': 8, 'name': 'container + scale', 'instances_count': 121, 'def': '', 'synonyms': [], 'image_count': 121, 'frequency': '', 'synset': ''}, {'id': 9, 'name': 'filter cone', 'instances_count': 161, 'def': '', 'synonyms': [], 'image_count': 161, 'frequency': '', 'synset': ''}, {'id': 10, 'name': 'filter cone + mug', 'instances_count': 138, 'def': '', 'synonyms': [], 'image_count': 138, 'frequency': '', 'synset': ''}, {'id': 11, 'name': 'grinder (close)', 'instances_count': 745, 'def': '', 'synonyms': [], 'image_count': 745, 'frequency': '', 'synset': ''}, {'id': 12, 'name': 'grinder (open)', 'instances_count': 209, 'def': '', 'synonyms': [], 'image_count': 209, 'frequency': '', 'synset': ''}, {'id': 13, 'name': 'hand (left)', 'instances_count': 911, 'def': '', 'synonyms': [], 'image_count': 911, 'frequency': '', 'synset': ''}, {'id': 14, 'name': 'hand (right)', 'instances_count': 1026, 'def': '', 'synonyms': [], 'image_count': 1026, 'frequency': '', 'synset': ''}, {'id': 15, 'name': 'kettle', 'instances_count': 812, 'def': '', 'synonyms': [], 'image_count': 812, 'frequency': '', 'synset': ''}, {'id': 16, 'name': 'kettle (open)', 'instances_count': 97, 'def': '', 'synonyms': [], 'image_count': 97, 'frequency': '', 'synset': ''}, {'id': 17, 'name': 'lid (grinder)', 'instances_count': 127, 'def': '', 'synonyms': [], 'image_count': 127, 'frequency': '', 'synset': ''}, {'id': 18, 'name': 'lid (kettle)', 'instances_count': 39, 'def': '', 'synonyms': [], 'image_count': 39, 'frequency': '', 'synset': ''}, {'id': 19, 'name': 'measuring cup', 'instances_count': 1073, 'def': '', 'synonyms': [], 'image_count': 1073, 'frequency': '', 'synset': ''}, {'id': 20, 'name': 'mug', 'instances_count': 162, 'def': '', 'synonyms': [], 'image_count': 162, 'frequency': '', 'synset': ''}, {'id': 21, 'name': 'paper filter', 'instances_count': 38, 'def': '', 'synonyms': [], 'image_count': 38, 'frequency': '', 'synset': ''}, {'id': 22, 'name': 'paper filter (quarter)', 'instances_count': 102, 'def': '', 'synonyms': [], 'image_count': 102, 'frequency': '', 'synset': ''}, {'id': 23, 'name': 'paper filter (semi)', 'instances_count': 43, 'def': '', 'synonyms': [], 'image_count': 43, 'frequency': '', 'synset': ''}, {'id': 24, 'name': 'paper filter + filter cone', 'instances_count': 39, 'def': '', 'synonyms': [], 'image_count': 39, 'frequency': '', 'synset': ''}, {'id': 25, 'name': 'paper filter + filter cone + mug', 'instances_count': 251, 'def': '', 'synonyms': [], 'image_count': 251, 'frequency': '', 'synset': ''}, {'id': 26, 'name': 'paper filter bag', 'instances_count': 655, 'def': '', 'synonyms': [], 'image_count': 655, 'frequency': '', 'synset': ''}, {'id': 27, 'name': 'paper towel', 'instances_count': 412, 'def': '', 'synonyms': [], 'image_count': 412, 'frequency': '', 'synset': ''}, {'id': 28, 'name': 'scale (off)', 'instances_count': 688, 'def': '', 'synonyms': [], 'image_count': 688, 'frequency': '', 'synset': ''}, {'id': 29, 'name': 'scale (on)', 'instances_count': 80, 'def': '', 'synonyms': [], 'image_count': 80, 'frequency': '', 'synset': ''}, {'id': 30, 'name': 'switch', 'instances_count': 513, 'def': '', 'synonyms': [], 'image_count': 513, 'frequency': '', 'synset': ''}, {'id': 31, 'name': 'thermometer (close)', 'instances_count': 453, 'def': '', 'synonyms': [], 'image_count': 453, 'frequency': '', 'synset': ''}, {'id': 32, 'name': 'thermometer (open)', 'instances_count': 147, 'def': '', 'synonyms': [], 'image_count': 147, 'frequency': '', 'synset': ''}, {'id': 33, 'name': 'timer', 'instances_count': 624, 'def': '', 'synonyms': [], 'image_count': 624, 'frequency': '', 'synset': ''}, {'id': 34, 'name': 'timer (20)', 'instances_count': 29, 'def': '', 'synonyms': [], 'image_count': 29, 'frequency': '', 'synset': ''}, {'id': 35, 'name': 'timer (30)', 'instances_count': 28, 'def': '', 'synonyms': [], 'image_count': 28, 'frequency': '', 'synset': ''}, {'id': 36, 'name': 'timer (else)', 'instances_count': 111, 'def': '', 'synonyms': [], 'image_count': 111, 'frequency': '', 'synset': ''}, {'id': 37, 'name': 'trash can', 'instances_count': 29, 'def': '', 'synonyms': [], 'image_count': 29, 'frequency': '', 'synset': ''}, {'id': 38, 'name': 'used paper filter', 'instances_count': 60, 'def': '', 'synonyms': [], 'image_count': 60, 'frequency': '', 'synset': ''}, {'id': 39, 'name': 'used paper filter + filter cone', 'instances_count': 69, 'def': '', 'synonyms': [], 'image_count': 69, 'frequency': '', 'synset': ''}, {'id': 40, 'name': 'used paper filter + filter cone + mug', 'instances_count': 8, 'def': '', 'synonyms': [], 'image_count': 8, 'frequency': '', 'synset': ''}, {'id': 41, 'name': 'water', 'instances_count': 603, 'def': '', 'synonyms': [], 'image_count': 603, 'frequency': '', 'synset': ''}, {'id': 42, 'name': 'water + coffee grounds + paper filter + filter cone + mug', 'instances_count': 252, 'def': '', 'synonyms': [], 'image_count': 252, 'frequency': '', 'synset': ''}]

# states_pair = []
# # find the key
# keys = []
# for _cate in MC50_CATEGORIES:
#     name = _cate['name']
#     if '+' in name or '(' in name:
#         continue
#     keys.append(name)
# print(keys)
# _list = []
# for key in keys:
#     _temp = []
#     for i, _cate in enumerate(MC50_CATEGORIES):
#         name = _cate['name']
#         if key in name:
#             _temp.append(name)
#     if len(_temp) != 1:
#         # print(_temp)
#         _list.append(_temp)
# print(_list)


