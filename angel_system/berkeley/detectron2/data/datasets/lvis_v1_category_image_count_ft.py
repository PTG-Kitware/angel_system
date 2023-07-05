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

LVIS_CATEGORY_IMAGE_COUNT = LVIS_CATEGORIES = []
