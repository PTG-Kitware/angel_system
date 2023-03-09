import os
import numpy as np
import pandas as pd
import random
# from shared.niudt.detectron2.detectron2.data.datasets import MC50_CATEGORIES

def assign_color(cate):
    color_list = []
    for i in cate:
        _color = []
        r = random.randint(0, 255)/255
        g = random.randint(0, 255)/255
        b = random.randint(0, 255)/255
        _color.append(i)
        _color.append([r, g, b])
        color_list.append(_color)
    return color_list





if __name__ == '__main__':

    root = '/shared/niudt/detectron2/detectron2/data/datasets/MC_FT_CATE.csv'
    CATE = pd.read_csv(root).values
    CATE = CATE[:, 1]
    color_list = assign_color(CATE)
    color_list_pd = pd.DataFrame(color_list)
    print(color_list)
    print(len(color_list))
    color_list_pd.to_csv('/shared/niudt/detectron2/detectron2/utils/MC_COLOR.csv')

    # # test
    # root = '/shared/niudt/detectron2/detectron2/utils/MC_COLOR.csv'
    # data = pd.read_csv(root).values
    # color_mapping = {}
    # for color in data:
    #     color_value = color[2].split(',')
    #     s = 1
    #     r = float(color_value[0][1:])
    #     g = float(color_value[1][1:])
    #     b = float(color_value[2][1:-1])
    #     color_mapping[color[1]] = [r, g, b]
    # print(color_mapping)
