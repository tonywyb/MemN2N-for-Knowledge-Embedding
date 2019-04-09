# encoding: utf-8
from __future__ import unicode_literals
import json
import copy


def pre_process(filename="data/lic/train+test.json"):
    with open(filename, "rb") as f:
        dict = json.loads(f.read())
        total_num = len(dict["train"])
        train_num = int(total_num / 2)
        res_lis = list()
        for tri in dict["train"]:
            res_lis.append([])
            res_lis[int(tri)].append([])
            res_lis[int(tri)].append([])
            res_lis[int(tri)].append([])
            res_lis[int(tri)][0] = dict["train"][tri]["knowledge"]
            res_lis[int(tri)][1] = dict["train"][tri]["conversation"][0] + " " + dict["train"][tri]["conversation"][1]
            res_lis[int(tri)][2] = dict["train"][tri]["conversation"][2]
    return res_lis[:train_num], res_lis[train_num:]


if __name__ == "__main__":
    tr, te = pre_process()
