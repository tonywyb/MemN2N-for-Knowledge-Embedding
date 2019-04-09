# encoding: utf-8
from __future__ import unicode_literals
import json
import copy


def pre_process(filename):
    if filename == "data/lic/train+test.json":
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
                res_lis[int(tri)][0] = get_knowledge_lis(dict["train"][tri]["knowledge"])
                res_lis[int(tri)][1] = dict["train"][tri]["conversation"][0].strip().split(" ")\
                                       + dict["train"][tri]["conversation"][1].strip().split(" ")
                res_lis[int(tri)][2] = dict["train"][tri]["conversation"][2].strip().split(" ")
        return res_lis[:train_num], res_lis[train_num:]
    elif filename == "data/lic/train_part.json":
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            tri_lis = list()
            tre_lis = list()
            for tri in dict["train"]:
                tri_lis.append([])
                tri_lis[int(tri)].append([])
                tri_lis[int(tri)].append([])
                tri_lis[int(tri)].append([])
                tri_lis[int(tri)][0] = get_knowledge_lis(dict["train"][tri]["knowledge"])
                tri_lis[int(tri)][1] = dict["train"][tri]["conversation"][0].strip().split(" ")\
                                       + dict["train"][tri]["conversation"][1].strip().split(" ")
                tri_lis[int(tri)][2] = dict["train"][tri]["conversation"][2].strip().split(" ")
            for tei in dict["test"]:
                tre_lis.append([])
                tre_lis[int(tei)].append([])
                tre_lis[int(tei)].append([])
                tre_lis[int(tei)].append([])
                tre_lis[int(tei)][0] = get_knowledge_lis(dict["test"][tei]["knowledge"])
                tre_lis[int(tei)][1] = dict["test"][tei]["conversation"][0].strip().split(" ")\
                                       + dict["test"][tei]["conversation"][1].strip().split(" ")
                tre_lis[int(tei)][2] = dict["test"][tei]["conversation"][2].strip().split(" ")
        return tri_lis, tre_lis
    else:
        assert False, "lic only has 2 datasets! pick one!"


def get_knowledge_lis(lis):
    res = []
    for kth, k in enumerate(lis):
        res.append([])
        assert type(k) == list
        res[kth] = k[0].strip().split(" ") + k[1].strip().split(" ") + k[2].strip().split(" ")
    return res


if __name__ == "__main__":
    tr, te = pre_process()
