# encoding: utf-8
from __future__ import unicode_literals
import json
import pickle as pkl
from itertools import chain
import os


def pre_process(filename="data/lic/train+test.json", ground_filter=True):
    if filename == "data/lic/train+test.json":
        with open("data/lic/knowledge.pkl", "rb") as f:
            kg_word = pkl.load(f)
        # with codecs.open("data/lic/filter_200.txt", "r", "utf-8") as f:
        #     nil_word = f.read().strip().split(" ")
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            div_num = int(len(dict["train"]) / 10)
            res_lis = list()
            res_cnt = 0
            for tri in dict["train"]:
                kg = get_knowledge_lis(dict["train"][tri]["knowledge"])
                cur_len = 1 + (len(dict["train"][tri]["conversation"]) - 3) // 2
                for cnt in range(cur_len):
                    tmp1 = kg
                    if cnt == 0:
                        tmp2 = dict["train"][tri]["conversation"][0].strip().split(" ")\
                                               + dict["train"][tri]["conversation"][1].strip().split(" ")
                        if ground_filter:
                            tmp3 = list(
                                filter(lambda t: t in kg_word,
                                       dict["train"][tri]["conversation"][2].strip().split(" ")))
                        else:
                            tmp3 = dict["train"][tri]["conversation"][2].strip().split(" ")
                    else:
                        tmp2 = dict["train"][tri]["conversation"][2 * cnt + 1].strip().split(" ")
                        if ground_filter:
                            tmp3 = list(
                                filter(lambda t: t in kg_word,
                                       dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")))
                        else:
                            tmp3 = dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")
                    if len(tmp3) == 0:
                        continue
                    res_lis.append([])
                    for t in range(3):
                        res_lis[res_cnt].append([])
                    res_lis[res_cnt][0] = tmp1
                    res_lis[res_cnt][1] = tmp2
                    res_lis[res_cnt][2] = tmp3
                    res_cnt += 1
        return list(res_lis[:8 * div_num]), list(res_lis[8 * div_num:9 * div_num]), list(res_lis[9 * div_num:])
    elif filename == "data/lic/train_part.json":
        with open("data/lic/toy_knowledge.pkl", "rb") as f:
            kg_word = pkl.load(f)
        # with codecs.open("data/lic/toy_filter_200.txt", "r", "utf-8") as f:
        #     toy_nil_word = f.read().strip().split(" ")
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            tr_lis = list()
            te_lis = list()
            tr_cnt = 0
            te_cnt = 0
            for tri in dict["train"]:
                kg = get_knowledge_lis(dict["train"][tri]["knowledge"])
                cur_len = 1 + (len(dict["train"][tri]["conversation"]) - 3) // 2
                for cnt in range(cur_len):
                    tr_lis.append([])
                    for t in range(3):
                        tr_lis[tr_cnt].append([])
                    tr_lis[tr_cnt][0] = kg
                    if cnt == 0:
                        tr_lis[tr_cnt][1] = dict["train"][tri]["conversation"][0].strip().split(" ")\
                                               + dict["train"][tri]["conversation"][1].strip().split(" ")
                        if ground_filter:
                            tr_lis[tr_cnt][2] = list(
                                filter(lambda t: t in kg_word,
                                       dict["train"][tri]["conversation"][2].strip().split(" ")))
                        else:
                            tr_lis[tr_cnt][2] = dict["train"][tri]["conversation"][2].strip().split(" ")

                    else:
                        tr_lis[tr_cnt][1] = dict["train"][tri]["conversation"][2 * cnt + 1].strip().split(" ")
                        if ground_filter:
                            tr_lis[tr_cnt][2] = list(
                                filter(lambda t: t in kg_word,
                                       dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")))
                        else:
                            tr_lis[tr_cnt][2] = dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")
                    tr_cnt += 1
            for tei in dict["test"]:
                kg = get_knowledge_lis(dict["test"][tei]["knowledge"])
                cur_len = 1 + (len(dict["test"][tei]["conversation"]) - 3) // 2
                for cnt in range(cur_len):
                    te_lis.append([])
                    for t in range(3):
                        te_lis[te_cnt].append([])
                    te_lis[te_cnt][0] = kg
                    if cnt == 0:
                        te_lis[te_cnt][1] = dict["test"][tei]["conversation"][0].strip().split(" ")\
                                               + dict["test"][tei]["conversation"][1].strip().split(" ")
                        if ground_filter:
                            te_lis[te_cnt][2] = list(
                                filter(lambda t: t in kg_word,
                                       dict["test"][tei]["conversation"][2].strip().split(" ")))
                        else:
                            te_lis[te_cnt][2] = dict["test"][tei]["conversation"][2].strip().split(" ")
                    else:
                        te_lis[te_cnt][1] = dict["test"][tei]["conversation"][2 * cnt + 1].strip().split(" ")
                        if ground_filter:
                            te_lis[te_cnt][2] = list(
                                filter(lambda t: t in kg_word,
                                       dict["test"][tei]["conversation"][2 * cnt + 2].strip().split(" ")))
                        else:
                            te_lis[te_cnt][2] = dict["test"][tei]["conversation"][2 * cnt + 2].strip().split(" ")
                    te_cnt += 1
            val_num = int(len(dict["test"]) / 2)
        return tr_lis, list(te_lis[:val_num]), list(te_lis[val_num:])
    else:
        assert False, "lic only has 2 datasets! pick one!"


def get_knowledge_lis(lis):
    res = []
    for kth, k in enumerate(lis):
        res.append([])
        assert type(k) == list
        res[kth] = k[0].strip().split(" ") + k[1].strip().split(" ") + k[2].strip().split(" ")
    return res


def extract_knowledge(filename="data/lic/train+test.json"):
    all_kg = []
    if filename == "data/lic/train+test.json":
        if os.path.exists("data/lic/knowledge.pkl"):
            return
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            for tri in dict["train"]:
                kg = get_knowledge_lis(dict["train"][tri]["knowledge"])
                all_kg += kg
            all_kg = list(set(chain.from_iterable(all_kg)))
            pkl.dump(all_kg, open("data/lic/knowledge.pkl", "wb"))
    elif filename == "data/lic/train_part.json":
        if os.path.exists("data/lic/toy_knowledge.pkl"):
            return
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            for tri in dict["train"]:
                kg = get_knowledge_lis(dict["train"][tri]["knowledge"])
                all_kg += kg
            all_kg = list(set(chain.from_iterable(all_kg)))
            pkl.dump(all_kg, open("data/lic/toy_knowledge.pkl", "wb"))
    else:
        assert False, "lic only has 2 datasets! pick one!"


if __name__ == "__main__":
    extract_knowledge()
    # tr, te = pre_process()
