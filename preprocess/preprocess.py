# encoding: utf-8
from __future__ import unicode_literals
import json
import codecs


def pre_process(filename):
    if filename == "data/lic/train+test.json":
        with codecs.open("data/lic/filter_200.txt", "r", "utf-8") as f:
            nil_word = f.read().strip().split(" ")
        with open(filename, "rb") as f:
            dict = json.loads(f.read())
            train_num = int(3 * len(dict["train"]) / 4)
            res_lis = list()
            res_cnt = 0
            for tri in dict["train"]:
                kg = get_knowledge_lis(dict["train"][tri]["knowledge"])
                cur_len = 1 + (len(dict["train"][tri]["conversation"]) - 3) // 2
                for cnt in range(cur_len):
                    res_lis.append([])
                    for t in range(3):
                        res_lis[res_cnt].append([])
                        res_lis[res_cnt][0] = kg
                    if cnt == 0:
                        res_lis[res_cnt][1] = dict["train"][tri]["conversation"][0].strip().split(" ")\
                                               + dict["train"][tri]["conversation"][1].strip().split(" ")
                        res_lis[res_cnt][2] = list(
                            filter(lambda t: t not in nil_word,
                                   dict["train"][tri]["conversation"][2].strip().split(" ")))
                    else:
                        res_lis[res_cnt][1] = dict["train"][tri]["conversation"][2 * cnt + 1].strip().split(" ")
                        res_lis[res_cnt][2] = list(
                            filter(lambda t: t not in nil_word,
                                   dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")))
                    res_cnt += 1
        return res_lis[:train_num], res_lis[train_num:]
    elif filename == "data/lic/train_part.json":
        with codecs.open("data/lic/toy_filter_200.txt", "r", "utf-8") as f:
            toy_nil_word = f.read().strip().split(" ")
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
                        tr_lis[tr_cnt][2] = list(
                            filter(lambda t: t not in toy_nil_word,
                                   dict["train"][tri]["conversation"][2].strip().split(" ")))

                    else:
                        tr_lis[tr_cnt][1] = dict["train"][tri]["conversation"][2 * cnt + 1].strip().split(" ")
                        tr_lis[tr_cnt][2] = list(
                            filter(lambda t: t not in toy_nil_word,
                                   dict["train"][tri]["conversation"][2 * cnt + 2].strip().split(" ")))
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
                        te_lis[te_cnt][2] = list(
                            filter(lambda t: t not in toy_nil_word,
                                   dict["test"][tei]["conversation"][2].strip().split(" ")))
                    else:
                        te_lis[te_cnt][1] = dict["test"][tei]["conversation"][2 * cnt + 1].strip().split(" ")
                        te_lis[te_cnt][2] = list(
                            filter(lambda t: t not in toy_nil_word,
                                   dict["test"][tei]["conversation"][2 * cnt + 2].strip().split(" ")))
                    te_cnt += 1
        return tr_lis, te_lis
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
