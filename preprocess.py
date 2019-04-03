# encoding: utf-8
from __future__ import unicode_literals
import json
import copy


def pre_process(mode):
    for ith, i in enumerate(ori[mode].values()):
        tmp_knowledge = i["knowledge"]
        ori[mode][str(ith)]["knowledge"] = [[], []]
        for jth, j in enumerate(tmp_knowledge):
            topic_1 = ori[mode][str(ith)]["goal"][0][1]
            topic_2 = ori[mode][str(ith)]["goal"][0][2]
            if topic_1 in tmp_knowledge[jth]:
                ori[mode][str(ith)]["knowledge"][0].append(j)
            if topic_2 in tmp_knowledge[jth]:
                ori[mode][str(ith)]["knowledge"][1].append(j)

        tmp_conversation = i["conversation"]
        ori[mode][str(ith)]["conversation"] = [[], [], [], [], []]
        goal_1 = i["goal"][0][1]
        goal_2 = i["goal"][0][2]
        goal_1_flag = False
        goal_2_flag = False
        for kth, k in enumerate(tmp_conversation):
            if not goal_1_flag:
                if goal_1 not in k:
                    ori[mode][str(ith)]["conversation"][0].append(k)
                else:
                    ori[mode][str(ith)]["conversation"][1].append(k)
                    goal_1_flag = True
            elif not goal_2_flag:
                if goal_2 not in k:
                    ori[mode][str(ith)]["conversation"][2].append(k)
                else:
                    ori[mode][str(ith)]["conversation"][3].append(k)
                    goal_2_flag = True
            else:
                ori[mode][str(ith)]["conversation"][4].append(k)
        for lth, l in enumerate(ori[mode][str(ith)]["conversation"]):
            if len(l) == 0:
                if lth == 0:
                    ori[mode][str(ith)]["conversation"][lth] = ori[mode][str(ith)]["conversation"][lth + 1]
                else:
                    ori[mode][str(ith)]["conversation"][lth] = ori[mode][str(ith)]["conversation"][lth - 1]


ori = json.loads(open("train_part.json", "rb").read())
mode = ["train", "test"]
for m in mode:
    pre_process(m)

f = open("toy_train_part.json", "w")
res = json.dumps(ori)
f.write(res)
