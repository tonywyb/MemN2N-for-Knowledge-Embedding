#encoding: utf-8
from __future__ import unicode_literals
import json
import copy

ori = json.loads(open("train_part.json", "rb").read())
new = copy.deepcopy(ori)

for ith, i in enumerate(ori["train"].values()):
	new["train"][str(ith)]["knowledge"] = []
	new["train"][str(ith)]["goals"] = []
	for jth, j in enumerate(i["knowledge"]):
		new["train"][str(ith)]["knowledge"].append(" ".join(j))


for ith, i in enumerate(ori["test"].values()):
	new["train"][str(ith)]["knowledge"] = []
	new["test"][str(ith)]["goals"] = []
	for jth, j in enumerate(i["knowledge"]):
		new["train"][str(ith)]["knowledge"].append(" ".join(j))

f = open("new_train_part.json", "w")
res = json.dumps(new)
f.write(res)