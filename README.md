# MemN2N in Knowledge Embedding
Use [End-To-End Memory Network](https://arxiv.org/abs/1503.08895) to perform topic transfer in the first step. 


## Training
```shell
python src/run.py -c config/cfg.py -d data/bAbI/bAbIDataset.py -m models/trainer.py -o result -mode train -v --net.net_path models/memn2n.py --device cpu
```

## Results
To be continued.
