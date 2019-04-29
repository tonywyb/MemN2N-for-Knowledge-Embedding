from utils.customargparse import CustomArgumentParser, args_to_dict
from data.lic.toyDataset import toyDataset
import numpy as np
import os
import logging
import torch
import importlib
import random
import sys
sys.path.append('./')


class RuleExtractor:
    def __init__(self, device, logger, config_init, *args, **kwargs):
        self.device = device
        self.logger = logger
        self.hops = config_init["rule_hops"]

    def single_extract(self, story, query, unique):
        batch_size, memory_size, sentence_len = story.shape
        kg = np.zeros((batch_size, sentence_len))
        for bs in range(batch_size):
            ms_pool = np.array([], dtype=np.int)
            for ms in range(memory_size):
                if len(np.intersect1d(story[bs, ms, :], query[bs])) > 1:
                    ms_pool = np.append(ms_pool, ms)
            ms_pool = np.setdiff1d(ms_pool, np.array(unique[bs]))
            if len(ms_pool) == 0:
                while True:
                    ms_picked = random.randint(0, memory_size - 1)
                    if ms_picked not in unique[bs] and story[bs, ms_picked, :].sum() != 0:
                        kg[bs, :] = story[bs, ms_picked, :]
                        break
            else:
                ms_picked = random.randint(0, len(ms_pool) - 1)
                kg[bs, :] = story[bs, ms_picked, :]
                unique[bs].append(ms_picked)
        return kg

    def extract(self, story, query, unique):
        for ith, num in enumerate(self.hops):
            for _ in range(num):
                query = self.single_extract(story, query, unique)
            res = query if ith == 0 else np.stack((res, query), axis=1)
        return res


if __name__ == "__main__":
    # Create parser
    parser = CustomArgumentParser(description='Run exps')

    # Add arguments
    parser.add_argument('-c', '--configfile',
                        help='Input file for parameters, constants and initial settings')
    parser.add_argument('-d', '--datapath',
                        help='Path of python module to load the dataset in Torch format')
    parser.add_argument('-m', '--modelfile',
                        help='Input file for model')
    parser.add_argument('-o', '--outdir',
                        help='Output directory for results')
    parser.add_argument('-mode', '--mode',
                        choices=['train', 'eval'], default='train',
                        help='Mode: [train, eval]')
    parser.add_argument('-v', '--verbose',
                        help='Increase output verbosity', action='store_true')

    # Parse arguments
    args = parser.parse_args()
    config = args_to_dict(args)

    # Set up logger
    logf = open(os.path.join(config['outdir'], config['logging']['log_file']), 'w')
    logging.basicConfig(stream=logf, format=config['logging']['fmt'], level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    # Device: cpu/gpu
    if 'device' in config:
        if config['device'].startswith('cuda') and torch.cuda.is_available():
            device = torch.device(config['device'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')

    dataset_dir = "data/lic/"
    te_dataset = toyDataset(dataset_dir, train=False)

    # Import data and create data loaders
    spec = importlib.util.spec_from_file_location('data', config['datapath'])
    data_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_mod)
    te_loader = torch.utils.data.DataLoader(data_mod.te_dataset,
                                            batch_size=config['test']['batch_size'], shuffle=config['test']['shuffle'],
                                            **config['data_loader'])
    extractor = RuleExtractor(device, logger, config["init"])
    for batch_idx, (story, query, target) in enumerate(te_loader):
        story = story.cpu().numpy()
        query = query.cpu().numpy()

        def idx2sent(idx, idx_word):
            return " ".join([idx_word[i] for i in idx])


        unique_lis = np.zeros((story.shape[0], 1), dtype=np.int).tolist()
        res = extractor.extract(story, query, unique_lis)

        for r_batch in range(res.shape[0]):
            print("batch: {}".format(r_batch))
            for r_hop in range(res.shape[1]):
                print("hops: {}".format(r_hop))
                print(idx2sent(res[r_batch, r_hop], te_dataset.idx_word))
        print("=" * 30)


