import argparse
import logging
import sys
import os
sys.path.append('./')
from utils.customargparse import CustomArgumentParser, args_to_dict
from utils.misc_utils import create_directory
from memn2n.trainer import trainer


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="data/lic/train+test.json")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--max_hops", type=int, default=1)  # 3
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay_interval", type=int, default=25)
    parser.add_argument("--decay_ratio", type=float, default=0.5)
    parser.add_argument("--max_clip", type=float, default=40.0)
    parser.add_argument("--max_sentence_len", type=int, default=30)
    parser.add_argument("--dataset_option", type=str, default="lic")

    return parser.parse_args()


def main(config):

    # Create output directory if it does not exist
    create_directory(config['outdir'])

    # Set up logger
    logf = open(os.path.join(config['outdir'], config['logging']['log_file']), 'w')
    logging.basicConfig(stream=logf, format=config['logging']['fmt'], level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    t = trainer.Trainer(config)
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    main(config)
