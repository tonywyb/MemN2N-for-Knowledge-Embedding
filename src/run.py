import logging
import sys
import os
import time
import pickle
import torch
import importlib.util
import shutil
import numpy as np
sys.path.append('./')
from utils.customargparse import CustomArgumentParser, args_to_dict
from utils.misc_utils import create_directory

############################## Main method ###############################


def run(config, logger):
    # Extract configs
    cfg_tr = config['train']
    cfg_val = config['valid']
    cfg_te = config['test']
    cfg_net = config['net']
    cfg_eval = config['eval']

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
    logger.info('Using device: {}'.format(device))

    # Seeding
    if config['seed'] is not None:
        seed = config['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Generate global records dictionary
    global_records = {'info': {}, 'result': {}}

    # Save configs and models
    pickle.dump(config, file=open(os.path.join(config['outdir'], 'config.pkl'), 'wb'))
    shutil.copy(src=config['configfile'], dst=os.path.join(config['outdir'], 'configfile.py'))
    shutil.copy(src=config['modelfile'], dst=os.path.join(config['outdir'], 'modelfile.py'))
    shutil.copy(src=config['datapath'], dst=os.path.join(config['outdir'], 'dataset.py'))

    # Import data and create data loaders
    spec = importlib.util.spec_from_file_location('data', config['datapath'])
    data_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_mod)
    tr_loader = torch.utils.data.DataLoader(data_mod.tr_dataset,
                                            batch_size=cfg_tr['batch_size'], shuffle=cfg_tr['shuffle'],
                                            **config['data_loader'])
    te_loader = torch.utils.data.DataLoader(data_mod.te_dataset,
                                            batch_size=cfg_te['batch_size'], shuffle=cfg_te['shuffle'],
                                            **config['data_loader'])
    if 'val_dataset' in dir(data_mod) and data_mod.val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(data_mod.val_dataset,
                                                 batch_size=cfg_val['batch_size'], shuffle=cfg_val['shuffle'],
                                                 **config['data_loader'])
    else:
        val_loader = None

    # Import model

    spec = importlib.util.spec_from_file_location('model', config['modelfile'])
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    model = model_mod.Model(device, logger, global_records, config,
                            num_vocab=data_mod.tr_dataset.num_vocab, sentence_size=data_mod.tr_dataset.sentence_size)

    # Train mode
    if config['mode'] == 'train':
        # Train model
        tic = time.time()
        results = model.fit(tr_loader, val_loader)
        toc = time.time()
        global_records['tr_time'] = toc - tic

        # Save results
        pickle.dump(global_records, file=open(os.path.join(config['outdir'], 'tr_records.pkl'), 'wb'))

    # Eval mode
    elif config['mode'] == 'eval':
        # Decide data usage
        assert cfg_eval['usage'] in ['train', 'valid', 'test'], 'usage should be one of [train, valid, test]'
        if cfg_eval['usage'] == 'train':
            data_loader = tr_loader
        elif cfg_eval['usage'] == 'test':
            data_loader = te_loader
        elif cfg_eval['usage'] == 'valid':
            data_loader = val_loader

        # Evaluate on data
        tic = time.time()
        results = model.evaluate(data_loader)
        toc = time.time()
        global_records['eval_time'] = toc - tic

        # Save results
        pickle.dump(global_records, file=open(os.path.join(config['outdir'], 'eval_records.pkl'), 'wb'))


############################## Main code ###############################


if __name__ == '__main__':
    # Initial time
    t_init = time.time()

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

    # Create output directory if it does not exist
    create_directory(config['outdir'])

    # Set up logger
    logf = open(os.path.join(config['outdir'], config['logging']['log_file']), 'w')
    logging.basicConfig(stream=logf, format=config['logging']['fmt'], level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.getLevelName(config['logging']['level']))
    logger.disabled = (not config['verbose'])

    # Call the main method
    run(config, logger)

    # Final time
    t_final = time.time()
    logger.info('Program finished in {} secs.'.format(t_final - t_init))

    # Close logging file
    logf.close()
