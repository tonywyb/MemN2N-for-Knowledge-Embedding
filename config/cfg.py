config = {
    'device': 'cuda:0',                             # device to run on

    # config for logging
    'logging': {
        'log_file': 'run.log',                      # log file name
        'fmt': '%(asctime)s: %(message)s',          # logging format
        'level': 'DEBUG',                           # logger level
    },

    # config to load and save networks
    'net': {
        'net_path': 'memn2n/memn2n.py',             # path to load network from
        'saved_params_path': None                   # path to load saved weights in a loaded network
    },

    # config for data loader
    'data_loader': {
        'num_workers': 1,                           # Number of parallel CPU workers
        'pin_memory': False                         # Copy tensors into CUDA pinned memory before returning them
    },

    'seed': None,                                   # Random seed for numpy, torch and cuda (None, int)
    'K': 1,                                         # Top K labels to use for evaluating classifier

    # config to initialize net
    'init': {
        'optimizer': 'sgd',                         # {adam, sgd, adamax, adadelta, adagrad, rmsprop}
        'reg_param_l2': 0.0,                        # Coefficient for l2 regularization loss
        'loss_func': 'ce',                          # loss function during training
        'decay_interval': 25,                       # epoch interval for learning rate decay
        'decay_ratio': 0.5,                         # learning rate decay
        'max_clip': 40.0,                           # gradient noise and clip
        'max_hops': 3,                              # hops of memory network
        'embedding_dim': 20,                        # query/memory embedding in memory network
        'log_interval': 100,                        # Logging interval for debugging (None, int)
        'params': {
            'lr': 1e-2,                             # Learning rate: float
            'momentum': 0.0                         # Momentum: float in [0.0, 1.0]
        }
    },

    # config to control training
    'train': {
        'stop_crit':{
            'max_patience': 10,                     # patience for early stopping (int, None)
            'max_epoch': 50                         # maximum epochs to run for (int)
        },        
        'batch_size': 1,
        'debug_samples': [3, 37, 54],               # sample ids to debug with (None, int, list)
        'shuffle': True
    },
    # config to control validation
    'valid': {
        'batch_size': 1,
        'debug_samples': [0, 1, 2],                 # sample ids to debug with (None, int, list)
        'shuffle': False
    },
    # config to control testing
    'test': {
        'batch_size': 1,
        'debug_samples': [0, 1, 2],                 # sample ids to debug with (None, int, list)
        'shuffle': False
    },
    # config to control evaluation
    'eval': {
        'usage': 'test',                            # what dataset to use {train, valid, test}
        'debug_samples': list(range(10))            # sample ids to debug with (None, int, list)
    }
}
