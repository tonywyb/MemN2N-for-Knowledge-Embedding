config = {
    'device': 'cuda:0',                             # device to run on

    # config for logging
    'logging': {
        'log_file': 'run.log',                      # log file name
        'fmt': '%(asctime)s: %(message)s',          # logging format
        'level': 'INFO',                            # logger level
    },

    # config to load and save networks
    'net': {
        'net_path': 'models/net_nn_fc_mnist.py',    # path to load network from
        'saved_params_path': None                   # path to load saved weights in a loaded network
    },

    # config for data loader
    'data_loader': {
        'num_workers': 1,                           # Number of parallel CPU workers
        'pin_memory': False                         # Copy tensors into CUDA pinned memory before returning them
    },

    'seed': None,                                   # Random seed for numpy, torch and cuda (None, int)

    # config to initialize net
    'init': {
        'input_shape': (28, 28, 1),                 # Input shape of network
        'output_shape': (10,),                      # Output shape of network
        'log_interval': 500,                        # Logging interval for debugging (None, int)
        'hard_gated_linear_params': {
            'tau': [0.3, 0.3, 0.3],                 # Threshold for p
            'exp_thresh': [0.001, 0.001, 0.001],    # Expansion fraction threshold
            'init_sigma': [1.0, 0.8, 0.5],
            'cl_mech': 'unirep',                    # Continual Learning Mechanism: [None, 'reg', 'rep', 'unirep']
            'lmbda': 100.0,                         # Tradeoff between loss and cl_mech_loss; Recommended: {'reg': 0.1, 'rep': 0.001, 'unirep': 0.1}
            'log_interval': 500,                    # Logging interval for debugging (None, int)
            'activation': ['tanh', 'tanh', 'linear']# Layer activation
        },
        'hard_gated_conv_params': {
            'if_pad': [True, True, False, False],
            'tau': [0.5, 0.5, 0.5, 0.5],
            'exp_thresh': [0.1, 0.1, 0.1, 0.1],
            'init_sigma': [1.0, 1.0, 1.0, 1.0],
            'cl_mech': None,                       # Continual Learning Mechanism: [None, 'reg', 'rep', 'unirep']
            'lmbda': 1,                            # Tradeoff between loss and cl_mech_loss; Recommended: {'reg': 0.1, 'rep': 0.001, 'unirep': 0.1}
            'log_interval': 500,                   # Logging interval for debugging (None, int)
            'activation': ['linear', 'linear', 'linear', 'linear']                  # Layer activation
        },
        'n_tr_loops': 3,                           # Number of internal training loops required per minibatch
        'optimizer': 'sgd',                        # {adam, sgd, adamax, adadelta, adagrad, rmsprop}
        'reg_param_l2': 0.0,                       # Coefficient for l2 regularization loss
        'params': {
            'lr': 5e-5,                            # Learning rate: float
            'momentum': 0.0                        # Momentum: float in [0.0, 1.0]
        }
    },

    # config to control training
    'train': {
        'stop_crit':{
            'max_patience': 10,                     # patience for early stopping (int, None)
            'max_epoch': 15                         # maximum epochs to run for (int)
        },        
        'batch_size': 1,        
        'debug_samples': [3, 37, 54],               # sample ids to debug with (None, int, list)
        'shuffle': False
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
