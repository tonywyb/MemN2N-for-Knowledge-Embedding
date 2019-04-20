import os
import errno
import torch
import torch.nn.functional as F


#### Files and Directories ####

def delete_files(folder, recursive=False):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif recursive and os.path.isdir(file_path):
                delete_files(file_path, recursive)
                os.unlink(file_path)
        except Exception as e:
            print(e)


def create_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


#### Logging Dictionary Tools ####

def get_by_dotted_path(d, path, default=[]):
    """ Get an entry from nested dictionaries using a dotted path.

    Args:
        d: Dictionary
        path: Entry to extract

    Example:
    >>> get_by_dotted_path({'foo': {'a': 12}}, 'foo.a')
    12
    """
    if not path:
        return d
    split_path = path.split('.')
    current_option = d
    for p in split_path:
        if p not in current_option:
            return default
        current_option = current_option[p]
    return current_option


def add_record(key, value, global_logs):
    if 'logs' not in global_logs['info']:
        global_logs['info']['logs'] = {}
    logs = global_logs['info']['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


def get_records(key, global_logs):
    logs = global_logs['info'].get('logs', {})
    return get_by_dotted_path(logs, key)


def log_record_dict(usage, log_dict, global_logs):
    for log_key, value in log_dict.items():
        add_record('{}.{}'.format(usage, log_key), value, global_logs)


#### Controlling verbosity ####

def vprint(verbose, *args, **kwargs):
    ''' Prints only if verbose is True.
    '''
    if verbose:
        print(*args, **kwargs)


def vcall(verbose, fn, *args, **kwargs):
    ''' Calls function fn only if verbose is True.
    '''
    if verbose:
        fn(*args, **kwargs)


#### Torch aliases ####

optim_list = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}

activation_list = {
    'relu': F.relu,
    'sigmoid': F.sigmoid,
    'tanh': torch.tanh,
    'leaky_relu': F.leaky_relu,
    'log_softmax': F.log_softmax,
    'softmax': F.softmax,
    'elu': F.elu,
    'linear': (lambda x: x)
}

loss_list = {
    'ce': torch.nn.CrossEntropyLoss,
    'bce': torch.nn.BCELoss,
    'mse': torch.nn.MSELoss
}