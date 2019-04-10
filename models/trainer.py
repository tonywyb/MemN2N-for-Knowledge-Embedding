from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import sys
import os
import importlib.util
import time
import numpy as np

sys.path.append('./')
from utils.misc_utils import get_by_dotted_path, add_record, get_records, log_record_dict, optim_list
from utils.plot_utils import create_curve_plots


'''
Story: 32 * 20(max_num_story) * max_sent_len
Query: 32 * max_sent_len
Answer: 32 * max_sent_len

Data_story: train_num * max_num_story * max_sent_len
Data_answer: train_num * max_sent_len
Data_query:train_num * max_sent_len
'''


class Trainer:
    def __init__(self, device, logger, global_records, config, *args, **kwargs):
        # Initializations
        self.device = device
        self.logger = logger
        self.global_records = global_records
        self.config = config

        # Load net module
        assert 'net_path' in self.config['net'], "net_path not specified in config['net']"
        spec = importlib.util.spec_from_file_location('net_mod', self.config['net']['net_path'])
        net_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(net_mod)
        sys.modules['net_mod'] = net_mod
        # First load network
        self.net = net_mod.Net(self.device, self.logger, self.config['init'],
                               num_vocab=kwargs["num_vocab"],
                               sentence_size=kwargs["sentence_size"])
        # Then load its params if available
        if self.config['net'].get('saved_params_path', None) is not None:
            self.load_net(self.config['net']['saved_params_path'])
        # Transfer network to device
        self.net.to(self.device)
        self.logger.info(self.net)

    def fit(self, tr_loader, val_loader, *args, **kwargs):
        self.data_loader = tr_loader
        # Initialize params
        if 'max_epoch' in kwargs:
            max_epoch = kwargs['max_epoch']
        elif 'max_epoch' in self.config['train']['stop_crit']:
            max_epoch = self.config['train']['stop_crit']['max_epoch']
        else:
            max_epoch = 100

        if 'max_patience' in kwargs:
            max_patience = kwargs['max_patience']
        elif 'max_patience' in self.config['train']['stop_crit']:
            max_patience = self.config['train']['stop_crit']['max_patience']
        else:
            max_patience = 10

        # Train epochs
        best_valid_loss = np.inf
        best_valid_acc = 0.0
        best_valid_epoch = 0
        early_break = False
        for epoch in range(max_epoch):
            self.logger.info('\n' + 40 * '%' + '  EPOCH {}  '.format(epoch) + 40 * '%')

            # Run train epoch
            t = time.time()
            epoch_records = self.run_epoch(tr_loader, 'train', epoch, *args, **kwargs)
            lr = self._decay_learning_rate(self.net.optimizer, epoch)
            # Log and print train epoch records
            log_record_dict('train', epoch_records, self.global_records)
            self.print_record_dict(epoch_records, 'Train', time.time() - t)
            self.global_records['result'].update({
                'final_train_loss': epoch_records['loss'],
                'final_train_acc': epoch_records['acc'],
                'final_train_F1': epoch_records['F1'],
                'final_train_epoch': epoch
            })

            if val_loader is not None:
                # Run valid epoch
                t = time.time()
                epoch_records = self.run_epoch(val_loader, 'eval', epoch, *args, **kwargs)

                # Log and print valid epoch records
                log_record_dict('valid', epoch_records, self.global_records)
                self.print_record_dict(epoch_records, 'Valid', time.time() - t)
                self.global_records['result'].update({
                    'final_valid_loss': epoch_records['loss'],
                    'final_valid_acc': epoch_records['acc'],
                    'final_valid_F1': epoch_records['F1'],
                    'final_valid_epoch': epoch
                })

                # Check for early-stopping
                if epoch_records['loss'] < best_valid_loss:
                    best_valid_loss = epoch_records['loss']
                    best_valid_acc = epoch_records['acc']
                    best_valid_F1 = epoch_records['F1']
                    best_valid_epoch = epoch
                    self.global_records['result'].update({
                        'best_valid_loss': best_valid_loss,
                        'best_valid_acc': best_valid_acc,
                        'best_valid_F1': best_valid_F1,
                        'best_valid_epoch': best_valid_epoch
                    })
                    self.logger.info('    Best validation loss improved to {:.03f}'.format(best_valid_loss))
                    self.save_net(os.path.join(self.config['outdir'], 'best_valid_params.ptp'))

                if best_valid_loss < np.min(get_records('validation.loss', self.global_records)[-max_patience:]):
                    early_break = True

            # Produce plots
            plots = self._plot_helper()
            if plots is not None:
                for k, v in plots.items():
                    create_curve_plots(k, v['plot_dict'], v['coarse_range'], v['fine_range'], self.config['outdir'])

            # Save net
            self.save_net(os.path.join(self.config['outdir'], 'final_params.ptp'))

            # Early-stopping
            if early_break:
                self.logger.warning(
                    'Early Stopping because validation loss did not improve for {} epochs'.format(max_patience))
                break

    def run_epoch(self, data_loader, mode, epoch, *args, **kwargs):
        # Train
        if mode == 'train':
            ## Set network mode
            self.net.train()
            torch.set_grad_enabled(True)
            for batch_idx, (story, query, target) in enumerate(data_loader):
                story, query, target = story.to(self.device), query.to(self.device), target.to(self.device)
                ## Train on the data batch
                batch_loss = self.net.fit_batch(story, query, target)
                # Log stuff
                log = self.config['init']['log_interval']
                if batch_idx % log == 0:
                    self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(story), len(data_loader.dataset),
                               100.0 * (batch_idx + 1.0) / len(data_loader), batch_loss))

        # Evaluate
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_F1 = 0.0
        epoch_total = 0
        ## Set network mode
        self.net.eval()
        torch.set_grad_enabled(False)
        for batch_idx, (story, query, target) in enumerate(data_loader):
            story, query, target = story.to(self.device), query.to(self.device), target.to(self.device)
            ### initialize record
            batch_loss = 0.
            batch_correct = 0
            batch_F1 = 0.
            batch_total = story.shape[0]    # batch_total = batch_size
            ## Predict output
            net_out = self.net(story, query)
            ## Compute loss and accuracy
            self.logger.debug("Evaluation word output:")
            if len(target.shape) == 1:              # for bAbI
                batch_loss = self.net.total_loss(net_out[1], target).data
                batch_acc, batch_correct = self.compute_topK_acc(
                    nn.Softmax(dim=1)(net_out[0]), target, K=self.config['K'])
            elif len(target.shape) == 2:            # for lic
                self.logger.info("#" * 50)
                batch_loss = self.net.total_loss(net_out[1], target.float()).data
                batch_precision, batch_recall, batch_F1 = self.compute_F1(
                    nn.Softmax(dim=1)(net_out[0]), target)
            epoch_loss += batch_loss
            epoch_correct += batch_correct
            epoch_total += batch_total
            epoch_F1 += batch_F1

        # Return epoch records
        epoch_records = {
            'loss': epoch_loss.data.item() / float(epoch_total),
            'acc': 100.0 * float(epoch_correct) / float(epoch_total),
            'F1': 100.0 * float(epoch_F1) / float(len(data_loader))
        }
        return epoch_records

    def evaluate(self, data_loader, *args, **kwargs):
        # Run eval
        t = time.time()
        epoch_records = self.run_epoch(data_loader, 'eval', 0, *args, **kwargs)

        # Log and print epoch records
        log_record_dict('Eval', epoch_records, self.global_records)
        self.print_record_dict(epoch_records, 'Eval', time.time() - t)
        self.global_records['result'].update({
            'loss': epoch_records['loss'],
            'acc': epoch_records['acc'],
            'F1': epoch_records['F1']
        })

    def save_net(self, filename):
        torch.save(self.net.state_dict(), filename)
        self.logger.info('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.info('Loading params from {}'.format(filename))
        self.net.load_state_dict(torch.load(filename), strict=False)

    def compute_topK_acc(self, preds, labels, K=1):
        with torch.no_grad():
            # Compute top K predictions
            ord_ind = np.argsort(preds.data, axis=1, kind='mergesort')
            topK_ind = ord_ind[:, -K:]
            preds_topK = np.zeros_like(preds.data)
            for i in range(preds.data.shape[0]):
                preds_topK[i, topK_ind[i]] = 1

            # Compute top K accuracy
            total = labels.size(0)
            correct = np.sum(preds_topK[range(total), labels] > 0.0)
            return float(correct) / float(total), correct

    def compute_F1(self, preds, labels):
        with torch.no_grad():
            # Compute top K predictions
            ord_ind = np.argsort(preds.data, axis=1, kind='mergesort')
            batch_size = labels.shape[0]
            # Compute non-zero element each answer(label) batch
            labels_topK = labels.numpy()
            zero_num = (labels_topK != 0).sum(axis=1)
            preds_topK = np.zeros_like(preds.data)
            for i in range(preds.data.shape[0]):
                idx = ord_ind[i, -zero_num[i]:]
                preds_topK[i, idx] = 1
            labels_topK = np.flatnonzero(labels_topK)
            preds_topK = np.flatnonzero(preds_topK)
            # for tmp in preds_topK:
            #     try:
            #         self.logger.debug(self.data_loader.dataset.idx_word[tmp])
            #     except KeyError as e:
            #         print(tmp)
            #         assert False
            #     finally:
            #         pass
            TP = len(np.intersect1d(labels_topK, preds_topK))
            FN = len(np.setdiff1d(labels_topK, preds_topK))
            FP = len(np.setdiff1d(preds_topK, labels_topK))
            precision = float(TP) / float(TP + FP)
            recall = float(TP) / float(TP + FN)
            if TP == 0:
                F1 = 0.
            else:
                F1 = 2 * precision * recall / (precision + recall)
            self.logger.debug("Evaluation result: TP={}, FN={}, FP={}, precision={}, recall={}, F1={}"
                              .format(TP, FN, FP, precision, recall, F1))
            return precision, recall, F1

    def print_record_dict(self, record_dict, usage, t_taken):
        self.logger.info('{}: Loss: {:.3f}, F1 score: {:.3f}%, Top {} Accuracy: {:.3f}% took {:.3f}s'.format(
            usage, record_dict['loss'], record_dict['F1'], self.config['K'], record_dict['acc'], t_taken))

    def _plot_helper(self):
        plots = {
            'loss': {
                'plot_dict': {
                    'train': get_records('train.loss', self.global_records),
                    'valid': get_records('valid.loss', self.global_records)
                },
                'coarse_range': [0, 20],
                'fine_range': [0, 4]
            },
            'acc': {
                'plot_dict': {
                    'train': get_records('train.acc', self.global_records),
                    'valid': get_records('valid.acc', self.global_records)
                },
                'coarse_range': [0, 100],
                'fine_range': [60, 100]
            },
            'F1': {
                'plot_dict': {
                    'train': get_records('train.F1', self.global_records),
                    'valid': get_records('valid.F1', self.global_records)
                },
                'coarse_range': [0, 100],
                'fine_range': [60, 100]
            }
        }
        return plots

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config["init"]["decay_interval"]
        decay_ratio    = self.config["init"]["decay_ratio"]

        decay_count = max(0, epoch // decay_interval)
        lr = self.config["init"]["params"]["lr"] * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr


Model = Trainer
