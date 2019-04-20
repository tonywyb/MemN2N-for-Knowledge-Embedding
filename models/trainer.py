# encoding:utf-8
import torch
import torch.nn as nn
import os
import importlib.util
import time
import numpy as np

import sys
sys.path.append('./')
from utils.misc_utils import get_by_dotted_path, add_record, get_records, log_record_dict
from utils.plot_utils import create_curve_plots
from models.rule import RuleExtractor


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
        self.rule = RuleExtractor(self.device, self.logger, self.config['init'])
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
                'final_train_Rule_F1_1_hop': epoch_records['Rule_F1_1_hop'],
                'final_train_Rule_F1_2_hops': epoch_records['Rule_F1_2_hops'],
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
                    'final_valid_Rule_F1_1_hop': epoch_records['Rule_F1_1_hop'],
                    'final_valid_Rule_F1_2_hops': epoch_records['Rule_F1_2_hops'],
                    'final_valid_epoch': epoch
                })

                # Check for early-stopping
                if epoch_records['loss'] < best_valid_loss:
                    best_valid_loss = epoch_records['loss']
                    best_valid_acc = epoch_records['acc']
                    best_valid_F1 = epoch_records['F1']
                    best_valid_Rule_F1_1_hop = epoch_records['Rule_F1_1_hop']
                    best_valid_Rule_F1_2_hops = epoch_records['Rule_F1_2_hops']
                    best_valid_epoch = epoch
                    self.global_records['result'].update({
                        'best_valid_loss': best_valid_loss,
                        'best_valid_acc': best_valid_acc,
                        'best_valid_F1': best_valid_F1,
                        'best_valid_Rule_F1_1_hop': best_valid_Rule_F1_1_hop,
                        'best_valid_Rule_F1_2_hops': best_valid_Rule_F1_2_hops,
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
        log = self.config['init']['log_interval']
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
                if batch_idx % log == 0:
                    self.logger.debug('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(story), len(data_loader.dataset),
                               100.0 * (batch_idx + 1.0) / len(data_loader), batch_loss))

        # Evaluate
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_F1 = 0.0
        epoch_rule_F1 = np.zeros((len(self.config['init']['rule_hops'])))
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
            rule_out = self.rule.extract(story.cpu().numpy(), query.cpu().numpy(),
                                         np.zeros((story.shape[0], 1), dtype=np.int).tolist())
            batch_rule_F1 = np.zeros((len(self.config['init']['rule_hops'])))
            # rule_out = self.random_choose(story, query)
            ## Compute loss and accuracy
            if len(target.shape) == 1:              # for bAbI
                batch_loss = self.net.total_loss(net_out[1], target).data
                batch_acc, batch_correct = self.compute_topK_acc(
                    nn.Softmax(dim=1)(net_out[0]), target, K=self.config['K'])
            elif len(target.shape) == 2:            # for lic
                batch_loss = self.net.total_loss(net_out[1], target.float()).data
                log_flag = True if batch_idx % log == 0 else False
                batch_precision, batch_recall, batch_F1 = self.compute_F1(
                    nn.Softmax(dim=1)(net_out[0]), target, log_flag, -1)

                idx_word = self.data_loader.dataset.idx_word

                def print_sent(idx, idx_word):
                    idx = idx.tolist()
                    res = [idx_word[i] for i in idx]
                    res = list(set(res))
                    return res

                if log_flag:
                    print("Key word extraction for sentence \'{}\'".format(print_sent(query[0, :].numpy(), idx_word)))
                    for ith in range(rule_out.shape[1]):
                        print("{} hop rule output:{}".format(ith + 1, print_sent(rule_out[0, ith, :], idx_word)))

                def process_rule(rule_out):
                    res = []
                    for ith in range(rule_out.shape[1]):
                        ans = torch.zeros_like(target)
                        for bs in range(rule_out.shape[0]):
                            for r in rule_out[bs, ith, :]:
                                if r != 0:
                                    ans[bs, int(r)] = 1
                        res.append(ans)
                    return res

                rule_out_lis = process_rule(rule_out)
                for ith, rule_res in enumerate(rule_out_lis):
                    rule_precision, rule_recall, batch_rule_F1[ith] = self.compute_F1(
                        rule_res, target, log_flag, ith + 1)

            epoch_loss += batch_loss
            epoch_correct += batch_correct
            epoch_F1 += batch_F1
            epoch_rule_F1 += batch_rule_F1
            epoch_total += batch_total

        epoch_rule_F1 = epoch_rule_F1.tolist()
        # Return epoch records
        epoch_records = {
            'loss': epoch_loss.data.item() / float(epoch_total),
            'acc': 100.0 * float(epoch_correct) / float(epoch_total),
            'F1': 100.0 * float(epoch_F1) / float(len(data_loader)),
            'Rule_F1_1_hop': 100.0 * float(epoch_rule_F1[0]) / float(len(data_loader)),
            'Rule_F1_2_hops': 100.0 * float(epoch_rule_F1[1]) / float(len(data_loader)),
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
            'F1': epoch_records['F1'],
            'Rule_F1_1_hop': epoch_records['Rule_F1_1_hop'],
            'Rule_F1_2_hops': epoch_records['Rule_F1_1_hop'],
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

    def compute_F1(self, preds, labels, log_flag, rule_hops):
        with torch.no_grad():
            # Compute top K predictions
            ord_ind = np.argsort(preds.data, axis=1, kind='mergesort')
            # Compute non-zero element each answer(label) batch
            labels_topK = labels.cpu().numpy()
            zero_num = (labels_topK != 0).sum(axis=1)
            preds_topK = np.zeros_like(preds.data)
            for i in range(preds.data.shape[0]):
                idx = ord_ind[i, -zero_num[i]:]
                preds_topK[i, idx] = 1

            if log_flag and rule_hops == -1:
                print("MemN2N output:")
                tmp = np.flatnonzero(preds_topK[0])
                res = [self.data_loader.dataset.idx_word[t] for t in tmp]
                print(res)

            labels_topK = np.flatnonzero(labels_topK)
            preds_topK = np.flatnonzero(preds_topK)

            TP = len(np.intersect1d(labels_topK, preds_topK))
            FN = len(np.setdiff1d(labels_topK, preds_topK))
            FP = len(np.setdiff1d(preds_topK, labels_topK))
            precision = float(TP) / float(TP + FP)
            recall = float(TP) / float(TP + FN)
            if TP == 0:
                F1 = 0.
            else:
                F1 = 2 * precision * recall / (precision + recall)
            if log_flag:
                prefix = "" if rule_hops == -1 else "Rule hops " + str(rule_hops) + ":"
                self.logger.debug(prefix + "TP={}, FN={}, FP={}, precision={}, recall={}, F1={}"
                                  .format(TP, FN, FP, precision, recall, F1))
            return precision, recall, F1

    def print_record_dict(self, record_dict, usage, t_taken):
        self.logger.info('{}: Loss: {:.3f}, F1 score: {:.3f}%, Rule_F1_1_hop score: {:.3f}%,'
                         ' Rule_F1_2_hops score: {:.3f}%, Top {} Accuracy: {:.3f}% took {:.3f}s'
                         .format(usage, record_dict['loss'], record_dict['F1'], record_dict['Rule_F1_1_hop'],
                                 record_dict['Rule_F1_2_hops'], self.config['K'], record_dict['acc'], t_taken))

    def _plot_helper(self):
        plots = {
            'loss': {
                'plot_dict': {
                    'train': get_records('train.loss', self.global_records),
                    'valid': get_records('valid.loss', self.global_records)
                },
                'coarse_range': [0, 200],
                'fine_range': [0, 50]
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
                    'valid': get_records('valid.F1', self.global_records),
                    'Rule_F1_1_hop_train': get_records('train.Rule_F1_1_hop', self.global_records),
                    'Rule_F1_1_hop_valid': get_records('valid.Rule_F1_1_hop', self.global_records),
                    'Rule_F1_2_hops_train': get_records('train.Rule_F1_2_hops', self.global_records),
                    'Rule_F1_2_hops_valid': get_records('valid.Rule_F1_2_hops', self.global_records)
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
