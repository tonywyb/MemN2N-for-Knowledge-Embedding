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
        self.net = net_mod.Net(self.device, self.logger, num_vocab=kwargs["num_vocab"],
                               sentence_size=kwargs["sentence_size"], **self.config['init'])
        # Then load its params if available
        if self.config['net'].get('saved_params_path', None) is not None:
            self.load_net(self.config['net']['saved_params_path'])
        # Transfer network to device
        self.net.to(self.device)
        self.logger.info(self.net)

    def fit(self, tr_loader, val_loader, *args, **kwargs):
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
                    'final_valid_epoch': epoch
                })

                # Check for early-stopping
                if epoch_records['loss'] < best_valid_loss:
                    best_valid_loss = epoch_records['loss']
                    best_valid_acc = epoch_records['acc']
                    best_valid_epoch = epoch
                    self.global_records['result'].update({
                        'best_valid_loss': best_valid_loss,
                        'best_valid_acc': best_valid_acc,
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
        epoch_total = 0
        ## Set network mode
        self.net.eval()
        torch.set_grad_enabled(False)
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            ## Predict output
            net_out = self.net(data)
            ## Compute loss and accuracy
            batch_loss = self.net.total_loss(net_out, target).data
            batch_acc, batch_correct, batch_total = self.compute_topK_acc(
                nn.Softmax(dim=1)(net_out), target, K=self.config['K'])
            epoch_loss += batch_loss
            epoch_correct += batch_correct
            epoch_total += batch_total

        # Return epoch records
        epoch_records = {
            'loss': epoch_loss.data.item() / float(epoch_total),
            'acc': 100.0 * float(epoch_correct) / float(epoch_total)
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
        })

    def save_net(self, filename):
        torch.save(self.net.state_dict(), filename)
        self.logger.info('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.info('Loading params from {}'.format(filename))
        self.net.load_state_dict(torch.load(filename), strict=False)

    def compute_acc(self, preds, labels):
        with torch.no_grad():
            _, predicted = torch.max(preds.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
            return float(correct) / float(total), correct, total

    def compute_topK_acc(self, preds, labels, K=1):
        if K == 1:
            return self.compute_acc(preds, labels)
        with torch.no_grad():
            # Compute top K predictions
            ord_ind = np.argsort(preds.data, axis=1, kind='mergesort')
            topK_ind = ord_ind[:, -1:-1 - K:-1]
            preds_topK = np.zeros_like(preds.data)
            for i in range(preds.data.shape[0]):
                preds_topK[i, topK_ind[i]] = 1

            # Compute top K accuracy
            total = labels.size(0)
            correct = np.sum(preds_topK[range(total), labels] > 0.0)
            return float(correct) / float(total), correct, total

    def print_record_dict(self, record_dict, usage, t_taken):
        self.logger.info('{}: Loss: {:.3f}, Top {} Accuracy: {:.3f}% took {:.3f}s'.format(
            usage, record_dict['loss'], self.config['K'], record_dict['acc'], t_taken))

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
        }
        return plots

    def _decay_learning_rate(self, opt, epoch):
        decay_interval = self.config.decay_interval
        decay_ratio    = self.config.decay_ratio

        decay_count = max(0, epoch // decay_interval)
        lr = self.config.lr * (decay_ratio ** decay_count)
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        return lr

Model = Trainer

'''
        settings = {
            "use_cuda": config.cuda,
            "num_vocab": self.train_data.num_vocab,
            "embedding_dim": 20,
            "sentence_size": self.train_data.sentence_size,
            "max_hops": config.max_hops
        }

        print("Longest sentence length", self.train_data.sentence_size)
        print("Longest story length", self.train_data.max_story_size)
        print("Average story length", self.train_data.mean_story_size)
        print("Number of vocab", self.train_data.num_vocab)
        if config.dataset_option == "lic":
            print("largest Number of story", self.train_data.max_num_story)

        self.mem_n2n = MemN2N(settings)
        self.ce_fn = nn.CrossEntropyLoss(size_average=False)
        self.mse_fn = nn.MSELoss()
        self.opt = torch.optim.SGD(self.mem_n2n.parameters(), lr=config.lr)
        print(self.mem_n2n)

        if config.cuda:
            self.ce_fn   = self.ce_fn.cuda()
            self.mem_n2n = self.mem_n2n.cuda()

        self.start_epoch = 0
        self.config = config

    def fit(self):
        config = self.config
        for epoch in range(self.start_epoch, config.max_epochs):
            loss = self._train_single_epoch(epoch)
            lr = self._decay_learning_rate(self.opt, epoch)

            if (epoch+1) % 10 == 0:
                train_acc = self.evaluate("train")
                test_acc = self.evaluate("test")
                print(epoch+1, loss, train_acc, test_acc)
        print(train_acc, test_acc)

    def load(self, directory):
        pass

    def evaluate(self, _data="test"):
        correct = 0
        loader = self.train_loader if _data == "train" else self.test_loader
        for step, (story, query, answer) in enumerate(loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if self.config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            pred_prob = self.mem_n2n(story, query)[1]
            pred = pred_prob._data.max(1)[1] # max func return (max, argmax)
            correct += pred.eq(answer._data).cpu().sum()

        acc = correct / len(loader.dataset)
        return acc

    def _train_single_epoch(self, epoch):
        config = self.config
        num_steps_per_epoch = len(self.train_loader)
        for step, (story, query, answer) in enumerate(self.train_loader):
            story = Variable(story)
            query = Variable(query)
            answer = Variable(answer)

            if config.cuda:
                story = story.cuda()
                query = query.cuda()
                answer = answer.cuda()

            self.opt.zero_grad()
            loss = self.ce_fn(self.mem_n2n(story, query)[0], answer)
            loss.backward()

            self._gradient_noise_and_clip(self.mem_n2n.parameters(),
                                          noise_stddev=1e-3, max_clip=config.max_clip)
            self.opt.step()

        return loss.data[0]
'''

