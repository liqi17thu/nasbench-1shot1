import argparse
import pickle

import numpy as np

from nasbench_analysis.search_spaces.search_space_1 import SearchSpace1
from nasbench_analysis.search_spaces.search_space_2 import SearchSpace2
from nasbench_analysis.search_spaces.search_space_3 import SearchSpace3

import json
import logging
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as dset
from torch.autograd import Variable

from optimizers.darts import utils
from optimizers.darts.genotypes import PRIMITIVES
from nasbench_analysis.utils import NasbenchWrapper
# from optimizers.pc_darts.model_search import PCDARTSNetwork as Network
from optimizers.darts.model_search import Network


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DartsWrapper:
    def __init__(self, save_path, seed, batch_size, grad_clip, epochs, num_intermediate_nodes, search_space, cutout,
                 resume_iter=None, init_channels=16):
        args = {}
        args['data'] = '../data'
        args['epochs'] = epochs
        args['learning_rate'] = 0.025
        args['batch_size'] = batch_size
        args['learning_rate_min'] = 0.001
        args['momentum'] = 0.9
        args['weight_decay'] = 3e-4
        args['init_channels'] = init_channels
        # Adapted to nasbench
        args['layers'] = 9
        args['drop_path_prob'] = 0.3
        args['grad_clip'] = grad_clip
        args['train_portion'] = 0.5
        args['seed'] = seed
        args['log_interval'] = 50
        args['save'] = save_path
        args['gpu'] = 0
        args['cuda'] = False
        args['cutout'] = cutout
        args['cutout_length'] = 16
        args['report_freq'] = 50
        args['output_weights'] = True
        args['steps'] = num_intermediate_nodes
        args['search_space'] = search_space.search_space_number
        self.search_space = search_space
        args = AttrDict(args)
        self.args = args

        # Dump the config of the run, but if only if it doesn't yet exist
        config_path = os.path.join(args.save, 'config.json')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as fp:
                json.dump(args.__dict__, fp)
        self.seed = seed

        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.cuda.set_device(args.gpu)
        cudnn.benchmark = False
        cudnn.enabled = True
        cudnn.deterministic = True
        # torch.cuda.manual_seed_all(args.seed)

        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        self.train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        self.valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=0, worker_init_fn=np.random.seed(args.seed))

        _, test_transform = utils._data_transforms_cifar10(args)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)
        self.test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

        self.train_iter = iter(self.train_queue)
        self.valid_iter = iter(self.valid_queue)

        self.steps = 0
        self.epochs = 0
        self.total_loss = 0
        self.start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        # criterion = criterion.cuda()
        self.criterion = criterion



        model = Network(args.init_channels, 10, args.layers, self.criterion, output_weights=args.output_weights,
                        search_space=search_space, steps=args.steps)

        # model = model.cuda()
        self.model = model

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
        self.optimizer = optimizer

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        if resume_iter is not None:
            self.steps = resume_iter
            self.epochs = int(resume_iter / len(self.train_queue))
            logging.info("Resuming from epoch %d" % self.epochs)
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
            for i in range(self.epochs):
                self.scheduler.step()

        size = 0
        for p in model.parameters():
            size += p.nelement()
        logging.info('param size: {}'.format(size))

        total_params = sum(x.data.nelement() for x in model.parameters())
        logging.info('Args: {}'.format(args))
        logging.info('Model total parameters: {}'.format(total_params))

    def train_batch(self, arch):
        args = self.args
        if self.steps % len(self.train_queue) == 0:
            self.scheduler.step()
            self.objs = utils.AvgrageMeter()
            self.top1 = utils.AvgrageMeter()
            self.top5 = utils.AvgrageMeter()
        lr = self.scheduler.get_lr()[0]

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        step = self.steps % len(self.train_queue)
        input, target = next(self.train_iter)

        self.model.train()
        n = input.size(0)

        # input = input.cuda()
        # target = target.cuda(non_blocking=True)

        # get a random_ws minibatch from the search queue with replacement
        self.optimizer.zero_grad()
        logits = self.model(input, discrete=True)
        loss = self.criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(self.model.parameters(), args.grad_clip)
        self.optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        self.objs.update(loss.data.item(), n)
        self.top1.update(prec1.data.item(), n)
        self.top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, self.objs.avg, self.top1.avg, self.top5.avg)

        self.steps += 1
        if self.steps % len(self.train_queue) == 0:
            # Save the model weights
            self.epochs += 1
            self.train_iter = iter(self.train_queue)
            valid_err = self.evaluate(arch)
            logging.info('epoch %d  |  train_acc %f  |  valid_acc %f' % (self.epochs, self.top1.avg, 1 - valid_err))
            self.save(epoch=self.epochs)

    def evaluate(self, arch, split=None):
        # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        self.model.eval()

        if split is None:
            n_batches = 10
        else:
            n_batches = len(self.valid_queue)

        for step in range(n_batches):
            try:
                input, target = next(self.valid_iter)
            except Exception as e:
                logging.info('looping back over valid set')
                self.valid_iter = iter(self.valid_queue)
                input, target = next(self.valid_iter)
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=True)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1 - 0.01 * top1.avg

    def evaluate_test(self, arch, split=None, discrete=False, normalize=True):
        # Return error since we want to minimize obj val
        logging.info(arch)
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        weights = self.get_weights_from_arch(arch)
        self.set_arch_model_weights(weights)

        self.model.eval()

        if split is None:
            n_batches = 10
        else:
            n_batches = len(self.test_queue)

        for step in range(n_batches):
            try:
                input, target = next(self.test_iter)
            except Exception as e:
                logging.info('looping back over valid set')
                self.test_iter = iter(self.test_queue)
                input, target = next(self.test_iter)
            # input = input.cuda()
            # target = target.cuda(non_blocking=True)

            logits = self.model(input, discrete=discrete, normalize=normalize)
            loss = self.criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % self.args.report_freq == 0:
                logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return 1 - 0.01 * top1.avg

    def save(self, epoch):
        utils.save(self.model, os.path.join(self.args.save, 'one_shot_model_{}.pt'.format(epoch)))

    def load(self, epoch=None):
        if epoch is not None:
            model_obj_path = os.path.join(self.args.save, 'one_shot_model_{}.obj'.format(epoch))
            if os.path.exists(model_obj_path):
                utils.load(self.model, model_obj_path)
            else:
                model_pt_path = os.path.join(self.args.save, 'one_shot_model_{}.pt'.format(epoch))
                utils.load(self.model, model_pt_path)
        else:
            utils.load(self.model, os.path.join(self.args.save, 'weights.obj'))

    def get_weights_from_arch(self, arch):
        adjacency_matrix, node_list = arch
        num_ops = len(PRIMITIVES)

        # Assign the sampled ops to the mixed op weights.
        # These are not optimized
        alphas_mixed_op = Variable(torch.zeros(self.model._steps, num_ops), requires_grad=False)
        for idx, op in enumerate(node_list):
            alphas_mixed_op[idx][PRIMITIVES.index(op)] = 1

        # Set the output weights
        alphas_output = Variable(torch.zeros(1, self.model._steps + 1), requires_grad=False)
        for idx, label in enumerate(list(adjacency_matrix[:, -1][:-1])):
            alphas_output[0][idx] = label

        # Initialize the weights for the inputs to each choice block.
        if type(self.model.search_space) == SearchSpace1:
            begin = 3
        else:
            begin = 2
        alphas_inputs = [Variable(torch.zeros(1, n_inputs), requires_grad=False) for n_inputs in
                         range(begin, self.model._steps + 1)]
        for alpha_input in alphas_inputs:
            connectivity_pattern = list(adjacency_matrix[:alpha_input.shape[1], alpha_input.shape[1]])
            for idx, label in enumerate(connectivity_pattern):
                alpha_input[0][idx] = label

        # Total architecture parameters
        arch_parameters = [
            alphas_mixed_op,
            alphas_output,
            *alphas_inputs
        ]
        return arch_parameters

    def set_arch_model_weights(self, weights):
        self.model._arch_parameters = weights

    def sample_arch(self):
        adjacency_matrix, op_list = self.search_space.sample(with_loose_ends=True, upscale=False)
        return adjacency_matrix, op_list



def correlation_with_weights(model_path, config, epoch):
    if config['search_space'] == '1':
        search_space = SearchSpace1()
    elif config['search_space'] == '2':
        search_space = SearchSpace2()
    elif config['search_space'] == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')
    model = DartsWrapper(save_path=model_path, seed=0, batch_size=128, grad_clip=5, epochs=200,
                         num_intermediate_nodes=search_space.num_intermediate_nodes, search_space=search_space,
                         cutout=False)
    if 'random_ws' in model_path:
        discrete = True
        normalize = False
    else:
        discrete = False
        normalize = True

    model.load(epoch=epoch)
    nb_test_errors = []
    nb_valid_errors = []
    one_shot_test_errors = []
    for adjacency_matrix, ops, model_spec in search_space.generate_search_space_without_loose_ends():
        if str(config['search_space']) == '1' or str(config['search_space']) == '2':
            adjacency_matrix_ss = np.delete(np.delete(adjacency_matrix, -2, 0), -2, 0)
            # Remove input, output and 5th node
            ops_ss = ops[1:-2]
        elif str(config['search_space']) == '3':
            adjacency_matrix_ss = adjacency_matrix
            # Remove input and output node
            ops_ss = ops[1:-1]
        else:
            raise ValueError('Unknown search space')

        one_shot_test_error = model.evaluate_test((adjacency_matrix_ss, ops_ss), split='test', discrete=discrete,
                                                  normalize=normalize)
        one_shot_test_errors.extend(np.repeat(one_shot_test_error, 3))
        # Query NASBench
        data = nasbench.query(model_spec)
        nb_test_errors.extend([1 - item['test_accuracy'] for item in data])
        nb_valid_errors.extend([1 - item['validation_accuracy'] for item in data])
        print('NB', nb_test_errors[-1], 'OS', one_shot_test_errors[-1], 'weights', model.model.arch_parameters())

    correlation = np.corrcoef(one_shot_test_errors, nb_test_errors)[0, -1]
    return correlation, nb_test_errors, nb_valid_errors, one_shot_test_errors


def eval_directory_on_epoch(path, epoch):
    """Evaluates all one-shot architecture methods in the directory."""
    # Read in config
    with open(os.path.join(path, 'config.json')) as fp:
        config = json.load(fp)
    correlations = []
    nb_test_errors, nb_valid_errors, one_shot_test_errors = [], [], []
    correlation, nb_test_error, nb_valid_error, one_shot_test_error = \
        correlation_with_weights(model_path=path,
                                        config=config,
                                        epoch=epoch)
    correlations.append(correlation)
    nb_test_errors.append(nb_test_error)
    nb_valid_error.append(nb_valid_error)
    one_shot_test_errors.append(one_shot_test_error)

    with open(os.path.join(path, 'correlation_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(correlations, fp)

    with open(os.path.join(path, 'nb_test_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(nb_test_errors, fp)

    with open(os.path.join(path, 'nb_valid_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(nb_valid_errors, fp)

    with open(os.path.join(path, 'one_shot_test_errors_{}.obj'.format(epoch)), 'wb') as fp:
        pickle.dump(one_shot_test_errors, fp)


def understanding(model_path):
    with open(os.path.join(model_path, 'config.json')) as fp:
        config = json.load(fp)

    config['search_space'] = '3'

    if config['search_space'] == '1':
        search_space = SearchSpace1()
    elif config['search_space'] == '2':
        search_space = SearchSpace2()
    elif config['search_space'] == '3':
        search_space = SearchSpace3()
    else:
        raise ValueError('Unknown search space')
    model = DartsWrapper(save_path=model_path, seed=0, batch_size=128, grad_clip=5, epochs=200,
                         num_intermediate_nodes=search_space.num_intermediate_nodes, search_space=search_space,
                         cutout=False)

    for adjacency_matrix, ops, model_spec in search_space.generate_search_space_without_loose_ends():
        if str(config['search_space']) == '1' or str(config['search_space']) == '2':
            adjacency_matrix_ss = np.delete(np.delete(adjacency_matrix, -2, 0), -2, 0)
            # Remove input, output and 5th node
            ops_ss = ops[1:-2]
        elif str(config['search_space']) == '3':
            adjacency_matrix_ss = adjacency_matrix
            # Remove input and output node
            ops_ss = ops[1:-1]
        else:
            raise ValueError('Unknown search space')

        arch_parameters = model.get_weights_from_arch((adjacency_matrix_ss, ops_ss))


        acces = []
        means = []
        variances = []
        maximum = []
        minimum = []
        ranges = []
        L1 = []
        L2 = []

        data = nasbench.query(model_spec)

        print(data)

        a = model.model.extract_sub(arch_parameters[0]).state_dict()
        keys = [key for key in a if
                'bn' not in key and 'se' not in key and 'classifier' not in key and 'weight' in key and len(
                    a[key].shape) == 4]

        acces.append()

        weights_list = [a[key].cpu().numpy() for key in keys]
        means.append(np.mean([np.mean(weights) for weights in weights_list]))
        variances.append(np.mean([np.var(weights) for weights in weights_list]))
        maximum.append(np.mean([np.max(weights) for weights in weights_list]))
        minimum.append(np.mean([np.min(weights) for weights in weights_list]))
        ranges.append(np.mean([np.max(weights) - np.min(weights) for weights in weights_list]))
        L2.append(np.mean([np.linalg.norm(weights) for weights in weights_list]))
        L1.append(np.mean([np.abs(weights).mean() for weights in weights_list]))


        print("arch parameters:")
        print(arch_parameters)
        print("model spec")
        print(model_spec.matrix)
        print(model_spec.ops)
        # print('adjacency_matrix_ss:')
        # print(adjacency_matrix_ss)
        # print('ops_ss:')
        # print(ops_ss)
        print()
    print(model.model)

def main():
    understanding(args.model_path)
    # Load NASBench
    # eval_directory_on_epoch(args.model_path, args.epoch)


parser = argparse.ArgumentParser("correlation_analysis")
parser.add_argument('--data', type=str, default='../data', help='location of the darts corpus')
parser.add_argument('--model_path', default="/Users/liqi17thu/Desktop/darts/search_space_1/search-baseline-20200623-134823-0-1",
                    help='Path to where the models are stored.')
parser.add_argument('--epoch', type=int, help='Epoch', default=108)
args = parser.parse_args()

if __name__ == '__main__':
    nasbench = NasbenchWrapper('/Users/liqi17thu/Documents/GitHub/nasbench/nasbench_full.tfrecord')
    main()
