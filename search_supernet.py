import os
import copy
import math
import random
import time
import logging
import argparse
import heapq

import mxnet as mx
from mxnet import gluon
from mxnet import nd

from oneshot_nas_network import get_shufflenas_oneshot
from utils.calculate_flops import get_flops
from oneshot_nas_blocks import NasBatchNorm

FLOP_MAX = 1
PARAM_MAX = 1
SCORE_ACC_RATIO = 1    # flop_param_score_weight/acc_weight for fitness


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')

    # ---------------------------- data ------------------------------- #
    parser.add_argument('--rec-train', type=str,
                        default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str,
                        default='~/.mxnet/datasets/imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str,
                        default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str,
                        default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--input-size', type=int, default=224,
                        help='the size of the input image')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='crop ratio during validation. default is 0.875')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of preprocessing workers')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU)')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training')
    parser.add_argument('--shuffle-train', type=bool, default=False,
                        help='whether to do shuffle in training data for BN update')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use')

    # ---------------------------- model ------------------------------- #
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext and ShuffleNas')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='whether to follow MobileNet V3 last conv after pooling style')

    # ----------------------- search supernet -------------------------- #
    parser.add_argument('--supernet-params', type=str,
                        default='./params/ShuffleNasOneshot-imagenet-supernet.params',
                        help='supernet parameter directory')
    parser.add_argument('--search-mode', type=str, default='random',
                        help="search mode, options: ['random', 'genetic'] ")
    parser.add_argument('--comparison-model', type=str, default='SinglePathOneShot',
                        help="model to compare with when searching, "
                             "options: ['MobileNetV3_large', 'MobileNetV2_1.4', "
                             "'SinglePathOneShot', 'ShuffleNetV2+_medium']")
    parser.add_argument('--topk', type=int, default=3,
                        help='get top k models')
    parser.add_argument('--search-iters', type=int, default=20,
                        help='how many search iterations')
    parser.add_argument('--update-bn-images', type=int, default=20000,
                        help='How many images to update the BN statistics.')

    # ----------------------- genetic search -------------------------- #
    parser.add_argument('--population-size', type=int, default=50,
                        help='the size of population to keep during searching')
    parser.add_argument('--retain-length', type=int, default=10,
                        help='how many items to keep after fitness')
    parser.add_argument('--random-select', type=float, default=0.1,
                        help='probability of a rejected network remaining in the population')
    parser.add_argument('--mutate-chance', type=float, default=0.1,
                        help='probability a network will be randomly mutated')

    args = parser.parse_args()
    return args


def get_data(batch_size, num_gpus, rec_train, rec_train_idx, rec_val, rec_val_idx,
             input_size, crop_ratio, num_workers, shuffle_train):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    batch_size *= max(1, num_gpus)

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec=rec_train,
        path_imgidx=rec_train_idx,
        preprocess_threads=int(num_workers//2),
        shuffle=shuffle_train,
        batch_size=batch_size,

        resize=resize,
        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2]
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec=rec_val,
        path_imgidx=rec_val_idx,
        preprocess_threads=num_workers,
        shuffle=shuffle_train,
        batch_size=batch_size,

        resize=resize,
        data_shape=(3, input_size, input_size),
        mean_r=mean_rgb[0],
        mean_g=mean_rgb[1],
        mean_b=mean_rgb[2],
        std_r=std_rgb[0],
        std_g=std_rgb[1],
        std_b=std_rgb[2]
    )
    return train_data, val_data, batch_fn


def get_flop_param_score(block_choices, channel_choices, comparison_model='SinglePathOneShot'):
    """ Return the flops and num of params """
    # build fix_arch network and calculate flop
    fixarch_net = get_shufflenas_oneshot(block_choices, channel_choices)
    fixarch_net._initialize()
    if not os.path.exists('./symbols'):
        os.makedirs('./symbols')
    fixarch_net.hybridize()

    # calculate flops and num of params
    dummy_data = nd.ones([1, 3, 224, 224])
    fixarch_net(dummy_data)
    fixarch_net.export("./symbols/ShuffleNas_fixArch", epoch=1)

    flops, model_size = get_flops(symbol_path="./symbols/ShuffleNas_fixArch-symbol.json")  # both in Millions

    # proves ShuffleNet series calculate == google paper's
    if args.comparison_model == 'MobileNetV3_large':
        flops_constraint = 217
        parameter_number_constraint = 5.4

    # proves MicroNet challenge doubles what google paper claimed
    elif args.comparison_model == 'MobileNetV2_1.4':
        flops_constraint = 585
        parameter_number_constraint = 6.9

    elif args.comparison_model == 'SinglePathOneShot':
        flops_constraint = 328
        parameter_number_constraint = 3.4

    # proves mine calculation == ShuffleNet series' == google paper's
    elif args.comparison_model == 'ShuffleNetV2+_medium':
        flops_constraint = 222
        parameter_number_constraint = 5.6

    else:
        raise ValueError("Unrecognized comparison model: {}".format(comparison_model))

    flop_score = flops / flops_constraint
    model_size_score = model_size / parameter_number_constraint

    return flops, model_size, flop_score, model_size_score


def get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask,
                 acc_top1=mx.metric.Accuracy(), acc_top5=mx.metric.TopKAccuracy(5),
                 ctx=[mx.cpu()], dtype='float32'):
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return top1


def set_nas_bn(net, inference_update_stat=False):
    if isinstance(net, NasBatchNorm):
        net.inference_update_stat = inference_update_stat
    elif len(net._children) != 0:
        for k, v in net._children.items():
            set_nas_bn(v,inference_update_stat=inference_update_stat)
    else:
        return


def update_bn(net, batch_fn, train_data, block_choices, full_channel_mask,
              ctx=[mx.cpu()], dtype='float32', batch_size=256, update_bn_images=20000):
    train_data.reset()
    net.cast('float16')
    net.load_parameters('./params/ShuffleNasOneshot-imagenet-supernet.params')
    net.cast('float32')
    set_nas_bn(net, inference_update_stat=True)
    for i, batch in enumerate(train_data):
        if (i + 1) * batch_size * len(ctx) >= update_bn_images:
            break
        data, _ = batch_fn(batch, ctx)
        _ = [net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
    set_nas_bn(net, inference_update_stat=False)


def update_log(elem, logger=None):
    """
    Print log
    Args:
        elem: a tuple of (overall_score, accuracy, norm_score, flops, model_size, block_choice, channel_choice)
        logger:
    """
    if logger:
        logger.info('-' * 40)
        logger.info("Overall score:     {}".format(elem[0]))
        logger.info("Val accuracy:      {}".format(elem[1]))
        logger.info("Model normalized score: {}.".format(elem[2]))
        logger.info('Flops:             {} MFLOPS'.format(elem[3]))
        logger.info('# parameters:      {} M'.format(elem[4]))
        logger.info("Block choices:     {}".format(elem[5]))
        logger.info("Channel choices:   {}".format(elem[6]))
    else:
        print('-' * 40)
        print("Overall score:     {}".format(elem[0]))
        print("Val accuracy:      {}".format(elem[1]))
        print("Model normalized score: {}.".format(elem[2]))
        print('Flops:             {} MFLOPS'.format(elem[3]))
        print('# parameters:      {} M'.format(elem[4]))
        print("Block choices:     {}".format(elem[5]))
        print("Channel choices:   {}".format(elem[6]))


class TopKHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem[0] > topk_small[0]:
                heapq.heapreplace(self.data, elem)

    def topk(self):
        return reversed([heapq.heappop(self.data) for _ in range(len(self.data))])


class Evolver():
    """ Class that implements genetic algorithm for supernet selection. """

    def __init__(self, net, train_data, val_data, batch_fn, param_dict,
                 dtype='float32', ctx=[mx.cpu()], comparison_model='SinglePathOneShot',
                 update_bn_images=20000, search_iters=50, batch_size=256,
                 population_size=500, retain_length=100, random_select=0.1, mutate_chance=0.1):

        self.net = net
        self.train_data = train_data
        self.val_data = val_data
        self.batch_fn = batch_fn
        self.param_dict = param_dict
        self.dtype = dtype
        self.ctx = ctx
        self.comparision_model = comparison_model
        self.update_bn_images = update_bn_images
        self.search_iters = search_iters
        self.batch_size = batch_size
        self.population_size = population_size
        self.retain_length = retain_length
        self.random_select = random_select
        self.mutate_chance = mutate_chance

    def create_population(self):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the size of the population
        Returns:
            (list): Population of random networks
        """
        population = []
        while len(population) < self.population_size:
            # Create a random network.
            instance = {}
            for param_name in self.param_dict:
                instance[param_name] = [random.choice(self.param_dict[param_name]) for _ in range(20)]

            block_choices = nd.array(instance['block']).astype(self.dtype, copy=False)
            channel_choices = instance['channel']
            flops, model_size, flop_score, model_size_score = \
                get_flop_param_score(block_choices, channel_choices, comparison_model='SinglePathOneShot')

            combined_score = 0.5 * flop_score + 0.5 * model_size_score
            if flop_score > FLOP_MAX:
                print("[SKIPPED] Current model normalized score: {}.".format(combined_score))
                print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
                print("[SKIPPED] Channel choices:   {}".format(channel_choices))
                print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
                print('[SKIPPED] # parameters:      {} M'.format(model_size))
                continue
            if model_size_score > PARAM_MAX:
                print("[SKIPPED] Current model normalized score: {}.".format(combined_score))
                print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
                print("[SKIPPED] Channel choices:   {}".format(channel_choices))
                print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
                print('[SKIPPED] # parameters:      {} M'.format(model_size))
                continue
            print("Population size + 1, total {}, with normalized score: {}".format(len(population) + 1, combined_score))
            # Add the network to our population.
            instance['flops'] = flops
            instance['model_size'] = model_size
            instance['score'] = combined_score
            population.append(instance)

        return population

    def fitness(self, block_choices, channel_choice):
        """ Return the accuracy, which is our second fitness function. """
        # get block choices
        block_choices = nd.array(block_choices).astype(self.dtype, copy=False)

        # get channel mask
        channel_mask = []
        global_max_length = int(self.net.stage_out_channels[-1] // 2 * self.net.candidate_scales[-1])
        for i in range(len(self.net.stage_repeats)):
            for j in range(self.net.stage_repeats[i]):
                local_mask = [0] * global_max_length
                channel_choice_index = len(channel_mask)  # channel_choice index is equal to current channel_mask length
                channel_num = int(self.net.stage_out_channels[i] // 2 *
                                  self.net.candidate_scales[channel_choice[channel_choice_index]])
                local_mask[:channel_num] = [1] * channel_num
                channel_mask.append(local_mask)
        channel_mask = nd.array(channel_mask).astype(self.dtype, copy=False)

        # Update BN
        tic = time.time()
        update_bn(self.net, self.batch_fn, self.train_data, block_choices, channel_mask, ctx=self.ctx, dtype=self.dtype,
                  batch_size=self.batch_size, update_bn_images=self.update_bn_images)
        print("BN statistics updated. Time used: {}".format(time.time() - tic))

        # get accuracy
        tic = time.time()
        top1 = get_accuracy(self.net, self.val_data, self.batch_fn, block_choices, channel_mask,
                            ctx=self.ctx, dtype=self.dtype)
        print("Validation accuracy evaluated. Time used: {}. Val acc: {}".format(time.time() - tic, top1))

        return top1

    def breed(self, mother, father):
        """ Make two children.
        Args:
            mother (dict): Network parameters
            father (dict): Network parameters
        Returns:
            (list): Two network objects
        """
        children = []
        for _ in range(2):

            child = {}

            # Crossover: loop through the parameters and pick params for the kid.
            for param_name in self.param_dict.keys():
                child[param_name] = [0] * len(father[param_name])
                for i in range(len(father[param_name])):
                    child[param_name][i] = random.choice([mother[param_name][i], father[param_name][i]])

                    # Mutation: randomly mutate some of the children.
                    if self.mutate_chance > random.random():
                        child[param_name][i] = random.choice(self.param_dict[param_name])

            children.append(child)

        return children

    def evolve(self, population, topk_items, logger=None):
        """ Evolve a population of networks.
        Args:
            population: A list of network parameters
            retain_length: How many items to keep after fitness
            topk_items: the heap to store top k items
        Return:
            A list of the evolved population of networks
        """

        # fitness
        for person in population:
            if 'acc' not in person.keys():
                person['acc'] = self.fitness(person['block'], person['channel'])
                net_obj = (SCORE_ACC_RATIO * person['score'] + person['acc'],
                           person['acc'], person['score'], person['flops'], person['model_size'],
                           copy.deepcopy(person['block']), copy.deepcopy(person['channel']))
                topk_items.push(net_obj)
                update_log(net_obj, logger)
        population = population.sort(key=lambda x: SCORE_ACC_RATIO * x['score'] + x['acc'], reverse=True)

        # The parents are every network we want to keep.
        parents = population[:self.retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in population[self.retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)

            # Assuming they aren't the same network...
            if male != female:
                male = parents[male]
                female = parents[female]

                # Breed them.
                babies = self.breed(male, female)

                # Add the children one at a time.
                for baby in babies:
                    # Don't grow larger than desired length.
                    if len(children) >= desired_length:
                        break

                    block_choices = nd.array(baby['block']).astype(self.dtype, copy=False)
                    channel_choices = baby['channel']
                    flops, model_size, flop_score, model_size_score = \
                        get_flop_param_score(block_choices, channel_choices, comparison_model='SinglePathOneShot')

                    combined_score = 0.5 * flop_score + 0.5 * model_size_score
                    if flop_score > FLOP_MAX:
                        print("[SKIPPED] Current model normalized score: {}.".format(combined_score))
                        print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
                        print("[SKIPPED] Channel choices:   {}".format(channel_choices))
                        print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
                        print('[SKIPPED] # parameters:      {} M'.format(model_size))
                        continue
                    if model_size_score > PARAM_MAX:
                        print("[SKIPPED] Current model normalized score: {}.".format(combined_score))
                        print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
                        print("[SKIPPED] Channel choices:   {}".format(channel_choices))
                        print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
                        print('[SKIPPED] # parameters:      {} M'.format(model_size))
                        continue
                    print("Children size + 1, size {}, with normalized score: {}".format(len(children) + 1, combined_score))
                    # Add the network to our population.
                    baby['flops'] = flops
                    baby['model_size'] = model_size
                    baby['score'] = combined_score
                    children.append(baby)

        parents.extend(children)
        return parents


def random_search(net, dtype='float32', logger=None, ctx=[mx.cpu()], comparison_model='SinglePathOneShot',
                  update_bn_images=20000, search_iters=50000, batch_size=128, topk=3, **data_kwargs):
    """ Search within the pre-trained supernet. """

    train_data, val_data, batch_fn = get_data(batch_size=batch_size, num_gpus=len(ctx), **data_kwargs)

    topk_nets = TopKHeap(topk)  # a list of tuple (acc, score, flops, model_size, block_choices, channel_choices)
    net_obj = None

    for i in range(search_iters):
        print("\nSearching iter: {}".format(i))

        # get selected blocks and channels
        block_choices = net.random_block_choices(select_predefined_block=False, dtype=dtype)
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False, dtype=dtype)

        # calculate
        flops, model_size, flop_score, model_size_score = \
            get_flop_param_score(block_choices, channel_choices, comparison_model)
        combined_score = 0.5 * flop_score + 0.5 * model_size_score
        if combined_score > 1:
            print("[SKIPPED] Current model normalized score: {}.".format(combined_score))
            print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
            print("[SKIPPED] Channel choices:   {}".format(channel_choices))
            print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
            print('[SKIPPED] # parameters:      {} M'.format(model_size))
            continue

        print("Target size + 1, with normalized score: {}".format(combined_score))
        # update BN
        tic = time.time()
        update_bn(net, batch_fn, train_data, block_choices, full_channel_mask, ctx=ctx, dtype=dtype,
                  batch_size=batch_size, update_bn_images=update_bn_images)
        print("BN statistics updated. Time used: {}".format(time.time() - tic))

        # get validation accuracy
        tic = time.time()
        val_acc = get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask, ctx=ctx)
        print("Validation accuracy evaluated. Time used: {}".format(time.time() - tic))

        # update the list of best networks
        # net_obj is (accuracy, norm_score, flops, model_size, block_choice, channel_choice)
        net_obj = (val_acc, flop_score + model_size_score, flops, model_size,
                copy.deepcopy(block_choices.asnumpy()), copy.deepcopy(channel_choices))
        topk_nets.push(net_obj)
        update_log(net_obj, logger)

    # summary
    if logger:
        logger.info('-' * 40)
        logger.info('Best models:')
    else:
        print('-' * 40)
        print('Best models:')
    for net_obj in topk_nets.data:
        update_log(net_obj, logger)


def genetic_search(net, dtype='float32', logger=None, ctx=[mx.cpu()], comparison_model='SinglePathOneShot',
                   update_bn_images=20000, search_iters=50000, batch_size=128, topk=3,
                   population_size=500, retain_length=100, random_select=0.1, mutate_chance=0.1, **data_kwargs):

    # get data
    train_data, val_data, batch_fn = get_data(batch_size=batch_size, num_gpus=len(ctx), **data_kwargs)

    topk_nets = TopKHeap(topk)  # a list of tuple (acc, score, flops, model_size, block_choices, channel_choices)

    # set channel and block value list
    param_dict = {'channel': [2, 3, 4, 5, 6, 7, 8, 9],
                  'block': [0, 1, 2, 3]}

    # evolution
    evolver = Evolver(net, train_data, val_data, batch_fn, param_dict,
                      dtype=dtype,
                      ctx=ctx,
                      comparison_model=comparison_model,
                      update_bn_images=update_bn_images,
                      search_iters=search_iters,
                      batch_size=batch_size,
                      population_size=population_size,
                      retain_length=retain_length,
                      random_select=random_select,
                      mutate_chance=mutate_chance)
    population = evolver.create_population()

    for i in range(search_iters):
        print("\nSearching iter: {}".format(i))
        logger.info("\nSearching iter: {}".format(i))
        population = evolver.evolve(population, topk_nets, logger)

    # summary
    if logger:
        logger.info('-' * 40)
        logger.info('Best models:')
    else:
        print('-' * 40)
        print('Best models:')
    for net_obj in topk_nets.data:
        update_log(net_obj, logger)


def main():
    context = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    net = get_shufflenas_oneshot(use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling)
    # net = get_shufflenas_oneshot(use_se=True, last_conv_after_pooling=True)
    net.cast(args.dtype)
    # net.cast('float16')
    net.load_parameters(args.supernet_params, ctx=context)
    net.cast('float32')
    print(net)

    filehandler = logging.FileHandler('./search_supernet_{}.log'.format(args.comparison_model))
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(args)

    data_kwargs = {
        "rec_train": args.rec_train,
        "rec_train_idx": args.rec_train_idx,
        "rec_val": args.rec_val,
        "rec_val_idx": args.rec_val_idx,
        "input_size": args.input_size,
        "crop_ratio": args.crop_ratio,
        "num_workers": args.num_workers,
        "shuffle_train": args.shuffle_train
    }

    if args.search_mode == 'random':
        random_search(net,
                      dtype='float32',
                      logger=logger,
                      ctx=context,
                      search_iters=100,
                      comparison_model=args.comparison_model,
                      update_bn_images=args.update_bn_images,
                      batch_size=args.batch_size,
                      topk=args.topk,
                      **data_kwargs
                      )
    elif args.search_mode == 'genetic':
        genetic_search(net,
                       dtype='float32',
                       logger=logger,
                       ctx=context,
                       search_iters=args.search_iters,
                       comparison_model=args.comparison_model,
                       update_bn_images=args.update_bn_images,
                       batch_size=args.batch_size,
                       topk=args.topk,
                       population_size=args.population_size,
                       retain_length=args.retain_length,
                       random_select=args.random_select,
                       mutate_chance=args.mutate_chance,
                       **data_kwargs)
    else:
        raise ValueError("Unrecognized search mode: {}".format(args.search_mode))


if __name__ == '__main__':
    args = parse_args()
    main()
