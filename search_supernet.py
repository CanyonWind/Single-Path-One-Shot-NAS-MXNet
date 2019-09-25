import mxnet as mx
from mxnet import gluon
from mxnet import nd
import os
import copy
import math
from itertools import count
import random

from oneshot_nas_network import get_shufflenas_oneshot
from calculate_flops import get_flops
from oneshot_nas_blocks import NasBatchNorm
import heapq


def generate_random_data_label(ctx=mx.gpu(0)):
    data = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224), ctx=ctx)
    label = None
    return data, label


def get_data(rec_train='~/.mxnet/datasets/imagenet/rec/train.rec', 
             rec_train_idx='~/.mxnet/datasets/imagenet/rec/train.idx',
             rec_val='~/.mxnet/datasets/imagenet/rec/val.rec', 
             rec_val_idx='~/.mxnet/datasets/imagenet/rec/val.idx',
             input_size=224, crop_ratio=0.875, num_workers=8, batch_size=256, num_gpus=0):
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
        shuffle=False,
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
        shuffle=False,
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


def get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask,
                 acc_top1=None, acc_top5=None, ctx=[mx.cpu()], dtype='float32'):
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
    print("Updating BN statistics...")
    set_nas_bn(net, inference_update_stat=True)
    for i, batch in enumerate(train_data):
        if (i + 1) * batch_size * len(ctx) >= update_bn_images:
            break
        data, _ = batch_fn(batch, ctx)
        _ = [net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
    set_nas_bn(net, inference_update_stat=False)


class TopKHeap(object):
    def __init__(self, k, cnt):
        self.k = k
        self.data = []
        self.cnt = cnt

    def push(self, elem, cnt):
        if len(self.data) < self.k:
            heapq.heappush(self.data, (elem['acc'], next(cnt), elem))
        else:
            topk_small = self.data[0]
            if elem['acc'] > topk_small[0]:
                heapq.heapreplace(self.data, (elem['acc'], next(cnt), elem))

    def topk(self):
        # element is in format of (accuracy, second_order, net_selected_params)
        return [x[2] for x in reversed([heapq.heappop(self.data) for _ in range(len(self.data))])]


class Evolver():
    """ Class that implements genetic algorithm for supernet selection. """

    def __init__(self, net, train_data, val_data, batch_fn, param_dict, update_bn_images=20000,
                 search_iters=1000, random_select=0.1, mutate_chance=0.1,
                 num_gpus=0, dtype='float32', batch_size=256, flops_constraint=585, parameter_number_constraint=6.9):
        """
        Args:
            param_dict (dict):     Possible network paremters  {'block': [] * 4, 'channel': [] * 10}
            update_bn_images:
            search_iters:
            retain (float):        Percentage of population to retain after each generation
            random_select (float): Probability of a rejected network remaining in the population
            mutate_chance (float): Probability a network will be randomly mutated
            num_gpus:
            dtype:
            batch_size:
            flops_constraint:      MobileNetV3-Large-1.0 [219M],
                                   MicroNet Challenge standard MobileNetV2-1.4 [585M]
            param_num_constraint:  MobileNetV3-Large-1.0 [5.4M],
                                   MicroNet Challenge standard MobileNetV2-1.4 [6.9M]
        """
        self.net = net
        self.train_data = train_data
        self.val_data = val_data
        self.batch_fn = batch_fn
        self.param_dict = param_dict
        self.update_bn_images = update_bn_images
        self.search_iters = search_iters
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.num_gpus = num_gpus
        self.dtype = dtype
        self.batch_size = batch_size
        self.flops_constraint = flops_constraint
        self.parameter_number_constraint = parameter_number_constraint

    def create_population(self, count=2000):
        """Create a population of random networks.
        Args:
            count (int): Number of networks to generate, aka the
                size of the population
        Returns:
            (list): Population of random networks
        """
        population = []
        for _ in range(0, count):
            # Create a random network.
            instance = {}
            for param_name in self.param_dict:
                instance[param_name] = [random.choice(self.param_dict[param_name]) for _ in range(20)]

            # Add the network to our population.
            population.append(instance)

        return population

    def fitness_1st_stage(self, block_choices, channel_choices):
        """ Return the flop score + num_param score, which is our first fitness function. """
        # build fix_arch network and calculate flop
        fixarch_net = get_shufflenas_oneshot(block_choices, channel_choices)
        fixarch_net._initialize()
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        fixarch_net.hybridize()
        dummy_data = nd.ones([1, 3, 224, 224])
        fixarch_net(dummy_data)
        fixarch_net.export("./symbols/ShuffleNas_fixArch", epoch=1)
        flops, model_size = get_flops()  # both in Millions
        normalized_score = flops / self.flops_constraint + model_size / self.parameter_number_constraint
        if normalized_score >= 1.1:
            print("[SKIPPED] Current model normalized score: {}.".format(normalized_score))
            print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
            print("[SKIPPED] Channel choices:   {}".format(channel_choices))
            print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
            print('[SKIPPED] # parameters:      {} M'.format(model_size))
        return normalized_score

    def fitness_2nd_stage(self, block_choices, full_channel_mask, acc_top1=mx.metric.Accuracy(),
                          acc_top5=mx.metric.TopKAccuracy(5), dtype='float32'):
        """ Return the accuracy, which is our second fitness function. """
        # Update BN
        ctx = [mx.gpu(i) for i in range(self.num_gpus)] if self.num_gpus > 0 else [mx.cpu()]
        update_bn(self.net, block_choices, full_channel_mask, self.batch_fn, self.train_data, ctx=ctx, dtype=self.dtype,
                  batch_size=self.batch_size, update_bn_images=self.update_bn_images)

        self.val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        print("BN statistics updated.")

        for i, batch in enumerate(self.val_data):
            # TODO: remove debug code
            if i >= 1:
                break
            # End debug
            data, label = self.batch_fn(batch, ctx)
            outputs = [self.net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
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
                child[param_name] = [0] * len(self.param_dict[param_name])
                for i in range(len(self.param_dict[param_name])):
                    child[param_name][i] = random.choice([mother[param_name][i], father[param_name][i]])

                    # Mutation: randomly mutate some of the children.
                    if self.mutate_chance > random.random():
                        child[param_name][i] = random.choice(self.param_dict[param_name])

            children.append(child)

        return children

    def evolve(self, population, retain_length1=1000, retain_length2=500, topk=3):
        """Evolve a population of networks.
        Args:
            population (list): A list of network parameters
        Returns:
            (list): The evolved population of networks
        """

        # fitness first stage -------------------------------
        selected1 = [(self.fitness_1st_stage(person['block'], person['channel']), person) for person in population]

        # Sort on the scores.
        selected1 = [x[1] for x in sorted(selected1, key=lambda x: x[0])]

        # The parents are every network we want to keep.
        parents = selected1[:retain_length1]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in selected1[retain_length1:]:
            if self.random_select > random.random():
                parents.append(individual)

        # fitness second stage -------------------------------
        # TODO: get channel mask
        selected2 = [(self.fitness_2nd_stage(person['block'], person['channel']), person) for person in parents]

        # Sort on the scores.
        # TODO: Add accuracy
        selected2 = [x[1] for x in sorted(selected2, key=lambda x: x[0], reverse=True)]

        # The parents are every network we want to keep.
        parents = selected2[:retain_length2]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in selected2[retain_length2:]:
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
                    if len(children) < desired_length:
                        children.append(baby)

        parents.extend(children)
        return parents, selected2[:topk]


def random_search(net, search_iters=2000, update_bn_images=20000, dtype='float32', batch_size=256,
                    flops_constraint=585, parameter_number_constraint=6.9, ctx=[mx.cpu()]):
    """
    Search within the pre-trained supernet.
    :param net:
    :param search_iters:
    :param update_bn_images:
    :param dtype:
    :param batch_size:
    :param flops_constraint: MobileNetV3-Large-1.0 [219M], MicroNet Challenge standard MobileNetV2-1.4 [585M]
    :param parameter_number_constraint: MobileNetV3-Large-1.0 [5.4M], MicroNet Challenge standard MobileNetV2-1.4 [6.9M]
    :return:
    """
    # TODO: use a heapq here to store top-5 models
    train_data, val_data, batch_fn = get_data(num_gpus=len(ctx), batch_size=batch_size)
    best_acc, best_acc_flop, best_acc_size = 0, 0, 0
    best_block_choices = None
    best_channel_choices = None
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    for i in range(search_iters):
        print("\nSearching iter: {}".format(i))
        block_choices = net.random_block_choices(select_predefined_block=False, dtype=dtype)
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False, dtype=dtype)
        
        # build fix_arch network and calculate flop
        fixarch_net = get_shufflenas_oneshot(block_choices.asnumpy(), channel_choices)
        fixarch_net._initialize()
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        fixarch_net.hybridize()
        dummy_data = nd.ones([1, 3, 224, 224])
        fixarch_net(dummy_data)
        fixarch_net.export("./symbols/ShuffleNas_fixArch", epoch=1)
        flops, model_size = get_flops()  # both in Millions
        normalized_score = flops / flops_constraint + model_size / parameter_number_constraint
        if normalized_score >= 2:
            print("[SKIPPED] Current model normalized score: {}.".format(normalized_score))
            print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
            print("[SKIPPED] Channel choices:   {}".format(channel_choices))
            print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
            print('[SKIPPED] # parameters:      {} M'.format(model_size))
            continue

        # Update BN
        update_bn(net, batch_fn, train_data, block_choices, full_channel_mask, ctx=ctx, dtype=dtype,
                  batch_size=batch_size, update_bn_images=update_bn_images)
        print("BN statistics updated.")
        # Get validation accuracy
        val_acc = get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask,
                               acc_top1=acc_top1, acc_top5=acc_top5, ctx=ctx)
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_flop = flops
            best_acc_size = model_size
            best_block_choices = copy.deepcopy(block_choices.asnumpy())
            best_channel_choices = copy.deepcopy(channel_choices)
        print('-' * 40)
        print("Current model normalized score: {}.".format(normalized_score))
        print("Val accuracy:      {}".format(val_acc))
        print("Block choices:     {}".format(block_choices.asnumpy()))
        print("Channel choices:   {}".format(channel_choices))
        print('Flops:             {} MFLOPS'.format(flops))
        print('# parameters:      {} M'.format(model_size))
    
    print('-' * 40)
    print("Current model normalized score: {}.".
          format(best_acc_flop / flops_constraint + best_acc_size / parameter_number_constraint))
    print("Best val accuracy:    {}".format(best_acc))
    print("Block choices:        {}".format(best_block_choices))
    print("Channel choices:      {}".format(best_channel_choices))
    print('Flops:                {} MFLOPS'.format(best_acc_flop))
    print('# parameters:         {} M'.format(best_acc_size))


def genetic_search(net, num_gpus=4, batch_size=256, ctx=[mx.cpu()]):
    # get data
    train_data, val_data, batch_fn = get_data(num_gpus=num_gpus, batch_size=batch_size)

    # set channel and block value list
    param_dict = {'channel': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  'block': [0, 1, 2, 3]}

    # keep global topk params
    cnt = count()  # for heap
    global_topk = TopKHeap(3, cnt)

    # evolution
    num_iter = 50
    evolver = Evolver(net, train_data, val_data, batch_fn, param_dict)
    population = evolver.create_population()

    for i in range(num_iter):
        print("\nSearching iter: {}".format(i))
        population, local_topk = evolver.evolve(population)
        for elem in local_topk:
            global_topk.push(elem, cnt)

    print(global_topk)
    return None


def main(num_gpus=4, supernet_params='./params/ShuffleNasOneshot-imagenet-supernet.params',
         dtype='float32', batch_size=256, search_mode='random'):
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = get_shufflenas_oneshot(use_se=True, last_conv_after_pooling=True)
    net.load_parameters(supernet_params, ctx=context)
    print(net)
    if search_mode == 'random':
        random_search(net, search_iters=2000, dtype=dtype,
                        batch_size=batch_size, update_bn_images=20000, ctx=context)
    elif search_mode == 'genetic':
        genetic_search()
    else:
        raise ValueError("Unrecognized search mode: {}".format(search_mode))


if __name__ == '__main__':
    main(1)

