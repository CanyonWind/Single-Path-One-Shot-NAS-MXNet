import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os
import copy

import random
from oneshot_nas_network import get_shufflenas_oneshot
from calculate_flops import get_flops


def update_bn(net):
    # TODO: add this before searching block and channel
    # https://www.d2l.ai/chapter_convolutional-modern/batch-norm.html#implementation-from-scratch
    return None


def generate_random_data_label():
    data = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224))
    label = None
    return data, label


def search_supernet(net, search_iters=2000, bn_iters=50000):
    # TODO: use a heapq here to store top-5 models
    best_acc = 0
    best_block_choices = None
    best_channel_choices = None
    for _ in range(search_iters):
        block_choices = net.random_block_choices(select_predefined_block=False, dtype='float32')
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False, dtype='float32')
        # Update BN
        for _ in range(bn_iters):
            data, _ = generate_random_data_label()
            net(data, block_choices, full_channel_mask)
        # Test validation accuracy 
        val_acc = random.uniform(0, 1)  # TODO: update according to y and y_hat
        if val_acc > best_acc:
            best_acc = val_acc
            best_block_choices = copy.deepcopy(block_choices)
            best_channel_choices = copy.deepcopy(channel_choices)
        # build fix_arch network and calculate flop
        fixarch_net = get_shufflenas_oneshot(block_choices.asnumpy(), channel_choices)
        fixarch_net.initialize()
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        fixarch_net.hybridize()
        dummy_data = nd.ones([1, 3, 224, 224])
        fixarch_net(dummy_data)
        fixarch_net.export("./symbols/ShuffleNas_fixArch", epoch=1)
        flops, model_size = get_flops()
        print('-' * 40)
        print("Val accuracy: {}".format(val_acc))
        print('flops: ', str(flops), ' MFLOPS')
        print('model size: ', str(model_size), ' MB')
    
    print('-' * 40)
    print("Best val accuracy: {}".format(best_acc))
    print("Best block choices: {}".format(best_block_choices.asnumpy()))
    print("Best channel choices: {}".format(best_channel_choices))
        


def main(num_gpus=4, supernet_params='./params/ShuffleNasOneshot-imagenet-supernet.params'):
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = get_shufflenas_oneshot()
    net.load_parameters(supernet_params, ctx=context)
    net.load_parameters(supernet_params, ctx=context)
    print(net)
    search_supernet(net, search_iters=10, bn_iters=1)


if __name__ == '__main__':
    main(0)

