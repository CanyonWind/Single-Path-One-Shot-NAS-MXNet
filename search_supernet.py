import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os

import random
from oneshot_nas_network import get_shufflenas_oneshot


def update_bn(net):
    # TODO: add this before searching block and channel
    # https://www.d2l.ai/chapter_convolutional-modern/batch-norm.html#implementation-from-scratch
    return None


def generate_random_data_label():
    data = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224))
    label = None
    return data, label


def search_supernet(net):
    block_choices = net.random_block_choices(select_predefined_block=False, dtype='float32')
    full_channel_mask = net.random_channel_mask(select_all_channels=False, dtype='float32')
    val_acc = 0
    # BN update
    for i in range(50000):
        data, _ = generate_random_data_label()
        # Update BN
        net(data, block_choices, full_channel_mask)
    # Test validation accuracy
    for i in range(50000):
        data, y = generate_random_data_label()
        # Update BN
        y_hat = net(data, block_choices, full_channel_mask)
        val_acc = random.uniform(0, 1)  # TODO: update according to y and y_hat


    print("Searching")


def main(num_gpus=4, supernet_params='./params/ShuffleNasOneshot-imagenet-supernet.params'):
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = get_shufflenas_oneshot()
    net.load_parameters(supernet_params, ctx=context)
    net.load_parameters(supernet_params, ctx=context)
    print(net)
    search_supernet(net)


if __name__ == '__main__':
    main(0)

