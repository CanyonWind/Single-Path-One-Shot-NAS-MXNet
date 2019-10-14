import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os
from sys import maxsize
import numpy as np
import random
import sys
sys.path.append('..')
sys.path.append('../utils')

from oneshot_nas_network import get_shufflenas_oneshot
from calculate_flops import get_flops


def convert():
    architecture = [0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0]
    scale_ids = [8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6]
    net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids, use_se=True,
                                 last_conv_after_pooling=True,  shuffle_by_conv=True)

    # load params
    net._initialize(force_reinit=True, dtype='float32')
    net.cast('float16')
    net.load_parameters('../models/oneshot-s+model-0000.params', allow_missing=True)
    net.cast('float32')

    # save both gluon model and symbols
    test_data = nd.ones([5, 3, 224, 224], dtype='float32')
    _ = net(test_data)
    net.summary(test_data)
    net.hybridize()

    if not os.path.exists('./symbols'):
        os.makedirs('./symbols')
    if not os.path.exists('./params'):
        os.makedirs('./params')
    net.cast('float16')
    net.load_parameters('../models/oneshot-s+model-0000.params', allow_missing=True)
    net.cast('float32')
    net.hybridize()
    net(test_data)
    net.save_parameters('./params/ShuffleNas_fixArch_shuffleByConv-0000.params')
    net.export("./symbols/ShuffleNas_fixArch_shuffleByConv", epoch=0)
    flops, model_size = get_flops(symbol_path='./symbols/ShuffleNas_fixArch_shuffleByConv-symbol.json')
    print("Last conv after pooling: {}, use se: {}".format(True, True))
    print("FLOPS: {}M, # parameters: {}M".format(flops, model_size))


if __name__ == '__main__':
    convert()
