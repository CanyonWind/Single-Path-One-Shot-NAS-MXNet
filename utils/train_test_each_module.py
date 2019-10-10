import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import os
import logging
import sys

file_path = os.path.dirname(os.path.realpath(__file__))
upper_level = file_path[:file_path.rfind('/')]
sys.path.append(upper_level)
from oneshot_nas_blocks import *
from oneshot_nas_network import get_shufflenas_oneshot


def transform(data, label):
    return mx.nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)


train_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST(train=True, transform=transform), batch_size=128, shuffle=True)

validation_data = gluon.data.DataLoader(
    gluon.data.vision.FashionMNIST(train=False, transform=transform), batch_size=128, shuffle=False)


def init(net, optimizer='sgd', learning_rate=0.1, weight_decay=1e-6, ctx=mx.gpu()):
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(),
                            optimizer,
                            {'learning_rate': learning_rate, 'wd': weight_decay})
    return trainer


def accuracy(data_iterator, net, ctx=mx.gpu()):
    acc = mx.metric.Accuracy()
    for (data, label) in data_iterator:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = mx.nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


def plot_accuracies(training_accuracies, validation_accuracies):
    epochs = len(training_accuracies)
    plt.clf()
    fig, ax = plt.subplots()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    train_plot, = ax.plot(range(epochs), training_accuracies, label="Training accuracy")
    validation_plot, = ax.plot(range(epochs), validation_accuracies, label="Validation accuracy")
    plt.legend(handles=[train_plot,validation_plot])
    plt.xticks(np.arange(0, epochs, 5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
    plt.show()


def train(net, trainer, train_data, validation_data, epochs, ctx=mx.gpu()):
    training_accuracies = []
    validation_accuracies = []
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    for e in range(epochs):
        tic = time.time()
        for (data, label) in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                loss.backward()
            trainer.step(data.shape[0])
        toc = time.time()
        train_accuracy = accuracy(train_data, net)
        training_accuracies.append(train_accuracy)
        validation_accuracy = accuracy(validation_data, net)
        validation_accuracies.append(validation_accuracy)
        # print("Epoch#%d Time=%.2f Training=%.4f Validation=%.4f Diff=%.4f"
        #       % (e, toc-tic, train_accuracy, validation_accuracy, train_accuracy-validation_accuracy))
    return training_accuracies, validation_accuracies


def get_block(block_mode='conv', act_mode='relu', use_se=False):
    if block_mode == 'just-conv':
        net = gluon.nn.HybridSequential()
        net.add(
            nn.Conv2D(16, kernel_size=3, strides=2,
                      padding=1, use_bias=False, prefix='1st_conv_'),
            nn.BatchNorm(momentum=0.1),
            Activation(act_mode)
        )
        if use_se:
            net.add(SE(16))
        net.add(
            nn.Conv2D(32, in_channels=16, kernel_size=3, strides=2,
                      padding=1, use_bias=False, prefix='2nd_conv_'),
            nn.BatchNorm(momentum=0.1),
            Activation(act_mode)
        )
    elif block_mode == 'SNB':
        net = gluon.nn.HybridSequential()
        net.add(
            nn.Conv2D(16, kernel_size=3, strides=2,
                      padding=1, use_bias=False, prefix='1st_conv_'),
            nn.BatchNorm(momentum=0.1),
            Activation(act_mode)
        )
        if use_se:
            net.add(SE(16))
        net.add(
            ShuffleNetBlock(16, 32, 16, bn=nn.BatchNorm,
                            block_mode='ShuffleNetV2', ksize=3, stride=1,
                            use_se=use_se, act_name=act_mode)
        )
    elif block_mode == 'SNB-x':
        net = gluon.nn.HybridSequential()
        net.add(
            nn.Conv2D(16, kernel_size=3, strides=2,
                      padding=1, use_bias=False, prefix='1st_conv_'),
            nn.BatchNorm(momentum=0.1),
            Activation(act_mode)
        )
        if use_se:
            net.add(SE(16))
        net.add(
            ShuffleNetBlock(16, 32, 16, bn=nn.BatchNorm,
                            block_mode='ShuffleXception', ksize=3, stride=1,
                            use_se=use_se, act_name=act_mode)
        )
    elif block_mode == 'ShuffleNas_fixArch':
        architecture = [0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 1, 1, 0, 0, 1, 2, 2, 0, 2, 0]
        scale_ids = [8, 6, 5, 7, 6, 7, 3, 4, 2, 4, 2, 3, 4, 3, 6, 7, 5, 3, 4, 6]
        net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                     use_se=True, last_conv_after_pooling=True)
    else:
        raise ValueError("Unrecognized mode: {}".format(block_mode))

    if block_mode != 'ShuffleNas_fixArch':
        net.add(nn.GlobalAvgPool2D(),
                nn.Conv2D(10, in_channels=32, kernel_size=1, strides=1,
                          padding=0, use_bias=True),
                nn.Flatten()
                )
    else:
        net.output = nn.HybridSequential(prefix='output_')
        with net.output.name_scope():
            net.output.add(
                nn.Conv2D(10, in_channels=1024, kernel_size=1, strides=1,
                          padding=0, use_bias=True),
                nn.Flatten()
            )
    return net


def verify_mkl():
    epochs = 1
    block_candidates = ['just-conv',
                        'just-conv-swish',
                        'just-conv-sigmoid',
                        'just-conv-swish-sigmoid',
                        'SNB',
                        'SNB-x',
                        'ShuffleNas_fixArch']
    block_candidates = block_candidates[-1:]
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    for block_name in block_candidates:
        if 'just-conv' in block_name:
            block_mode = 'just-conv'
        elif 'SNB' in block_name:
            block_mode = 'SNB'
        elif 'SNB-x' in block_name:
            block_mode = 'SNB-x'
        elif 'ShuffleNas_fixArch' in block_name:
            block_mode = 'ShuffleNas_fixArch'
        else:
            raise ValueError('Unrecognized block mode: {}'.format(block_name))
        act_mode = 'swish' if 'swish' in block_name else 'relu'
        use_se = True if 'sigmoid' in block_name else False

        net = get_block(block_mode=block_mode, act_mode=act_mode, use_se=use_se)
        trainer = init(net, optimizer='sgd', learning_rate=0.01)
        net.hybridize()
        train_acc, val_acc = train(net, trainer, train_data, validation_data, epochs)
        logger.info('\nVerifying {}: \nGPU trained train_acc: {}, val_acc: {}'.
                    format(block_name, train_acc[-1], val_acc[-1]))

        # Save model
        symbol_forder = './symbol-block-FashionM'
        if not os.path.exists(symbol_forder):
            os.makedirs(symbol_forder)
        param_forder = './param-block-FashionM'
        if not os.path.exists(param_forder):
            os.makedirs(param_forder)
        net.save_parameters(os.path.join(param_forder, 'Block-test-{}-000{}.params'.format(block_name, epochs)))
        net.export(os.path.join(symbol_forder, 'Block-test-{}'.format(block_name)), epoch=epochs)

        # Evaluate on CPU
        net_cpu = get_block(block_mode=block_mode, act_mode=act_mode, use_se=use_se)
        net_cpu.load_parameters(os.path.join(param_forder, 'Block-test-{}-000{}.params'.format(block_name, epochs)),
                                ctx=mx.cpu())
        cpu_train_accuracy = accuracy(train_data, net_cpu, ctx=mx.cpu())
        cpu_validation_accuracy = accuracy(validation_data, net_cpu, ctx=mx.cpu())
        logger.info('CPU inference train_acc: {}, val_acc: {}'.
                    format(cpu_train_accuracy, cpu_validation_accuracy))


def main():
    verify_mkl()


if __name__ == '__main__':
    main()
