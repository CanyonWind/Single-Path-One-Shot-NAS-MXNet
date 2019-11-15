import os
import sys
import argparse
import json
import pprint
import numpy as np
from mxnet import nd
from mxnet.gluon import nn
from calculate_flops import get_flops

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = dir_path[:dir_path.rfind('/')]
sys.path.append(parent_path)
from oneshot_nas_blocks import ShuffleNetBlock, Activation, SE


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a flop and param lookup table.')
    parser.add_argument('--use-se', action='store_false',
                        help='use SE layers or not in resnext and ShuffleNas')
    parser.add_argument('--last-conv-after-pooling', action='store_false',
                        help='whether to follow MobileNet V3 last conv after pooling style')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    opts = parser.parse_args('')
    return opts


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def get_block_flop(block, input_data):
    fixarch_net = block
    fixarch_net.initialize()
    if not os.path.exists('./symbols'):
        os.makedirs('./symbols')
    fixarch_net.hybridize()

    # calculate flops and num of params
    output_data = fixarch_net(input_data)
    fixarch_net.export("./symbols/ShuffleNas_block", epoch=1)

    data_shapes = [('data', input_data.shape)]
    flops, model_size = get_flops(symbol_path="./symbols/ShuffleNas_block-symbol.json", data_shapes=data_shapes)
    return flops, model_size, output_data


def generate_lookup_table():
    stage_repeats = [4, 4, 8, 4]
    if args.channels_layout == 'OneShot':
        stage_out_channels = [64, 160, 320, 640]
    elif args.channels_layout == 'ShuffleNetV2+':
        stage_out_channels = [48, 128, 256, 512]
    else:
        raise ValueError('Unrecognized channel layout')
    channel_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    first_conv_out_channel = 16
    input_size = 224
    last_conv_out_channel = 1024
    input_data = nd.ones((1, 3, input_size, input_size))
    lookup_table = dict()

    lookup_table['config'] = dict()
    lookup_table['config']['use_se'] = args.use_se
    lookup_table['config']['last_conv_after_pooling'] = args.last_conv_after_pooling
    lookup_table['config']['channels_layout'] = args.channels_layout
    lookup_table['config']['stage_repeats'] = stage_repeats
    lookup_table['config']['stage_out_channels'] = stage_out_channels
    lookup_table['config']['channel_scales'] = channel_scales
    lookup_table['config']['first_conv_out_channel'] = first_conv_out_channel
    lookup_table['config']['input_size'] = input_size
    lookup_table['config']['last_conv_out_channel'] = last_conv_out_channel

    # input block
    input_block = nn.HybridSequential()
    input_block.add(
        nn.Conv2D(first_conv_out_channel, in_channels=3, kernel_size=3, strides=2,
                  padding=1, use_bias=False, prefix='first_conv_'),
        nn.BatchNorm(momentum=0.1),
        Activation('hard_swish' if args.use_se else 'relu')
    )
    input_block_flops, input_block_model_size, input_data = get_block_flop(input_block, input_data)
    lookup_table['flops'] = dict()
    lookup_table['params'] = dict()
    lookup_table['flops']['input_block'] = input_block_flops
    lookup_table['params']['input_block'] = input_block_model_size

    # mid blocks
    lookup_table['flops']['nas_block'] = []  # 20 x 4 x 10, num_of_blocks x num_of_block_choices x num_of_channel_scales
    lookup_table['params']['nas_block'] = []
    input_channel = first_conv_out_channel
    for stage_id in range(len(stage_repeats)):
        numrepeat = stage_repeats[stage_id]
        output_channel = stage_out_channels[stage_id]

        if args.use_se:
            act_name = 'hard_swish' if stage_id >= 1 else 'relu'
            block_use_se = True if stage_id >= 2 else False
        else:
            act_name = 'relu'
            block_use_se = False
        # create repeated blocks for current stage
        for i in range(numrepeat):
            stride = 2 if i == 0 else 1
            output_data = None
            block_flops = [[0] * len(channel_scales) for _ in range(4)]
            block_params = [[0] * len(channel_scales) for _ in range(4)]
            for scale_i, scale in enumerate(channel_scales):
                # TODO: change back to make_divisible
                # mid_channel = make_divisible(int(output_channel // 2 * channel_scales[block_id]))
                mid_channel = int(output_channel // 2 * scale)
                # SNB 3x3
                snb3 = ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                       block_mode='ShuffleNetV2', ksize=3, stride=stride,
                                       use_se=block_use_se, act_name=act_name)
                snb3_block_flops, snb3_block_model_size, _ = get_block_flop(snb3, input_data)
                # SNB 5x5
                snb5 = ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                       block_mode='ShuffleNetV2', ksize=5, stride=stride,
                                       use_se=block_use_se, act_name=act_name)
                snb5_block_flops, snb5_block_model_size, _ = get_block_flop(snb5, input_data)
                # SNB 7x7
                snb7 = ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                       block_mode='ShuffleNetV2', ksize=7, stride=stride,
                                       use_se=block_use_se, act_name=act_name)
                snb7_block_flops, snb7_block_model_size, _ = get_block_flop(snb7, input_data)
                # SXB 3x3
                sxb3 = ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                       block_mode='ShuffleXception', ksize=3, stride=stride,
                                       use_se=block_use_se, act_name=act_name)
                sxb3_block_flops, sxb3_block_model_size, output_data = get_block_flop(sxb3, input_data)
                # fill the table
                block_flops[0][scale_i] = snb3_block_flops
                block_params[0][scale_i] = snb3_block_model_size
                block_flops[1][scale_i] = snb5_block_flops
                block_params[1][scale_i] = snb5_block_model_size
                block_flops[2][scale_i] = snb7_block_flops
                block_params[2][scale_i] = snb7_block_model_size
                block_flops[3][scale_i] = sxb3_block_flops
                block_params[3][scale_i] = sxb3_block_model_size

            lookup_table['flops']['nas_block'].append(block_flops)
            lookup_table['params']['nas_block'].append(block_params)
            input_data = output_data
            input_channel = output_channel

    # output block
    output_block = nn.HybridSequential()
    if args.last_conv_after_pooling:
        # MobileNet V3 approach
        output_block.add(
            nn.GlobalAvgPool2D(),
            # no last SE for MobileNet V3 style
            nn.Conv2D(last_conv_out_channel, kernel_size=1, strides=1,
                      padding=0, use_bias=True, prefix='conv_fc_'),
            # No bn for the conv after pooling
            Activation('hard_swish' if args.use_se else 'relu')
        )
    else:
        if args.use_se:
            # ShuffleNetV2+ approach
            output_block.add(
                nn.Conv2D(make_divisible(last_conv_out_channel * 0.75), in_channels=input_channel,
                          kernel_size=1, strides=1,
                          padding=0, use_bias=False, prefix='last_conv_'),
                nn.BatchNorm(momentum=0.1),
                Activation('hard_swish' if args.use_se else 'relu'),
                nn.GlobalAvgPool2D(),
                SE(make_divisible(last_conv_out_channel * 0.75)),
                nn.Conv2D(last_conv_out_channel, in_channels=make_divisible(last_conv_out_channel * 0.75),
                          kernel_size=1, strides=1,
                          padding=0, use_bias=True, prefix='conv_fc_'),
                # No bn for the conv after pooling
                Activation('hard_swish' if args.use_se else 'relu')
            )
        else:
            # original Oneshot Nas approach
            output_block.add(
                nn.Conv2D(last_conv_out_channel, in_channels=input_channel, kernel_size=1, strides=1,
                          padding=0, use_bias=False, prefix='last_conv_'),
                nn.BatchNorm(momentum=0.1),
                Activation('hard_swish' if args.use_se else 'relu'),
                nn.GlobalAvgPool2D()
            )

    # Dropout ratio follows ShuffleNetV2+ for se
    output_block.add(
        nn.Dropout(0.2 if args.use_se else 0.1),
        nn.Conv2D(1000, in_channels=last_conv_out_channel, kernel_size=1, strides=1,
                  padding=0, use_bias=True),
        nn.Flatten()
    )
    output_block_flops, output_block_model_size, output_data = get_block_flop(output_block, input_data)
    lookup_table['flops']['output_block'] = output_block_flops
    lookup_table['params']['output_block'] = output_block_model_size

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(lookup_table)

    with open('../models/lookup_table.json', 'w') as fp:
        json.dump(lookup_table, fp, indent=4)


def get_flop_params(block_list, channel_list, lookup_table):
    flops = lookup_table['flops']['input_block'] + lookup_table['flops']['output_block']
    params = lookup_table['params']['input_block'] + lookup_table['params']['output_block']

    for i in range(len(block_list)):
        block_choice = block_list[i]
        channel_choice = channel_list[i]
        flops += lookup_table['flops']['nas_block'][i][block_choice][channel_choice]
        params += lookup_table['params']['nas_block'][i][block_choice][channel_choice]

    return flops, params


if __name__ == '__main__':
    args = parse_args()
    print(args)
    # generate_lookup_table()
    with open('../models/lookup_table.json', 'r') as fp:
        lookup_table = json.load(fp)
    block_list = [0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0]
    channel_list = [8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6]
    flops, params = get_flop_params(block_list, channel_list, lookup_table)
    print('FLOPs: {}, params: {}'.format(flops, params))
