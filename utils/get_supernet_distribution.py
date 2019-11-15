import os, random
import sys
import argparse
import json
import matplotlib.pyplot as plt
from mxnet import nd
from calculate_flops import get_flops
from lookup_table import get_flop_params as lookup_flop_params

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = dir_path[:dir_path.rfind('/')]
sys.path.append(parent_path)
from oneshot_nas_network import get_shufflenas_oneshot

PARAM_DICT = {'channel': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
              'block': [0, 1, 2, 3]}
epoch_delay_early = {0: 0,  # 8
                     1: 1, 2: 1,  # 7
                     3: 2, 4: 2, 5: 2,  # 6
                     6: 3, 7: 3, 8: 3, 9: 3,  # 5
                     10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
                     15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
                     21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 27: 6, 28: 6,
                     29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7}
epoch_delay_late = {0: 0,
                    1: 1,
                    2: 2,
                    3: 3,
                    4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                    6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
                    9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
                    13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                    18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales
FAST_LOOKUP = True


def parse_args():
    parser = argparse.ArgumentParser(description='Get supernet flop/param distribution.')
    parser.add_argument('--use-se', action='store_false',
                        help='use SE layers or not in resnext and ShuffleNas')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='whether to follow MobileNet V3 last conv after pooling style')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    parser.add_argument('--sample-count', type=int, default=10000,
                        help='How many subnet to be sampled')
    parser.add_argument('--compare', action='store_true',
                        help='whether to plot the regular supernet and se supernet together to compare.')
    parser.add_argument('--use-evolution', type=bool, default=True,
                        help='whether to use evolution search or random selection')
    opts = parser.parse_args('')
    return opts


def get_flop_param_score(block_choices, channel_choices, comparison_model='SinglePathOneShot',
                         use_se=False, last_conv_after_pooling=False, channels_layout='OneShot'):
    """ Return the flops and num of params """
    # build fix_arch network and calculate flop
    fixarch_net = get_shufflenas_oneshot(block_choices, channel_choices,
                                         use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                         channels_layout=channels_layout)
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
    if comparison_model == 'MobileNetV3_large':
        flops_constraint = 217
        parameter_number_constraint = 5.4

    # proves MicroNet challenge doubles what google paper claimed
    elif comparison_model == 'MobileNetV2_1.4':
        flops_constraint = 585
        parameter_number_constraint = 6.9

    elif comparison_model == 'SinglePathOneShot':
        flops_constraint = 328
        parameter_number_constraint = 3.4

    # proves mine calculation == ShuffleNet series' == google paper's
    elif comparison_model == 'ShuffleNetV2+_medium':
        flops_constraint = 222
        parameter_number_constraint = 5.6

    else:
        raise ValueError("Unrecognized comparison model: {}".format(comparison_model))

    flop_score = flops / flops_constraint
    model_size_score = model_size / parameter_number_constraint

    return flops, model_size, flop_score, model_size_score


def get_distribution():
    net = get_shufflenas_oneshot(use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling,
                                 channels_layout=args.channels_layout)
    if args.compare:
        net = get_shufflenas_oneshot()
    print(net)
    # TODO: find out why argparser does not work.
    args.compare = True
    print(args)
    flop_list = []
    param_list = []
    se_flop_list = []
    se_param_list = []
    pool = []
    with open('../models/lookup_table_OneShot.json', 'r') as fp:
        lookup_table = json.load(fp)
    with open('../models/lookup_table_se_lastConvAfterPooling_OneShot.json', 'r') as fp:
        se_lookup_table = json.load(fp)

    for i in range(args.sample_count):
        candidate = dict()
        if not args.use_evolution or len(pool) < 10:
            _, block_choices = net.random_block_choices(select_predefined_block=False, return_choice_list=True)
            _, channel_choices = net.random_channel_mask(select_all_channels=False)

        elif len(pool) < 20:
            # randomly select parents from current pool
            mother = random.choice(pool)
            father = random.choice(pool)

            # make sure mother and father are different
            while father is mother:
                mother = random.choice(pool)

            # breed block choice
            block_choices = [0] * len(father['block'])
            for i in range(len(block_choices)):
                block_choices[i] = random.choice([mother['block'][i], father['block'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < 0.3:
                    block_choices[i] = random.choice(PARAM_DICT['block'])

            # breed channel choice
            channel_choices = [0] * len(father['channel'])
            for i in range(len(channel_choices)):
                channel_choices[i] = random.choice([mother['channel'][i], father['channel'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < 0.2:
                    channel_choices[i] = random.choice(PARAM_DICT['channel'])
            pool.pop(0)

        candidate['block'] = block_choices
        candidate['channel'] = channel_choices

        if args.compare:
            if FAST_LOOKUP:
                flops, model_size = lookup_flop_params(block_choices, channel_choices, lookup_table)
                se_flops, se_model_size = lookup_flop_params(block_choices, channel_choices, se_lookup_table)
            else:
                flops, model_size, _, _ = \
                    get_flop_param_score(block_choices, channel_choices, use_se=False, last_conv_after_pooling=False,
                                         channels_layout=args.channels_layout)

                se_flops, se_model_size, _, _ = \
                    get_flop_param_score(block_choices, channel_choices, use_se=True, last_conv_after_pooling=True,
                                         channels_layout=args.channels_layout)

            flop_list.append(flops)
            param_list.append(model_size)
            se_flop_list.append(se_flops)
            se_param_list.append(se_model_size)
        else:
            flops, model_size, _, _ = \
                get_flop_param_score(block_choices, channel_choices, use_se=args.use_se,
                                     last_conv_after_pooling=args.last_conv_after_pooling,
                                     channels_layout=args.channels_layout)

            flop_list.append(flops)
            param_list.append(model_size)

        if flops > 300 or model_size > 4.5 or not args.use_evolution:
            continue
        pool.append(candidate)

    # plot
    if args.compare:
        plt.style.use("ggplot")
        fig, (axs1, axs2) = plt.subplots(1, 2, sharex=True, sharey=True)
        axs1.scatter(flop_list, param_list, alpha=0.8, c='mediumaquamarine', s=50, label='subnet')
        axs1.set_title('Original SuperNet Distribution')
        axs1.legend(loc="lower right")
        axs2.scatter(se_flop_list, se_param_list, alpha=0.8, c='mediumaquamarine', s=50, label='subnet')
        axs2.set_title('SE-SuperNet Distribution')
        axs2.legend(loc="lower right")
        for axs in [axs1, axs2]:
            axs.set(xlabel='Flops', ylabel='Params amount')
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            axs.label_outer()

        plt.savefig('../images/supernet_flops_params_dist_compare.png')
        plt.show()
        plt.close()
    else:
        plt.style.use("ggplot")
        plt.figure()
        plt.scatter(flop_list, param_list, alpha=0.8, c='mediumaquamarine', s=50, label='subnet')
        plt.title('Flops Param Distribution')
        plt.xlabel("Flops")
        plt.ylabel("Params amount")
        plt.legend(loc="lower right")
        plt.savefig('../images/supernet_flops_params_dist.png')
        plt.show()
        plt.close()


if __name__ == '__main__':
    args = parse_args()
    get_distribution()
