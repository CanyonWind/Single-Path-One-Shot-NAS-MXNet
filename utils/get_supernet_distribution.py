import os
import sys
import argparse
import matplotlib.pyplot as plt
from mxnet import nd
from calculate_flops import get_flops

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = dir_path[:dir_path.rfind('/')]
sys.path.append(parent_path)
from oneshot_nas_network import get_shufflenas_oneshot


def parse_args():
    parser = argparse.ArgumentParser(description='Get supernet flop/param distribution.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext and ShuffleNas')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='whether to follow MobileNet V3 last conv after pooling style')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    parser.add_argument('--sample-count', type=int, default=1000,
                        help='How many subnet to be sampled')
    parser.add_argument('--compare', action='store_true',
                        help='whether to plot the regular supernet and se supernet together to compare.')
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
    for _ in range(args.sample_count):
        block_choices = net.random_block_choices(select_predefined_block=False)
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False)
        if args.compare:
            flops, model_size, _, _ = \
                get_flop_param_score(block_choices, channel_choices, use_se=False, last_conv_after_pooling=False,
                                     channels_layout=args.channels_layout)
            flop_list.append(flops)
            param_list.append(model_size)
            se_flops, se_model_size, _, _ = \
                get_flop_param_score(block_choices, channel_choices, use_se=True, last_conv_after_pooling=True,
                                     channels_layout=args.channels_layout)
            se_flop_list.append(se_flops)
            se_param_list.append(se_model_size)
        else:
            flops, model_size, _, _ = \
                get_flop_param_score(block_choices, channel_choices, use_se=args.use_se,
                                     last_conv_after_pooling=args.last_conv_after_pooling,
                                     channels_layout=args.channels_layout)
            flop_list.append(flops)
            param_list.append(model_size)

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
