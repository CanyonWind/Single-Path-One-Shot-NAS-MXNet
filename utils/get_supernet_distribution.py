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
    parser.add_argument('--sample-count', type=int, default=217,
                        help='How many subnet to be sampled')
    opts = parser.parse_args()
    return opts


def get_flop_param_score(block_choices, channel_choices, comparison_model='SinglePathOneShot'):
    """ Return the flops and num of params """
    # build fix_arch network and calculate flop
    fixarch_net = get_shufflenas_oneshot(block_choices, channel_choices,
                                         use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling,
                                         channels_layout=args.channels_layout)
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
    print(net)
    flop_list = []
    param_list = []
    for _ in range(args.sample_count):
        block_choices = net.random_block_choices(select_predefined_block=False)
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False)
        flops, model_size, _, _ = \
                get_flop_param_score(block_choices, channel_choices)
        flop_list.append(flops)
        param_list.append(model_size)

    # plot
    plt.style.use("ggplot")
    plt.figure()
    axes = plt.gca()
    # axes.set_xlim([-0.005, 0.5])
    # axes.set_ylim([0.15, 0.7])
    plt.scatter(flop_list, param_list, alpha=0.8, c='steelblue', s=50, label='subnet')
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
