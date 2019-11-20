import os, random
import sys
import argparse
import json
import matplotlib.pyplot as plt
from mxnet import nd
from calculate_flops import get_flops
from lookup_table import get_flop_params, load_lookup_table

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = dir_path[:dir_path.rfind('/')]
sys.path.append(parent_path)
from oneshot_nas_network import get_shufflenas_oneshot


def parse_args():
    parser = argparse.ArgumentParser(description='Get supernet flop/param distribution.')
    parser.add_argument('--use-se', action='store_false',
                        help='use SE layers or not in resnext and ShuffleNas')
    parser.add_argument('--last-conv-after-pooling', action='store_false',
                        help='whether to follow MobileNet V3 last conv after pooling style')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    parser.add_argument('--sample-count', type=int, default=200,
                        help='How many subnet to be sampled')
    parser.add_argument('--nas-root', type=str, default='..',
                        help='root path of nas repo')
    opts = parser.parse_args('')
    return opts


def get_shufflenas_flop_param(block_choices, channel_choices, comparison_model='SinglePathOneShot',
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


def get_distribution(net, PICK_ID = 0, FIND_MAX_PARAM = True, MAX_FLOP = 210):
    return


class Maintainer:
    def __init__(self, net, pool, lock, finished_flag, pool_target_size=10, children_size=10, parent_size=5,
                 sample_counts=250, mutate_ratio=0.1, upper_flops=330, flops_interval=20, flops_cuts=8,
                 children_pick_interval=3, nas_root='../', upper_model_size=5.0, bottom_model_size=2.8):
        self.net = net
        self.pool = pool
        self.lock = lock
        self.finished_flag = finished_flag
        self.pool_target_size = pool_target_size
        self.children_size = children_size
        self.parent_size = parent_size
        self.sample_counts = sample_counts
        self.mutate_ratio = mutate_ratio
        self.upper_flops = upper_flops
        self.flops_interval = flops_interval
        self.flops_cuts = flops_cuts
        self.parents = []
        self.children = []
        self.flops_ranges = [max(upper_flops - i * flops_interval, 0) for i in range(flops_cuts)] + \
                            [max(upper_flops - i * flops_interval, 0) for i in range(flops_cuts)][::-1]
        print(self.flops_ranges)
        self.children_pick_ids = list(range(0, children_size, children_pick_interval)) + \
                                 list(reversed(range(0, children_size, children_pick_interval)))
        model_size_interval = (upper_model_size - bottom_model_size) / (len(self.children_pick_ids) - 1)
        self.model_size_range = [upper_model_size - i * model_size_interval
                                 for i in range(len(self.children_pick_ids))]
        self.cur_step = 0
        self.max_step = 2 * flops_cuts * 2 * len(self.children_pick_ids) * sample_counts
        self.chioce_pool = {'channel': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'block': [0, 1, 2, 3]}

        self.lookup_table = load_lookup_table(use_se=net.use_se, last_conv_after_pooling=net.last_conv_after_pooling,
                                              channels_layout='OneShot'
                                              if net.stage_out_channels[0] == 64 else 'ShuffleNetV2+',
                                              nas_root=nas_root)

    def list_to_nd_choices(self, block_choices_list, channel_choices_list):
        # get block choices ndarray
        block_choices = nd.array(block_choices_list)

        # get channel mask
        channel_mask = []
        global_max_length = int(self.net.stage_out_channels[-1] // 2 * self.net.candidate_scales[-1])
        for i in range(len(self.net.stage_repeats)):
            for j in range(self.net.stage_repeats[i]):
                local_mask = [0] * global_max_length
                channel_choice_index = len(
                    channel_mask)  # channel_choice index is equal to current channel_mask length
                channel_num = int(self.net.stage_out_channels[i] // 2 *
                                  self.net.candidate_scales[channel_choices_list[channel_choice_index]])
                local_mask[:channel_num] = [1] * channel_num
                channel_mask.append(local_mask)
        full_channel_mask = nd.array(channel_mask).astype(self.net.dtype, copy=False)
        return block_choices, full_channel_mask

    def evolve(self, pick_id, find_max_param, max_flop, upper_model_size, bottom_model_size):
        # Prepare random parents for the initial evolution
        while len(self.parents) < self.parent_size:
            _, block_choices = self.net.random_block_choices(return_choice_list=True)
            _, channel_choices = self.net.random_channel_mask()
            flops, model_size = get_flop_params(block_choices, channel_choices, self.lookup_table)
            candidate = dict()
            candidate['block'] = block_choices
            candidate['channel'] = channel_choices
            candidate['flops'] = flops
            candidate['model_size'] = model_size
            # TODO: remove comment
            # if flops > max_flop or model_size > self.upper_model_size:
            #     continue
            self.parents.append(candidate)

        # Breed children
        while len(self.children) < self.children_size:
            candidate = dict()

            # randomly select parents from current pool
            mother = random.choice(self.parents)
            father = random.choice(self.parents)

            # make sure mother and father are different
            while father is mother:
                mother = random.choice(self.parents)

            # breed block choice
            block_choices = [0] * len(father['block'])
            for i in range(len(block_choices)):
                block_choices[i] = random.choice([mother['block'][i], father['block'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < self.mutate_ratio:
                    block_choices[i] = random.choice(self.chioce_pool['block'])

            # breed channel choice
            channel_choices = [0] * len(father['channel'])
            for i in range(len(channel_choices)):
                channel_choices[i] = random.choice([mother['channel'][i], father['channel'][i]])
                # Mutation: randomly mutate some of the children.
                if random.random() < self.mutate_ratio:
                    channel_choices[i] = random.choice(self.chioce_pool['channel'])

            flops, model_size = get_flop_params(block_choices, channel_choices, self.lookup_table)

            # if flops > max_flop or model_size > upper_model_size:
            if flops < max_flop - self.flops_interval or flops > max_flop \
                    or model_size < bottom_model_size or model_size > upper_model_size:
                continue

            candidate['block'] = block_choices
            candidate['channel'] = channel_choices
            candidate['flops'] = flops
            candidate['model_size'] = model_size
            self.children.append(candidate)

        # Set target and select
        self.children.sort(key=lambda cand: cand['model_size'], reverse=find_max_param)
        selected_child = self.children[pick_id]

        # Update step for the strolling evolution
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            self.cur_step = 0

        # prepare for next evolve
        self.parents = self.children[:self.parent_size]
        self.children = []
        return selected_child

    def get_single_flops_params(self, pick_id, find_max_param, max_flop, upper_model_size):
        flop_list = []
        model_size_list = []
        for i in range(self.sample_counts):
            if i % 50 == 0:
                print(i)
            candidate = self.evolve(pick_id, find_max_param, max_flop, upper_model_size)
            flop_list.append(candidate['flops'])
            model_size_list.append(candidate['model_size'])
        return flop_list, model_size_list

    def get_all_flops_params(self):
        flop_list = []
        model_size_list = []
        for i, max_flop in enumerate(self.flops_ranges):
            for j, pick_id in enumerate(self.children_pick_ids):
                if (i % 2 == 0 and j < len(self.children_pick_ids) // 2) or \
                        (not i % 2 == 0 and j >= len(self.children_pick_ids) // 2):
                    find_max_param = True
                    print("max_flop {}, pick_id {}, find_max_param, upper model size {}, bottom model size {}"
                          .format(max_flop, pick_id, self.model_size_range[j], self.model_size_range[-1]))
                    for _ in range(self.sample_counts):
                        candidate = self.evolve(pick_id, find_max_param, max_flop,
                                                upper_model_size=self.model_size_range[j],
                                                bottom_model_size=self.model_size_range[-1])
                        flop_list.append(candidate['flops'])
                        model_size_list.append(candidate['model_size'])
                else:
                    find_max_param = False
                    print("max_flop {}, pick_id {}, find_min_param, upper model size {}, bottom model size {}"
                          .format(max_flop, pick_id, self.model_size_range[0], self.model_size_range[j]))
                    for _ in range(self.sample_counts):
                        candidate = self.evolve(pick_id, find_max_param, max_flop,
                                                upper_model_size=self.model_size_range[0],
                                                bottom_model_size=self.model_size_range[j])
                        flop_list.append(candidate['flops'])
                        model_size_list.append(candidate['model_size'])

        return flop_list, model_size_list

    def step(self):
        while not self.finished_flag.value:
            if self.pool.size < self.pool_target_size:
                pick_id, find_max_param, max_flop = self.get_cur_evolve_state()
                candidate = self.evolve(pick_id, find_max_param, max_flop)
                with self.lock:
                    block_choices_nd, full_channel_mask_nd = \
                        self.list_to_nd_choices(candidate['block'], candidate['channel'])
                    candidate['block'] = block_choices_nd
                    candidate['channel'] = full_channel_mask_nd
                    self.pool.append(candidate)


def main():
    args = parse_args()
    net = get_shufflenas_oneshot(use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling,
                                 channels_layout=args.channels_layout)

    print(args)
    m = Maintainer(net, None, None, None, sample_counts=156)
    flop_list, model_size_list = m.get_all_flops_params()
    # flop_list, model_size_list = m.get_single_flops_params(2, find_max_param=True, max_flop=210, upper_model_size=6.0)

    plt.style.use("ggplot")
    plt.figure()
    plt.scatter(flop_list, model_size_list, alpha=0.8, c='mediumaquamarine', s=50, label='subnet')
    plt.title('Flops Param Distribution Strolling Evolution')
    plt.xlabel("Flops")
    plt.ylabel("Params amount")
    axes = plt.gca()
    axes.set_xlim([0, 400])
    axes.set_ylim([0, 5.2])
    plt.legend(loc="lower right")
    # plt.savefig('../images/supernet_flops_params_dist.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
