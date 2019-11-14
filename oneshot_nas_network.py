import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os
from sys import maxsize
import random
import sys
import numpy as np

from oneshot_nas_blocks import NasHybridSequential, ShuffleNetBlock, ShuffleNasBlock, NasBatchNorm, Activation, SE, \
                               ShuffleChannelsConv, ShuffleChannels

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/utils')
from calculate_flops import get_flops


__all__ = ['get_shufflenas_oneshot', 'ShuffleNasOneShot', 'ShuffleNasOneShotFix']


class ShuffleNasOneShot(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None,
                 use_all_blocks=False, bn=nn.BatchNorm, use_se=False, last_conv_after_pooling=False,
                 shuffle_method=ShuffleChannels, stage_out_channels=None, candidate_scales=None,
                 last_conv_out_channel=1024):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShot, self).__init__()
        # Predefined
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [64, 160, 320, 640] if stage_out_channels is None else stage_out_channels
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] \
            if candidate_scales is None else candidate_scales
        self.use_all_blocks = use_all_blocks
        self.use_se = use_se

        first_conv_out_channel = 16
        last_conv_out_channel = last_conv_out_channel
        self.last_conv_after_pooling = last_conv_after_pooling

        if architecture is None and channel_scales is None:
            fix_arch = False
        elif architecture is not None and channel_scales is not None:
            fix_arch = True
            assert len(architecture) == len(channel_scales)
        else:
            raise ValueError("architecture and scale_ids should be both None or not None.")
        self.fix_arch = fix_arch

        assert input_size % 32 == 0
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        with self.name_scope():
            self.features = nn.HybridSequential() if fix_arch else NasHybridSequential(prefix='features_')
            with self.features.name_scope():
                # first conv
                self.features.add(
                    nn.Conv2D(first_conv_out_channel, in_channels=3, kernel_size=3, strides=2,
                              padding=1, use_bias=False, prefix='first_conv_'),
                    bn(momentum=0.1),
                    Activation('hard_swish' if self.use_se else 'relu')
                )

                # features
                input_channel = 16
                block_id = 0
                for stage_id in range(len(self.stage_repeats)):
                    numrepeat = self.stage_repeats[stage_id]
                    output_channel = self.stage_out_channels[stage_id]

                    if self.use_se:
                        act_name = 'hard_swish' if stage_id >= 1 else 'relu'
                        block_use_se = True if stage_id >= 2 else False
                    else:
                        act_name = 'relu'
                        block_use_se = False
                    # create repeated blocks for current stage
                    for i in range(numrepeat):
                        stride = 2 if i == 0 else 1
                        if fix_arch:
                            block_choice = architecture[block_id]
                            # TODO: change back to make_divisible
                            # mid_channel = make_divisible(int(output_channel // 2 * channel_scales[block_id]))
                            mid_channel = int(output_channel // 2 * channel_scales[block_id])
                            # print("Block {} mid channel: {}".format(block_id, mid_channel))
                            # print("Mid channel: {}".format(mid_channel))
                            block_id += 1
                            if block_choice == 0:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=3, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name,
                                                                  shuffle_method=shuffle_method))
                            elif block_choice == 1:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=5, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name,
                                                                  shuffle_method=shuffle_method))
                            elif block_choice == 2:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=7, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name,
                                                                  shuffle_method=shuffle_method))
                            elif block_choice == 3:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleXception', ksize=3, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name,
                                                                  shuffle_method=shuffle_method))
                            else:
                                raise NotImplementedError
                        else:
                            block_id += 1
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, stride=stride, bn=bn,
                                                              max_channel_scale=self.candidate_scales[-1],
                                                              use_all_blocks=self.use_all_blocks,
                                                              use_se=block_use_se, act_name=act_name))
                        # update input_channel for next block
                        input_channel = output_channel
                assert block_id == sum(self.stage_repeats)

                # last conv
                if self.last_conv_after_pooling:
                    # MobileNet V3 approach
                    self.features.add(
                        nn.GlobalAvgPool2D(),
                        # no last SE for MobileNet V3 style
                        nn.Conv2D(last_conv_out_channel, in_channels=input_channel, kernel_size=1, strides=1,
                                  padding=0, use_bias=True, prefix='conv_fc_'),
                        # No bn for the conv after pooling
                        Activation('hard_swish' if self.use_se else 'relu')
                    )
                else:
                    if self.use_se:
                        # ShuffleNetV2+ approach
                        self.features.add(
                            nn.Conv2D(make_divisible(last_conv_out_channel * 0.75), in_channels=input_channel,
                                      kernel_size=1, strides=1,
                                      padding=0, use_bias=False, prefix='last_conv_'),
                            bn(momentum=0.1),
                            Activation('hard_swish' if self.use_se else 'relu'),
                            nn.GlobalAvgPool2D(),
                            SE(make_divisible(last_conv_out_channel * 0.75)),
                            nn.Conv2D(last_conv_out_channel, in_channels=make_divisible(last_conv_out_channel * 0.75),
                                      kernel_size=1, strides=1,
                                      padding=0, use_bias=True, prefix='conv_fc_'),
                            # No bn for the conv after pooling
                            Activation('hard_swish' if self.use_se else 'relu')
                        )
                    else:
                        # original Oneshot Nas approach
                        self.features.add(
                            nn.Conv2D(last_conv_out_channel, in_channels=input_channel, kernel_size=1, strides=1,
                                      padding=0, use_bias=False, prefix='last_conv_'),
                            bn(momentum=0.1),
                            Activation('hard_swish' if self.use_se else 'relu'),
                            nn.GlobalAvgPool2D()
                        )

                # Dropout ratio follows ShuffleNetV2+ for se
                self.features.add(nn.Dropout(0.2 if self.use_se else 0.1))
            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(n_class, in_channels=last_conv_out_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=True),
                    nn.Flatten()
                )

    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False, dtype='float32',
                             return_choice_list=False):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            block_choices = []
            for i in range(block_number):
                block_choices.append(random.randint(0, num_of_block_choices - 1))
        if return_choice_list:
            return nd.array(block_choices).astype(dtype, copy=False), block_choices
        else:
            return nd.array(block_choices).astype(dtype, copy=False)

    def random_channel_mask(self, select_all_channels=False, dtype='float32', mode='sparse', epoch_after_cs=maxsize,
                            ignore_first_two_cs=False):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)
        # From [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] to [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0], warm-up stages are
        # not just 1 epoch, but 2, 3, 4, 5 accordingly.

        epoch_delay_early = {0: 0,  # 8
                             1: 1, 2: 1,  # 7
                             3: 2, 4: 2, 5: 2,  # 6
                             6: 3, 7: 3, 8: 3, 9: 3,  # 5
                             10: 4, 11: 4, 12: 4, 13: 4, 14: 4,
                             15: 5, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
                             21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 27: 6, 28: 6,
                             29: 6, 30: 6, 31: 6, 32: 6, 33: 6, 34: 6, 35: 6, 36: 7,
                           }
        epoch_delay_late = {0: 0,
                            1: 1,
                            2: 2,
                            3: 3,
                            4: 4, 5: 4,  # warm up epoch: 2 [1.0, 1.2, ... 1.8, 2.0]
                            6: 5, 7: 5, 8: 5,  # warm up epoch: 3 ...
                            9: 6, 10: 6, 11: 6, 12: 6,  # warm up epoch: 4 ...
                            13: 7, 14: 7, 15: 7, 16: 7, 17: 7,  # warm up epoch: 5 [0.4, 0.6, ... 1.8, 2.0]
                            18: 8, 19: 8, 20: 8, 21: 8, 22: 8, 23: 8}  # warm up epoch: 6, after 17, use all scales

        if 0 <= epoch_after_cs <= 23 and self.stage_out_channels[0] >= 64:
            delayed_epoch_after_cs = epoch_delay_late[epoch_after_cs]
        elif 0 <= epoch_after_cs <= 36 and self.stage_out_channels[0] < 64:
            delayed_epoch_after_cs = epoch_delay_early[epoch_after_cs]
        else:
            delayed_epoch_after_cs = epoch_after_cs

        if ignore_first_two_cs:
            min_scale_id = 2
        else:
            min_scale_id = 0

        channel_mask = []
        channel_choices = []
        global_max_length = make_divisible(int(self.stage_out_channels[-1] // 2 * self.candidate_scales[-1]))
        for i in range(len(self.stage_out_channels)):
            local_max_length = make_divisible(int(self.stage_out_channels[i] // 2 * self.candidate_scales[-1]))
            local_min_length = make_divisible(int(self.stage_out_channels[i] // 2 * self.candidate_scales[0]))
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    local_mask = [1] * global_max_length
                    channel_choices = [len(self.candidate_scales) - 1] * sum(self.stage_repeats)
                else:
                    local_mask = [0] * global_max_length
                    if mode == 'dense':
                        random_select_channel = random.randint(local_min_length, local_max_length)
                        # In dense mode, channel_choices is # channel
                        channel_choices.append(random_select_channel)
                    elif mode == 'sparse':
                        # this is for channel selection warm up: channel choice ~ (8, 9) -> (7, 9) -> ... -> (0, 9)
                        channel_scale_start = max(min_scale_id, len(self.candidate_scales) - delayed_epoch_after_cs - 2)
                        channel_choice = random.randint(channel_scale_start, len(self.candidate_scales) - 1)
                        random_select_channel = int(self.stage_out_channels[i] // 2 *
                                                    self.candidate_scales[channel_choice])
                        # In sparse mode, channel_choices is the indices of candidate_scales
                        channel_choices.append(channel_choice)
                        # To take full advantages of acceleration, # of channels should be divisible to 8.
                        random_select_channel = make_divisible(random_select_channel)
                    else:
                        raise ValueError("Unrecognized mode: {}".format(mode))
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return nd.array(channel_mask).astype(dtype, copy=False), channel_choices

    def _initialize(self, force_reinit=True, ctx=mx.cpu(), dtype='float32'):
        for k, v in self.collect_params().items():
            if 'conv' in k:
                if 'weight' in k:
                    if 'first' in k or 'output' in k or 'fc' in k or 'squeeze' in k or 'excitation' in k:
                        v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                    elif 'transpose' in k:
                        v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
                        v.set_data(nd.cast(generate_transpose_conv_kernel(v.shape[0]), dtype=dtype))
                        v.grad_req = 'null'
                    else:
                        v.initialize(mx.init.Normal(1.0 / v.shape[1]), force_reinit=force_reinit, ctx=ctx)
                if 'bias' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)
            elif 'batchnorm' in k:
                if 'gamma' in k:
                    v.initialize(mx.init.Constant(1), force_reinit=force_reinit, ctx=ctx)
                if 'beta' in k:
                    v.initialize(mx.init.Constant(0.0001), force_reinit=force_reinit, ctx=ctx)
                if 'running' in k:
                    v.initialize(mx.init.Constant(0), force_reinit=force_reinit, ctx=ctx)

    def hybrid_forward(self, F, x, full_arch, full_scale_mask, *args, **kwargs):
        x = self.features(x, full_arch, full_scale_mask)
        x = self.output(x)
        return x


class ShuffleNasOneShotFix(ShuffleNasOneShot):
    # Unlike its parent class, fix-arch model does not have the control of "use_all_blocks" and "bn"(for NasBN).
    # It should use the default False and nn.BatchNorm correspondingly.
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None,
                 use_se=False, last_conv_after_pooling=False, shuffle_method=ShuffleChannels,
                 stage_out_channels=None, candidate_scales=None, last_conv_out_channel=1024):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShotFix, self).__init__(input_size=input_size, n_class=n_class,
                                                   architecture=architecture, channel_scales=channel_scales,
                                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                                   shuffle_method=shuffle_method, stage_out_channels=stage_out_channels,
                                                   candidate_scales=candidate_scales, last_conv_out_channel=last_conv_out_channel)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


def generate_transpose_conv_kernel(channels):
    c = channels
    if c % 2 != 0:
        raise ValueError('Channel number should be even.')
    idx = np.zeros(c)
    idx[np.arange(0, c, 2)] = np.arange(c / 2)
    idx[np.arange(1, c, 2)] = np.arange(c / 2, c, 1)
    weights = np.zeros((c, c))
    weights[np.arange(c), idx.astype(int)] = 1.0
    return nd.expand_dims(nd.expand_dims(nd.array(weights), axis=2), axis=3)


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


def get_shufflenas_oneshot(architecture=None, scale_ids=None, use_all_blocks=False, n_class=1000,
                           use_se=False, last_conv_after_pooling=False, shuffle_by_conv=False,
                           channels_layout='OneShot'):
    if channels_layout == 'OneShot':
        stage_out_channels = [64, 160, 320, 640]
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        last_conv_out_channel = 1024
    elif channels_layout == 'ShuffleNetV2+':
        stage_out_channels = [48, 128, 256, 512]
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        last_conv_out_channel = 1024
    else:
        raise ValueError("Unrecognized channels_layout: {}. "
                         "Please choose from ['ShuffleNetV2', 'OneShot']".format(channels_layout))

    if architecture is None and scale_ids is None:
        # Nothing about architecture is specified, do random block selection and channel selection.
        net = ShuffleNasOneShot(n_class=n_class, use_all_blocks=use_all_blocks, bn=NasBatchNorm,
                                use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                stage_out_channels=stage_out_channels, candidate_scales=candidate_scales,
                                last_conv_out_channel=last_conv_out_channel)
    elif architecture is not None and scale_ids is not None:
        # Create the specified structure
        if use_all_blocks:
            raise ValueError("For fixed structure, use_all_blocks should not be allowed.")
        shuffle_method = ShuffleChannelsConv if shuffle_by_conv else ShuffleChannels

        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channel_scales = []
        for i in range(len(scale_ids)):
            # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
            channel_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNasOneShotFix(architecture=architecture, n_class=n_class, channel_scales=channel_scales,
                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling,
                                   shuffle_method=shuffle_method, stage_out_channels=stage_out_channels,
                                   candidate_scales=candidate_scales, last_conv_out_channel=last_conv_out_channel)
    else:
        raise ValueError("architecture and scale_ids should both be None for supernet "
                         "or both not None for fixed structure model.")
    return net


def main():
    # Save a toy SE SuperNet for playing with the search codes
    supernet = get_shufflenas_oneshot(use_se=True, last_conv_after_pooling=True, channels_layout='OneShot')
    supernet._initialize(force_reinit=True)
    if not os.path.exists('./params'):
        os.makedirs('./params')
    test_data = nd.ones([5, 3, 224, 224])
    for step in range(1):
        block_choices = supernet.random_block_choices(select_predefined_block=False, dtype='float32')
        full_channel_mask, _ = supernet.random_channel_mask(select_all_channels=False, dtype='float32')
        _ = supernet(test_data, block_choices, full_channel_mask)

    if not os.path.exists('./params'):
        os.makedirs('./params')
    supernet.save_parameters('./params/ShuffleNasOneshot-imagenet-supernet.params')

    # Profiling the OneShot-s+
    architecture = [0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0]
    scale_ids = [8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6]
    net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids, channels_layout='OneShot',
                                 use_se=True, last_conv_after_pooling=True)
    net._initialize(force_reinit=True)
    print(net)
    test_data = nd.ones([5, 3, 224, 224])
    for step in range(1):
        _ = net(test_data)
    net.summary(test_data)

    net.hybridize()
    if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
    net(test_data)
    net.export("./symbols/ShuffleNas_fixArch", epoch=0)
    flops, model_size = get_flops()
    print("Last conv after pooling: {}, use se: {}".format(True, True))
    print("FLOPS: {}M, # parameters: {}M".format(flops, model_size))


if __name__ == '__main__':
    main()
