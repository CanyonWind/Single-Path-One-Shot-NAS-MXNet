import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os
from sys import maxsize

import random
from oneshot_nas_blocks import NasHybridSequential, ShuffleNetBlock, ShuffleNasBlock, NasBatchNorm, Activation, SE


__all__ = ['get_shufflenas_oneshot', 'ShuffleNasOneShot', 'ShuffleNasOneShotFix']


class ShuffleNasOneShot(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None,
                 use_all_blocks=False, bn=nn.BatchNorm, use_se=False, last_conv_after_pooling=False):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShot, self).__init__()
        # Predefined
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [64, 160, 320, 640]
        self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        self.use_all_blocks = use_all_blocks
        self.use_se = use_se

        first_conv_out_channel = 16
        last_conv_out_channel = 1024
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
                        # TODO: update SE and Activation in ShuffleNetBlock and ShuffleNasBlock
                        if fix_arch:
                            block_choice = architecture[block_id]
                            mid_channel = int(output_channel // 2 * channel_scales[block_id])
                            # print("Mid channel: {}".format(mid_channel))
                            block_id += 1
                            if block_choice == 0:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=3, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name))
                            elif block_choice == 1:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=5, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name))
                            elif block_choice == 2:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleNetV2', ksize=7, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name))
                            elif block_choice == 3:
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel, bn=bn,
                                                                  block_mode='ShuffleXception', ksize=3, stride=stride,
                                                                  use_se=block_use_se, act_name=act_name))
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
                                  padding=0, use_bias=False, prefix='fc_'),
                        # No bn for the conv after pooling
                        Activation('hard_swish' if self.use_se else 'relu')
                    )
                else:
                    if self.use_se:
                        # ShuffleNetV2+ approach
                        self.features.add(
                            nn.Conv2D(last_conv_out_channel, in_channels=input_channel, kernel_size=1, strides=1,
                                      padding=0, use_bias=False, prefix='last_conv_'),
                            bn(momentum=0.1),
                            Activation('hard_swish' if self.use_se else 'relu'),
                            nn.GlobalAvgPool2D(),
                            SE(last_conv_out_channel),
                            nn.Conv2D(last_conv_out_channel, in_channels=last_conv_out_channel, kernel_size=1, strides=1,
                                      padding=0, use_bias=False, prefix='fc_'),
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

    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False, dtype='float32'):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            block_choices = []
            for i in range(block_number):
                block_choices.append(random.randint(0, num_of_block_choices - 1))
        return nd.array(block_choices).astype(dtype, copy=False)

    def random_channel_mask(self, select_all_channels=False, dtype='float32', mode='sparse', epoch_after_cs=maxsize):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        mode: str, "dense" or "sparse". Sparse mode select # channel from candidate scales. Dense mode selects
              # channels between randint(min_channel, max_channel).
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        channel_mask = []
        channel_choices = []
        global_max_length = int(self.stage_out_channels[-1] // 2 * self.candidate_scales[-1])
        for i in range(len(self.stage_out_channels)):
            local_max_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[-1])
            local_min_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[0])
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    local_mask = [1] * global_max_length
                else:
                    local_mask = [0] * global_max_length
                    # TODO: shouldn't random between min and max. But select candidate scales and return it.
                    if mode == 'dense':
                        random_select_channel = random.randint(local_min_length, local_max_length)
                        # In dense mode, channel_choices is # channel
                        channel_choices.append(random_select_channel)
                    elif mode == 'sparse':
                        # this is for channel selection warm up: channel choice ~ (8, 9) -> (7, 9) -> ... -> (0, 9)
                        channel_choice = random.randint(max(0, len(self.candidate_scales) - epoch_after_cs - 2),
                                                        len(self.candidate_scales) - 1)
                        random_select_channel = int(self.stage_out_channels[i] // 2 * self.candidate_scales[channel_choice])
                        # In sparse mode, channel_choices is the indices of candidate_scales
                        channel_choices.append(channel_choice)
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return nd.array(channel_mask).astype(dtype, copy=False), channel_choices

    def _initialize(self, force_reinit=True, ctx=mx.cpu()):
        for k, v in self.collect_params().items():
            if 'conv' in k:
                if 'weight' in k:
                    if 'first' in k or 'output' in k or 'fc' in k or 'squeeze' in k or 'excitation' in k:
                        v.initialize(mx.init.Normal(0.01), force_reinit=force_reinit, ctx=ctx)
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
                 use_se=False, last_conv_after_pooling=False):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShotFix, self).__init__(input_size=input_size, n_class=n_class,
                                                   architecture=architecture, channel_scales=channel_scales,
                                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shufflenas_oneshot(architecture=None, scale_ids=None, use_all_blocks=False,
                           use_se=False, last_conv_after_pooling=False):
    if architecture is None and scale_ids is None:
        # Nothing about architecture is specified, do random block selection and channel selection.
        net = ShuffleNasOneShot(use_all_blocks=use_all_blocks, bn=NasBatchNorm,
                                use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)
    elif architecture is not None and scale_ids is not None:
        # Create the specified structure
        if use_all_blocks:
            raise ValueError("For fixed structure, use_all_blocks should not be allowed.")
        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channel_scales = []
        for i in range(len(scale_ids)):
            channel_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNasOneShotFix(architecture=architecture, channel_scales=channel_scales,
                                   use_se=use_se, last_conv_after_pooling=last_conv_after_pooling)
    else:
        raise ValueError("architecture and scale_ids should both be None for supernet "
                         "or both not None for fixed structure model.")
    return net


FIX_ARCH = False
LAST_CONV_AFTER_POOLING = True
USE_SE = True


def main():
    from calculate_flops import get_flops

    if FIX_ARCH:
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                     use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)
    else:
        net = get_shufflenas_oneshot(use_se=USE_SE, last_conv_after_pooling=LAST_CONV_AFTER_POOLING)

    """ Test customized initialization """
    net._initialize(force_reinit=True)
    print(net)

    """ Test ShuffleNasOneShot """
    test_data = nd.ones([5, 3, 224, 224])
    for step in range(1):
        if FIX_ARCH:
            test_outputs = net(test_data)
            net.summary(test_data)
            net.hybridize()
        else:
            block_choices = net.random_block_choices(select_predefined_block=False, dtype='float32')
            full_channel_mask, _ = net.random_channel_mask(select_all_channels=False, dtype='float32')
            test_outputs = net(test_data, block_choices, full_channel_mask)
            net.summary(test_data, block_choices, full_channel_mask)
    if FIX_ARCH:
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        net(test_data)
        net.export("./symbols/ShuffleNas_fixArch", epoch=1)
        flops, model_size = get_flops()
        print("Last conv after pooling: {}, use se: {}".format(LAST_CONV_AFTER_POOLING, USE_SE))
        print("FLOPS: {}M, # parameters: {}M".format(flops, model_size))
    else:
        if not os.path.exists('./params'):
            os.makedirs('./params')
        net.save_parameters('./params/ShuffleNasOneshot-imagenet-supernet.params')
    print(test_outputs.shape)


if __name__ == '__main__':
    main()
