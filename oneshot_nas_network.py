from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import os

import random
from oneshot_nas_blocks import NasHybridSequential, ShuffleNetBlock, ShuffleNasBlock


__all__ = ['get_shufflenas_oneshot', 'ShuffleNasOneShot', 'ShuffleNasOneShotFix']


class ShuffleNasOneShot(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None):
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
        first_conv_out_channel = 16
        last_conv_out_channel = 1024

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
                              padding=1, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu')
                )

                # features
                input_channel = 16
                block_id = 0
                for stage_id in range(len(self.stage_repeats)):
                    numrepeat = self.stage_repeats[stage_id]
                    output_channel = self.stage_out_channels[stage_id]
                    # create repeated blocks for current stage
                    for i in range(numrepeat):
                        stride = 2 if i == 0 else 1
                        if fix_arch:
                            block_choice = architecture[block_id]
                            mid_channel = int(output_channel // 2 * channel_scales[block_id])
                            block_id += 1
                            if block_choice == 0:
                                print('Shuffle3x3')
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                                                  block_mode='ShuffleNetV2', ksize=3, stride=stride))
                            elif block_choice == 1:
                                print('Shuffle5x5')
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                                                  block_mode='ShuffleNetV2', ksize=5, stride=stride))
                            elif block_choice == 2:
                                print('Shuffle7x7')
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                                                  block_mode='ShuffleNetV2', ksize=7, stride=stride))
                            elif block_choice == 3:
                                print('ShuffleXception3x3')
                                self.features.add(ShuffleNetBlock(input_channel, output_channel, mid_channel,
                                                                  block_mode='ShuffleXception', ksize=3, stride=stride))
                            else:
                                raise NotImplementedError
                        else:
                            block_id += 1
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, stride=stride,
                                                              max_channel_scale=self.candidate_scales[-1]))
                        # update input_channel for next block
                        input_channel = output_channel
                assert block_id == sum(self.stage_repeats)

                # last conv
                self.features.add(
                    nn.Conv2D(last_conv_out_channel, in_channels=input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu')
                )
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Dropout(0.1))
            self.output = nn.HybridSequential(prefix='output_')
            with self.output.name_scope():
                self.output.add(
                    nn.Conv2D(n_class, in_channels=last_conv_out_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
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

    def random_channel_mask(self, select_all_channels=False, dtype='float32'):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
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
                    random_select_channel = random.randint(local_min_length, local_max_length)
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return nd.array(channel_mask).astype(dtype, copy=False)

    def hybrid_forward(self, F, x, full_arch, full_scale_mask, *args, **kwargs):
        x = self.features(x, full_arch, full_scale_mask)
        x = self.output(x)
        return x


class ShuffleNasOneShotFix(ShuffleNasOneShot):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None):
        """
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        stage_repeats = [4, 4, 8, 4]
        len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShotFix, self).__init__(input_size=input_size, n_class=n_class,
                                                   architecture=architecture, channel_scales=channel_scales)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shufflenas_oneshot(architecture=None, scale_ids=None):
    if architecture is None and scale_ids is None:
        # Nothing is specified, do random block selecting and channel selecting.
        net = ShuffleNasOneShot()
    elif architecture is not None and scale_ids is not None:
        # Create the specified structure
        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channel_scales = []
        for i in range(len(scale_ids)):
            channel_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNasOneShotFix(architecture=architecture, channel_scales=channel_scales)
    else:
        raise ValueError("architecture and scale_ids should be both None or not None.")
    return net


FIX_ARCH = False


def main():
    if FIX_ARCH:
        architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids)
    else:
        net = get_shufflenas_oneshot()
    net.initialize()
    print(net)

    """ Test ShuffleNasOneShot """
    test_data = nd.ones([5, 3, 224, 224])
    for step in range(10):
        if FIX_ARCH:
            test_outputs = net(test_data)
            if step == 0:
                net.summary(test_data)
            net.hybridize()
        else:
            block_choices = net.random_block_choices(select_predefined_block=False, dtype='float32')
            full_channel_mask = net.random_channel_mask(select_all_channels=False, dtype='float32')
            test_outputs = net(test_data, block_choices, full_channel_mask)
            net.summary(test_data, block_choices, full_channel_mask)
    if FIX_ARCH:
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        net.export("./symbols/ShuffleNas_fixArch", epoch=1)
    else:
        if not os.path.exists('./params'):
            os.makedirs('./params')
        net.save_parameters('./params/ShuffleNasOneshot-imagenet-supernet.params')
    print(test_outputs.shape)


if __name__ == '__main__':
    main()
