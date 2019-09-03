from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd

import random
from oneshot_nas_blocks import *


__all__ = ['get_shufflenas_oneshot', 'ShuffleNasOneShot']


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
        last_conv_out_channel = 100

        if architecture is None and channel_scales is None:
            fix_arch = False
        else:
            fix_arch = True
            assert len(architecture) == len(channel_scales)

        assert input_size % 32 == 0
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        with self.name_scope():
            self.features = NasHybridSequential(prefix='features_')
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
                                                                    block_mode='ShuffleXception', ksize=3,
                                                                    stride=stride))
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

    def random_block_choices(self, num_of_block_choices=4, select_predefined_block=False):
        if select_predefined_block:
            block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        else:
            block_number = sum(self.stage_repeats)
            block_choices = []
            for i in range(block_number):
                block_choices.append(random.randint(0, num_of_block_choices - 1))
        return nd.array(block_choices)

    def random_channel_mask(self, select_all_channels=False):
        """
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        """
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        channel_mask = []
        global_max_length = int(self.stage_out_channels[-1] // 2 * self.candidate_scales[-1])
        for i in range(len(self.stage_out_channels)):
            local_max_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[-1])
            local_min_length = int(self.stage_out_channels[i] // 2 * self.candidate_scales[0])
            for _ in range(self.stage_repeats[i]):
                if select_all_channels:
                    local_mask = [1] * global_max_length
                else:
                    local_mask = [0] * global_max_length
                    random_select_channel = random.randint(local_min_length, local_max_length)
                    for j in range(random_select_channel):
                        local_mask[j] = 1
                channel_mask.append(local_mask)
        return nd.array(channel_mask)

    def hybrid_forward(self, F, x, full_arch, full_scale_mask, *args, **kwargs):
        x = self.features(x, full_arch, full_scale_mask)
        x = self.output(x)
        return x


def get_shufflenas_oneshot(architecture=None, scale_ids=None):
    if architecture is None and scale_ids is None:
        net = ShuffleNasOneShot(architecture=None, channel_scales=None)
    else:
        scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        channel_scales = []
        for i in range(len(scale_ids)):
            channel_scales.append(scale_list[scale_ids[i]])
        net = ShuffleNasOneShot(architecture=architecture, channel_scales=channel_scales)
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
    # for fixed arch, block_choices is (required for forward but) actually ignored in using
    block_choices = net.random_block_choices(select_predefined_block=False)
    full_channel_mask = net.random_channel_mask(select_all_channels=False)
    test_outputs = net(test_data, block_choices, full_channel_mask)
    print(test_outputs.shape)


if __name__ == '__main__':
    main()