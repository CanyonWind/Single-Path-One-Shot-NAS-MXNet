from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
from oneshot_nas_blocks import ShuffleNasBlockFixArch, ShuffleNasBlock


class ShuffleNasOneShotFixArch(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None):
        """
        architecture   = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        stage_repeats = [4, 4, 8, 4]
        len(architecture) == len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShotFixArch, self).__init__()
        # Predefined
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [64, 160, 320, 640]
        first_conv_out_channel = 16
        last_conv_out_channel = 100

        assert input_size % 32 == 0
        assert architecture is not None and channel_scales is not None
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
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
                        block_choice = architecture[block_id]
                        mid_channel = int(output_channel // 2 * channel_scales[block_id])
                        block_id += 1
                        if block_choice == 0:
                            print('Shuffle3x3')
                            self.features.add(ShuffleNasBlockFixArch(input_channel, output_channel, mid_channel,
                                                                     block_mode='ShuffleNetV2', ksize=3, stride=stride))
                        elif block_choice == 1:
                            print('Shuffle5x5')
                            self.features.add(ShuffleNasBlockFixArch(input_channel, output_channel, mid_channel,
                                                                     block_mode='ShuffleNetV2', ksize=5, stride=stride))
                        elif block_choice == 2:
                            print('Shuffle7x7')
                            self.features.add(ShuffleNasBlockFixArch(input_channel, output_channel, mid_channel,
                                                                     block_mode='ShuffleNetV2', ksize=7, stride=stride))
                        elif block_choice == 3:
                            print('ShuffleXception3x3')
                            self.features.add(ShuffleNasBlockFixArch(input_channel, output_channel, mid_channel,
                                                                     block_mode='ShuffleXception', ksize=3, stride=stride))
                        else:
                            raise NotImplementedError
                        # update input_channel for next block
                        input_channel = output_channel
                assert block_id == len(architecture)

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

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class ShuffleNasOneShot(HybridBlock):
    def __init__(self, input_size=224, n_class=1000, architecture=None, channel_scales=None):
        """
        architecture   = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        scale_cand_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
        scale_candidate_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
        stage_repeats = [4, 4, 8, 4]
        len(architecture) == len(scale_cand_ids) == sum(stage_repeats) == # feature blocks == 20
        """
        super(ShuffleNasOneShot, self).__init__()
        # Predefined
        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [64, 160, 320, 640]
        first_conv_out_channel = 16
        last_conv_out_channel = 100

        assert input_size % 32 == 0
        assert architecture is not None and channel_scales is not None
        assert len(self.stage_repeats) == len(self.stage_out_channels)

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='features_')
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
                        block_choice = architecture[block_id]
                        mid_channel = int(output_channel // 2 * channel_scales[block_id])
                        block_id += 1
                        if block_choice == 0:
                            print('Shuffle3x3')
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, mid_channel,
                                                              block_mode='ShuffleNetV2', ksize=3, stride=stride))
                        elif block_choice == 1:
                            print('Shuffle5x5')
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, mid_channel,
                                                              block_mode='ShuffleNetV2', ksize=5, stride=stride))
                        elif block_choice == 2:
                            print('Shuffle7x7')
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, mid_channel,
                                                              block_mode='ShuffleNetV2', ksize=7, stride=stride))
                        elif block_choice == 3:
                            print('ShuffleXception3x3')
                            self.features.add(ShuffleNasBlock(input_channel, output_channel, mid_channel,
                                                              block_mode='ShuffleXception', ksize=3, stride=stride))
                        else:
                            raise NotImplementedError
                        # update input_channel for next block
                        input_channel = output_channel
                assert block_id == len(architecture)

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

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shufflenas_oneshot_fixarch(architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2],
                                   scale_ids=[6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]):
    scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    channel_scales = []
    for i in range(len(scale_ids)):
        channel_scales.append(scale_list[scale_ids[i]])
    net = ShuffleNasOneShotFixArch(architecture=architecture, channel_scales=channel_scales)
    return net


def main():
    net = get_shufflenas_oneshot_fixarch()
    net.initialize()
    print(net)

    """ Test ShuffleNasOneShot """
    test_data = nd.ones([5, 3, 224, 224])
    test_outputs = net(test_data)
    print(test_outputs.shape)


if __name__ == '__main__':
    main()