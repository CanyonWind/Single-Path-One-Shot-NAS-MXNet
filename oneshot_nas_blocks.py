from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import random

__all__ = ['ShuffleChannels', 'ShuffleNetBlock', 'ShuffleNasBlock', 'NasHybridSequential',
           'random_block_choices', 'random_channel_mask']


class ShuffleChannels(HybridBlock):
    """
    ShuffleNet channel shuffle Block.
    For reshape 0, -1, -2, -3, -4 meaning:
    https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=reshape#mxnet.ndarray.NDArray.reshape
    """
    def __init__(self, groups=2, **kwargs):
        super(ShuffleChannels, self).__init__()
        # For ShuffleNet v2, groups is always set 2
        assert groups == 2
        self.groups = groups

    def hybrid_forward(self, F, x, *args, **kwargs):
        batch_size, channels, height, width = x.shape
        assert channels % 2 == 0
        mid_channels = channels // 2
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        data_project = F.slice(data, begin=(None, None, None, None), end=(None, mid_channels, None, None))
        data_x = F.slice(data, begin=(None, mid_channels, None, None), end=(None, None, None, None))
        return data_project, data_x


class ChannelSelector(HybridBlock):
    """
    Random channel # selection
    """
    def __init__(self, channel_number, candidate_scales=None):
        super(ChannelSelector, self).__init__()
        if candidate_scales is None:
            self.candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        else:
            self.candidate_scales = candidate_scales
        self.channel_number = channel_number

    def hybrid_forward(self, F, x, block_channel_mask, *args, **kwargs):
        # TODO: fixed arch channel size is not 'max_channel' anymore. Dynamically slice the [all one mask]
        block_channel_mask = F.slice(block_channel_mask, begin=(None, None), end=(None, self.channel_number))
        block_channel_mask = F.reshape(block_channel_mask, shape=(1, self.channel_number, 1, 1))
        x = F.broadcast_mul(x, block_channel_mask)
        return x


class ShuffleNetBlock(HybridBlock):
    def __init__(self, input_channel, output_channel, mid_channel, ksize, stride,
                 block_mode='ShuffleNetV2', **kwargs):
        super(ShuffleNetBlock, self).__init__()
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert block_mode in ['ShuffleNetV2', 'ShuffleXception']

        self.stride = stride
        self.ksize = ksize
        self.padding = self.ksize // 2
        self.block_mode = block_mode

        self.input_channel = input_channel
        self.output_channel = output_channel
        # project_input_C == project_mid_C == project_output_C == main_input_channel
        self.project_channel = input_channel // 2 if stride == 1 else input_channel
        # stride 1, input will be split
        self.main_input_channel = input_channel // 2 if stride == 1 else input_channel
        self.main_mid_channel = mid_channel
        self.main_output_channel = output_channel - self.project_channel

        with self.name_scope():
            """
            Regular block: (We usually have the down-sample block first, then followed by repeated regular blocks)
            Input[64] -> split two halves -> main branch: [32] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                            |                                       |--> main_out_C[32] (final_out_C (64) - input_C[32]
                            |
                            |-----> project branch: [32], do nothing on this half
            Concat two copies: [64 - 32] + [32] --> [64] for final output channel

            =====================================================================

            In "Single path one shot nas" paper, Channel Search is searching for the main branch intermediate #channel.
            And the mid channel is controlled / selected by the channel scales (0.2 ~ 2.0), calculated from:
                mid channel = block final output # channel // 2 * scale

            Since scale ~ (0, 2), this is guaranteed: main mid channel < final output channel
            """
            self.channel_shuffle_and_split = ShuffleChannels(groups=2)
            self.main_branch = NasBaseHybridSequential()
            if block_mode == 'ShuffleNetV2':
                self.main_branch.add(
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    ChannelSelector(channel_number=self.main_mid_channel),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu'),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=self.stride, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_output_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu')
                )
            elif block_mode == 'ShuffleXception':
                self.main_branch.add(
                    # dw with linear output
                    nn.Conv2D(self.main_input_channel, in_channels=self.main_input_channel, kernel_size=self.ksize,
                              strides=self.stride, padding=self.padding, groups=self.main_input_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    ChannelSelector(channel_number=self.main_mid_channel),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu'),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=1, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    ChannelSelector(channel_number=self.main_mid_channel),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu'),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=1, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_output_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu')
                )
            if self.stride == 2:
                """
                Down-sample block:
                Input[16] -> two copies -> main branch: [16] --> mid_channels (final_output_C[64] // 2 * scale[1.4])
                                |                                   |--> main_out_C[48] (final_out_C (64) - input_C[16])
                                |
                                |-----> project branch: [16] --> project_mid_C[16] --> project_out_C[16]
                Concat two copies: [64 - 16] + [16] --> [64] for final output channel
                """
                self.proj_branch = nn.HybridSequential()
                self.proj_branch.add(
                    # dw with linear output
                    nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=self.ksize,
                              strides=stride, padding=self.padding, groups=self.project_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu')
                )

    def hybrid_forward(self, F, old_x, channel_choice, *args, **kwargs):
        if self.stride == 2:
            x_project = old_x
            x = old_x
            # [16 -> 16] + [16 -> 48] -> 64 output channel
            return F.concat(self.proj_branch(x_project), self.main_branch(x, channel_choice), dim=1)
        elif self.stride == 1:
            x_project, x = self.channel_shuffle_and_split(old_x)
            # [64 // 2 -> 32] + [64 // 2 -> 32] -> 64 output channel
            return F.concat(x_project, self.main_branch(x, channel_choice), dim=1)


class ShuffleNasBlock(HybridBlock):
    def __init__(self, input_channel, output_channel, stride, max_channel_scale=2.0, **kwargs):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]

        with self.name_scope():
            """
            Four pre-defined blocks
            """
            max_mid_channel = int(output_channel // 2 * max_channel_scale)
            self.block_sn_3x3 = ShuffleNetBlock(input_channel, output_channel, max_mid_channel,
                                                  3, stride, 'ShuffleNetV2', channel_selection=True)
            self.block_sn_5x5 = ShuffleNetBlock(input_channel, output_channel, max_mid_channel,
                                                  5, stride, 'ShuffleNetV2', channel_selection=True)
            self.block_sn_7x7 = ShuffleNetBlock(input_channel, output_channel, max_mid_channel,
                                                  7, stride, 'ShuffleNetV2', channel_selection=True)
            self.block_sx_3x3 = ShuffleNetBlock(input_channel, output_channel, max_mid_channel,
                                                  3, stride, 'ShuffleXception', channel_selection=True)

    def hybrid_forward(self, F, x, block_choice, block_channel_mask, *args, **kwargs):
        # TODO: ShuffleNasBlock will have three inputs
        #       and pass two inputs to the ShuffleNetBlockII (the one for nas)
        if block_choice == 0:
            x = self.block_sn_3x3(x, block_channel_mask)
        elif block_choice == 1:
            x = self.block_sn_3x3(x, block_channel_mask)
        elif block_choice == 2:
            x = self.block_sn_3x3(x, block_channel_mask)
        elif block_choice == 3:
            x = self.block_sn_3x3(x, block_channel_mask)
        return x


class NasBaseHybridSequential(nn.HybridSequential):
    def __init__(self, prefix=None, params=None):
        super(NasBaseHybridSequential, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, block_channel_mask, *args, **kwargs):
        for block in self._children.values():
            if isinstance(block, ChannelSelector):
                x = block(x, block_channel_mask)
            else:
                x = block(x)
        return x


class NasHybridSequential(nn.HybridSequential):
    def __init__(self, prefix=None, params=None):
        super(NasHybridSequential, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, x, full_arch, full_channel_mask):
        nas_index = 0
        base_index = 0
        for block in self._children.values():
            if isinstance(block, ShuffleNasBlock):
                block_choice = nd.slice(full_arch, begin=nas_index, end=nas_index + 1)
                block_channel_mask = nd.slice(full_channel_mask, begin=(nas_index, None), end=(nas_index + 1, None))
                x = block(x, block_choice, block_channel_mask)
                nas_index += 1
            elif isinstance(block, ShuffleNetBlock):
                block_channel_mask = nd.slice(full_channel_mask, begin=(base_index, None), end=(base_index + 1, None))
                x = block(x, block_channel_mask)
                base_index += 1
            else:
                x = block(x)
        assert (nas_index == full_arch.shape[0] == full_channel_mask.shape[0] or
                base_index == full_arch.shape[0] == full_channel_mask.shape[0])
        return x


def random_block_choices(stage_repeats=None, num_of_block_choices=4):
    if stage_repeats is None:
        stage_repeats = [4, 4, 8, 4]
    block_number = sum(stage_repeats)
    block_choices = []
    for i in range(block_number):
        block_choices.append(random.randint(0, num_of_block_choices - 1))
    return nd.array(block_choices)


def random_channel_mask(stage_repeats=None, stage_out_channels=None, candidate_scales=None,
                        select_all_channels=False):
    """
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    """
    if stage_repeats is None:
        stage_repeats = [4, 4, 8, 4]
    if stage_out_channels is None:
        stage_out_channels = [64, 160, 320, 640]
    if candidate_scales is None:
        candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

    assert len(stage_repeats) == len(stage_out_channels)

    channel_mask = []
    global_max_length = int(stage_out_channels[-1] // 2 * candidate_scales[-1])
    for i in range(len(stage_out_channels)):
        local_max_length = int(stage_out_channels[i] // 2 * candidate_scales[-1])
        local_min_length = int(stage_out_channels[i] // 2 * candidate_scales[0])
        for _ in range(stage_repeats[i]):
            if select_all_channels:
                local_mask = [1] * global_max_length
            else:
                local_mask = [0] * global_max_length
                random_select_channel = random.randint(local_min_length, local_max_length)
                for j in range(random_select_channel):
                    local_mask[j] = 1
            channel_mask.append(local_mask)
    return nd.array(channel_mask)


def main():
    """ Test ShuffleChannels """
    channel_shuffle = ShuffleChannels(groups=2)
    s = nd.ones([1, 8, 3, 3])
    s[:, 4:, :, :] *= 2
    s_project, s_main = channel_shuffle(s)
    print(s)
    print(s_project)
    print(s_main)
    print("Finished testing")

    """ Test ShuffleBlock with "ShuffleNetV2" mode """
    tensor = nd.ones([1, 4, 14, 14])
    tensor[:, 2:, :, :] = 2
    block0 = ShuffleNetBlock(input_channel=4, output_channel=16,
                             mid_channel=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleNetV2')
    block1 = ShuffleNetBlock(input_channel=16, output_channel=16,
                             mid_channel=int(16 // 2 * 1.2), ksize=3, stride=1, block_mode='ShuffleNetV2')
    block0.initialize()
    block1.initialize()
    temp0 = block0(tensor)
    temp1 = block1(temp0)
    print(temp0.shape)
    print(temp1.shape)
    print(block0)
    print(block1)
    print("Finished testing ShuffleNetV2 mode")

    """ Test ShuffleBlock with "ShuffleXception" mode """
    tensor = nd.ones([1, 4, 14, 14])
    tensor[:, 2:, :, :] = 2
    blockx0 = ShuffleNetBlock(input_channel=4, output_channel=16,
                              mid_channel=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleXception')
    blockx1 = ShuffleNetBlock(input_channel=16, output_channel=16,
                              mid_channel=int(16 // 2 * 1.2), ksize=3, stride=1, block_mode='ShuffleXception')
    blockx0.initialize()
    blockx1.initialize()
    tempx0 = blockx0(tensor)
    tempx1 = blockx1(temp0)
    print(tempx0.shape)
    print(tempx1.shape)
    print(blockx0)
    print(blockx1)
    print("Finished testing ShuffleXception mode")

    """ Test ChannelSelection """
    block_final_output_channel = 8
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    max_channel = int(block_final_output_channel // 2 * candidate_scales[-1])
    tensor = nd.ones([1, max_channel, 14, 14])
    for i in range(max_channel):
        tensor[:, i, :, :] = i
    channel_selector = ChannelSelector(block_output_channel=block_final_output_channel)
    print(channel_selector)
    for i in range(4):
        global_channel_mask = random_channel_mask(stage_out_channels=[8, 160, 320, 640])
        local_channel_mask = nd.slice(global_channel_mask, begin=(i, None), end=(i + 1, None))
        selected_tensor = channel_selector(tensor, local_channel_mask)
        print(selected_tensor.shape)


if __name__ == '__main__':
    main()