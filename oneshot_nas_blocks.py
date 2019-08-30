from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd


__all__ = ['ShuffleChannels', 'ShuffleNasBlock']


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
        batch_size, channels, height, width, = x.shape
        assert channels % 2 == 0
        mid_channels = channels // 2
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        data_project = F.slice(data, begin=(None, None, None, None), end=(None, mid_channels, None, None))
        data_x = F.slice(data, begin=(None, mid_channels, None, None), end=(None, None, None, None))
        return data_project, data_x


class ShuffleNasBlockFixArch(HybridBlock):
    def __init__(self, input_channel, output_channel, mid_channel, ksize, stride,
                 block_mode='ShuffleNetV2', **kwargs):
        super(ShuffleNasBlockFixArch, self).__init__()
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
            And the mid channel is controlled / selected by the channel scales (0.2 ~ 1.6), calculated from:
                mid channel = block final output # channel // 2 * scale

            Since scale ~ (0, 2), this is guaranteed: mid channel < output channel
            """
            self.channel_shuffle_and_split = ShuffleChannels(groups=2)
            self.main_branch = nn.HybridSequential()
            if block_mode == 'ShuffleNetV2':
                self.main_branch.add(
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
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
                    nn.BatchNorm(momentum=0.1),
                    nn.Activation('relu'),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=1, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    nn.BatchNorm(momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
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

    def hybrid_forward(self, F, old_x, *args, **kwargs):

        if self.stride == 2:
            x_project = old_x
            x = old_x
            # [16 -> 16] + [16 -> 48] -> 64 output channel
            return F.concat(self.proj_branch(x_project), self.main_branch(x), dim=1)
        elif self.stride == 1:
            x_project, x = self.channel_shuffle_and_split(old_x)
            # [64 // 2 -> 32] + [64 // 2 -> 32] -> 64 output channel
            return F.concat(x_project, self.main_branch(x), dim=1)


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
    block0 = ShuffleNasBlockFixArch(input_channel=4, output_channel=16,
                                    mid_channel=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleNetV2')
    block1 = ShuffleNasBlockFixArch(input_channel=16, output_channel=16,
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
    blockx0 = ShuffleNasBlockFixArch(input_channel=4, output_channel=16,
                                     mid_channel=int(16 // 2 * 1.4), ksize=3, stride=2, block_mode='ShuffleXception')
    blockx1 = ShuffleNasBlockFixArch(input_channel=16, output_channel=16,
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


if __name__ == '__main__':
    main()