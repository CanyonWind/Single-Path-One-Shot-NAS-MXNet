from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from mxnet import nd
import random
import numpy as np


__all__ = ['ShuffleNetBlock', 'ShuffleNasBlock', 'Activation', 'SE', 'NasBatchNorm', 'NasHybridSequential']


class Activation(HybridBlock):
    """Activation function used in MobileNetV3"""
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)


class ReLU6(HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")


class HardSigmoid(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return F.clip(x + 3, 0, 6, name="hard_sigmoid") / 6.


class HardSwish(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * (F.clip(x + 3, 0, 6, name="hard_swish") / 6.)


class SE(HybridBlock):
    def __init__(self, num_in, ratio=4,
                 act_func=("relu", "hard_sigmoid"), use_bn=False, **kwargs):
        super(SE, self).__init__(**kwargs)

        def make_divisible(x, divisible_by=8):
            # make the mid channel to be divisible to 8 can increase the cache hitting ratio
            return int(np.ceil(x * 1. / divisible_by) * divisible_by)

        self.use_bn = use_bn
        num_out = num_in
        num_mid = make_divisible(num_out // ratio)

        with self.name_scope():
            self.channel_attention = nn.HybridSequential()
            self.channel_attention.add(nn.GlobalAvgPool2D(),
                                       nn.Conv2D(channels=num_mid, in_channels=num_in, kernel_size=1, use_bias=True,
                                                 prefix='conv_squeeze_'),
                                       Activation(act_func[0]),
                                       nn.Conv2D(channels=num_out, in_channels=num_mid, kernel_size=1, use_bias=True,
                                                 prefix='conv_excitation_'),
                                       Activation(act_func[1]))

    def hybrid_forward(self, F, x):
        out = self.channel_attention(x)
        return F.broadcast_mul(x, out)


class ShuffleChannels(HybridBlock):
    """
    ShuffleNet channel shuffle Block.
    For reshape 0, -1, -2, -3, -4 meaning:
    https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=reshape#mxnet.ndarray.NDArray.reshape
    """
    def __init__(self, mid_channel, groups=2, **kwargs):
        super(ShuffleChannels, self).__init__()
        # For ShuffleNet v2, groups is always set 2
        assert groups == 2
        self.groups = groups
        self.mid_channel = mid_channel

    def hybrid_forward(self, F, x, *args, **kwargs):
        # batch_size, channels, height, width = x.shape
        # assert channels % 2 == 0
        # mid_channels = channels // 2
        data = F.reshape(x, shape=(0, -4, self.groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        data_project = F.slice(data, begin=(None, None, None, None), end=(None, self.mid_channel, None, None))
        data_x = F.slice(data, begin=(None, self.mid_channel, None, None), end=(None, None, None, None))
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
        block_channel_mask = F.slice(block_channel_mask, begin=(None, None), end=(None, self.channel_number))
        block_channel_mask = F.reshape(block_channel_mask, shape=(1, self.channel_number, 1, 1))
        x = F.broadcast_mul(x, block_channel_mask.as_in_context(x.context))
        return x


class ShuffleNetBlock(HybridBlock):
    def __init__(self, input_channel, output_channel, mid_channel, ksize, stride,
                 block_mode='ShuffleNetV2', fix_arch=True, bn=nn.BatchNorm, act_name='relu', use_se=False, **kwargs):
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
        self.fix_arch = fix_arch

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
            self.channel_shuffle_and_split = ShuffleChannels(mid_channel=input_channel // 2, groups=2)
            self.main_branch = nn.HybridSequential() if fix_arch else NasBaseHybridSequential()

            if block_mode == 'ShuffleNetV2':
                self.main_branch.add(
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False))
                if not fix_arch:
                    self.main_branch.add(ChannelSelector(channel_number=self.main_mid_channel))
                
                self.main_branch.add(
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    Activation(act_name),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=self.stride, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_output_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    bn(in_channels=self.main_output_channel, momentum=0.1),
                    Activation(act_name)
                )
            elif block_mode == 'ShuffleXception':
                self.main_branch.add(
                    # dw with linear output
                    nn.Conv2D(self.main_input_channel, in_channels=self.main_input_channel, kernel_size=self.ksize,
                              strides=self.stride, padding=self.padding, groups=self.main_input_channel, use_bias=False),
                    bn(in_channels=self.main_input_channel, momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_input_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False))
                if not fix_arch:
                    self.main_branch.add(ChannelSelector(channel_number=self.main_mid_channel))
                    
                self.main_branch.add(
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    Activation(act_name),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=1, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False))
                if not fix_arch:
                    self.main_branch.add(ChannelSelector(channel_number=self.main_mid_channel))
                    
                self.main_branch.add(
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    Activation(act_name),
                    # dw with linear output
                    nn.Conv2D(self.main_mid_channel, in_channels=self.main_mid_channel, kernel_size=self.ksize,
                              strides=1, padding=self.padding, groups=self.main_mid_channel, use_bias=False),
                    bn(in_channels=self.main_mid_channel, momentum=0.1),
                    # pw
                    nn.Conv2D(self.main_output_channel, in_channels=self.main_mid_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    bn(in_channels=self.main_output_channel, momentum=0.1),
                    Activation(act_name)
                )
            if use_se:
                self.main_branch.add(SE(self.main_output_channel))
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
                    bn(in_channels=self.project_channel, momentum=0.1),
                    # pw
                    nn.Conv2D(self.project_channel, in_channels=self.project_channel, kernel_size=1, strides=1,
                              padding=0, use_bias=False),
                    bn(in_channels=self.project_channel, momentum=0.1),
                    Activation(act_name)
                )

    def hybrid_forward(self, F, old_x, *args, **kwargs):
        if self.stride == 2:
            x_project = old_x
            x = old_x
            return F.concat(self.proj_branch(x_project), self.main_branch(x), dim=1)

        elif self.stride == 1:
            x_project, x = self.channel_shuffle_and_split(old_x)
            return F.concat(x_project, self.main_branch(x), dim=1)


class ShuffleNetCSBlock(ShuffleNetBlock):
    """
    ShuffleNetBlock with Channel Selecting
    """
    def __init__(self, input_channel, output_channel, mid_channel, ksize, stride,
                 block_mode='ShuffleNetV2', fix_arch=False,  bn=nn.BatchNorm, act_name='relu', use_se=False, **kwargs):
        super(ShuffleNetCSBlock, self).__init__(input_channel, output_channel, mid_channel, ksize, stride,
                                                block_mode=block_mode, fix_arch=fix_arch, bn=bn,
                                                act_name=act_name, use_se=use_se, **kwargs)

    def hybrid_forward(self, F, old_x, channel_choice, *args, **kwargs):
        if self.stride == 2:
            x_project = old_x
            x = old_x
            return F.concat(self.proj_branch(x_project), self.main_branch(x, channel_choice), dim=1)
        elif self.stride == 1:
            x_project, x = self.channel_shuffle_and_split(old_x)
            return F.concat(x_project, self.main_branch(x, channel_choice), dim=1)


class ShuffleNasBlock(HybridBlock):
    def __init__(self, input_channel, output_channel, stride, max_channel_scale=2.0, 
                 use_all_blocks=False, bn=nn.BatchNorm, act_name='relu', use_se=False, **kwargs):
        super(ShuffleNasBlock, self).__init__()
        assert stride in [1, 2]
        self.use_all_blocks = use_all_blocks
        with self.name_scope():
            """
            Four pre-defined blocks
            """
            max_mid_channel = int(output_channel // 2 * max_channel_scale)
            self.block_sn_3x3 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                  3, stride, 'ShuffleNetV2', bn=bn, act_name=act_name, use_se=use_se)
            self.block_sn_5x5 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                  5, stride, 'ShuffleNetV2', bn=bn, act_name=act_name, use_se=use_se)
            self.block_sn_7x7 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                  7, stride, 'ShuffleNetV2', bn=bn, act_name=act_name, use_se=use_se)
            self.block_sx_3x3 = ShuffleNetCSBlock(input_channel, output_channel, max_mid_channel,
                                                  3, stride, 'ShuffleXception', bn=bn, act_name=act_name, use_se=use_se)

    def hybrid_forward(self, F, x, block_choice, block_channel_mask, *args, **kwargs):
        # ShuffleNasBlock has three inputs and passes two inputs to the ShuffleNetCSBlock
        if self.use_all_blocks:
            temp1 = self.block_sn_3x3(x, block_channel_mask)
            temp2 = self.block_sn_5x5(x, block_channel_mask)
            temp3 = self.block_sn_7x7(x, block_channel_mask)
            temp4 = self.block_sx_3x3(x, block_channel_mask)
            x = temp1 + temp2 + temp3 + temp4
        else:
            if block_choice == 0:
                x = self.block_sn_3x3(x, block_channel_mask)
            elif block_choice == 1:
                x = self.block_sn_5x5(x, block_channel_mask)
            elif block_choice == 2:
                x = self.block_sn_7x7(x, block_channel_mask)
            elif block_choice == 3:
                x = self.block_sx_3x3(x, block_channel_mask)
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
                block_choice = F.slice(full_arch, begin=nas_index, end=nas_index + 1)
                block_channel_mask = F.slice(full_channel_mask, begin=(nas_index, None), end=(nas_index + 1, None))
                x = block(x, block_choice, block_channel_mask)
                nas_index += 1
            elif isinstance(block, ShuffleNetBlock):
                block_channel_mask = F.slice(full_channel_mask, begin=(base_index, None), end=(base_index + 1, None))
                x = block(x, block_channel_mask)
                base_index += 1
            else:
                x = block(x)
        # assert (nas_index == full_arch.shape[0] == full_channel_mask.shape[0] or
        #         base_index == full_arch.shape[0] == full_channel_mask.shape[0])
        return x


class NasBatchNorm(HybridBlock):
    def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
                 use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
                 running_mean_initializer='zeros', running_variance_initializer='ones',
                 in_channels=0, inference_update_stat=False, **kwargs):
        super(NasBatchNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': momentum,
                        'fix_gamma': not scale, 'use_global_stats': use_global_stats}
        self.inference_update_stat = inference_update_stat
        if in_channels != 0:
            self.in_channels = in_channels

        self.gamma = self.params.get('gamma', grad_req='write' if scale else 'null',
                                     shape=(in_channels,), init=gamma_initializer,
                                     allow_deferred_init=True,
                                     differentiable=scale)
        self.beta = self.params.get('beta', grad_req='write' if center else 'null',
                                    shape=(in_channels,), init=beta_initializer,
                                    allow_deferred_init=True,
                                    differentiable=center)
        self.running_mean = self.params.get('running_mean', grad_req='null',
                                            shape=(in_channels,),
                                            init=running_mean_initializer,
                                            allow_deferred_init=True,
                                            differentiable=False)
        self.running_var = self.params.get('running_var', grad_req='null',
                                           shape=(in_channels,),
                                           init=running_variance_initializer,
                                           allow_deferred_init=True,
                                           differentiable=False)

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(NasBatchNorm, self).cast(dtype)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        if self.inference_update_stat:
             # TODO: for multi gpu, generate ndarray.array and do multiplication
            mean = x.mean(axis=(0, 2, 3))
            mean_expanded = F.expand_dims(F.expand_dims(F.expand_dims(mean, axis=0), axis=2), axis=3)
            var = F.square(F.broadcast_minus(x, mean_expanded)).mean(axis=(0, 2, 3))

            # TODO: remove debug codes
            # print("Passed running_mean: {}, raw running_mean: {}".format(running_mean, self.running_mean.data()))
            # print("Passed running_var: {}, raw running_var: {}".format(running_var, self.running_var.data()))
            # print("Passed gamme: {}, beta: {}".format(gamma, beta))
            # var_expanded = F.expand_dims(F.expand_dims(F.expand_dims(var, axis=0), axis=2), axis=3)

            # normalized_x = (x - mean_expanded) / F.sqrt(var_expanded)
            # print("Calculated mean: {}".format(mean))
            # print("Calculated var: {}".format(var))
            # print("Normalized x: {}".format(normalized_x))

            # rst = (x - mean_expanded) / F.sqrt(var_expanded) * \
            #       F.expand_dims(F.expand_dims(F.expand_dims(gamma, axis=0), axis=2), axis=3) + \
            #       F.expand_dims(F.expand_dims(F.expand_dims(beta, axis=0), axis=2), axis=3)
            # print("Target rst: {}".format(rst))

            # update running mean and var
            momentum = F.array([self._kwargs['momentum']])
            momentum_rest = F.array([1 - self._kwargs['momentum']])
            running_mean = F.add(F.multiply(self.running_mean.data(), momentum),
                                 F.multiply(mean, momentum_rest))
            running_var = F.add(F.multiply(self.running_var.data(), momentum),
                                F.multiply(var, momentum_rest))
            self.running_mean.set_data(running_mean)
            self.running_var.set_data(running_var)
            return F.BatchNorm(x, gamma, beta, mean, var, name='fwd', **self._kwargs)
        else:
            return F.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd', **self._kwargs)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


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
    """Test ShuffleNetCSBlock"""
    block_mode = 'ShuffleXception'
    dummy = nd.random.uniform(-1, 1, shape=(1, 4, 224, 224))
    cs_block = ShuffleNetCSBlock(input_channel=4, output_channel=16, mid_channel=16, ksize=3, 
                                 stride=1, block_mode=block_mode)
    cs_block.initialize()

    # generate channel mask
    channel_mask = nd.array([[1] * 10 + [0] * 6])
    from mxnet import autograd as ag
    with ag.record():
        rst = cs_block(dummy, channel_mask)
    rst.backward()
    params = cs_block.collect_params()
    not_selected_channels_grad_is_zero = True
    for param_name in params:
        if 'weight' in param_name:
            grad = params[param_name]._grad[0].asnumpy()
            print("{} shape: {}".format(param_name, grad.shape))
            if 'conv0' in param_name and block_mode == 'ShuffleNetV2':
                zero_grad = grad[10:, :, :, :]
                unique_grad = list(np.unique(zero_grad))
                not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                     len(unique_grad) == 1 and unique_grad[0] == 0
            elif 'conv1' in param_name:
                zero_grad = grad[10:, :, :, :]
                unique_grad = list(np.unique(zero_grad))
                not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                     len(unique_grad) == 1 and unique_grad[0] == 0
            elif 'conv2' in param_name:
                if block_mode == 'ShuffleNetV2':
                    zero_grad = grad[:, 10:, :, :]
                    unique_grad = list(np.unique(zero_grad))
                    not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                        len(unique_grad) == 1 and unique_grad[0] == 0
                else:
                    zero_grad = grad[10:, :, :, :]
                    unique_grad = list(np.unique(zero_grad))
                    not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                        len(unique_grad) == 1 and unique_grad[0] == 0
            elif 'conv3' in param_name:
                zero_grad = grad[10:, 10:, :, :]
                unique_grad = list(np.unique(zero_grad))
                not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                    len(unique_grad) == 1 and unique_grad[0] == 0
            elif 'conv4' in param_name:
                zero_grad = grad[10:, :, :, :]
                unique_grad = list(np.unique(zero_grad))
                not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                    len(unique_grad) == 1 and unique_grad[0] == 0
            elif 'conv5' in param_name:
                zero_grad = grad[:, 10:, :, :]
                unique_grad = list(np.unique(zero_grad))
                not_selected_channels_grad_is_zero = not_selected_channels_grad_is_zero and \
                                                    len(unique_grad) == 1 and unique_grad[0] == 0
    print("Not selected channels grads are zero: {}".format(not_selected_channels_grad_is_zero))
    print("Finished testing ShuffleNetCSBlock\n")

    """ Test ShuffleChannels """
    channel_shuffle = ShuffleChannels(mid_channel=4, groups=2)
    s = nd.ones([1, 8, 3, 3])
    s[:, 4:, :, :] *= 2
    s_project, s_main = channel_shuffle(s)
    print(s)
    print(s_project)
    print(s_main)
    print("Finished testing ShuffleChannels\n")

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
    # print(block0)
    # print(block1)
    print("Finished testing ShuffleNetV2 mode\n")

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
    # print(blockx0)
    # print(blockx1)
    print("Finished testing ShuffleXception mode\n")

    """ Test ChannelSelection """
    block_final_output_channel = 8
    candidate_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    max_channel = int(block_final_output_channel // 2 * candidate_scales[-1])
    tensor = nd.ones([1, max_channel, 14, 14])
    for i in range(max_channel):
        tensor[:, i, :, :] = i
    channel_selector = ChannelSelector(channel_number=block_final_output_channel)
    print(channel_selector)
    for i in range(4):
        global_channel_mask = random_channel_mask(stage_out_channels=[8, 160, 320, 640])
        local_channel_mask = nd.slice(global_channel_mask, begin=(i, None), end=(i + 1, None))
        selected_tensor = channel_selector(tensor, local_channel_mask)
        print(selected_tensor.shape)
    print("Finished testing ChannelSelector\n")

    """ Test BN with inference statistic update """
    bn = NasBatchNorm(inference_update_stat=True, in_channels=4)
    bn.initialize()
    bn.running_mean.set_data(bn.running_mean.data() + 1)
    mean, std = 5, 2
    for i in range(100):
        dummy = nd.random.normal(mean, std, shape=(10, 4, 5, 5))
        rst = bn(dummy)
        # print(dummy)
        # print(rst)
    print("Defined mean: {}, running mean: {}".format(mean, bn.running_mean.data()))
    print("Defined std: {}, running var: {}".format(std, bn.running_var.data()))
    print("Finished testing NasBatchNorm\n")


if __name__ == '__main__':
    main()
