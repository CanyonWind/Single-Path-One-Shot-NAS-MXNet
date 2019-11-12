import os
import sys
import argparse
import numpy as np
from mxnet import nd
from calculate_flops import get_flops

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = dir_path[:dir_path.rfind('/')]
sys.path.append(parent_path)
from oneshot_nas_network import *
from oneshot_nas_blocks import *


def parse_arg():
    parser = argparse.ArgumentParser(description='Verify the test cases for Nas related modules.')
    parser.add_argument('--target', choices=['ShuffleNetCSBlock', 'ShuffleChannels', 'ShuffleNetV2Block',
                                             'ShuffleXceptionBlock', 'ChannelSelection', 'NasBN', 'ShuffleByConv',
                                             'ShuffleNasNetwork'],
                        help='Verify test case for which module (default: %(default)s)')
    parser.add_argument('--fix-arch', action='store_true', help='test the ShuffleNas network with fix arch or not.')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='Whether to follow MobileNet V3 last conv after pooling style.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in ShuffleNas. default is false.')
    parser.add_argument('--shuffle-by-conv', action='store_true',
                        help='Whether to replace reshape shuffling with 1 x 1 transpose conv')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    opts = parser.parse_args()
    return opts


def do_test():
    if args.target == 'ShuffleNetCSBlock':
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
    elif args.target == 'ShuffleChannels':
        """ Test ShuffleChannels """
        channel_shuffle = ShuffleChannels(mid_channel=4, groups=2)
        s = nd.ones([1, 8, 3, 3])
        s[:, 4:, :, :] *= 2
        s_project, s_main = channel_shuffle(s)
        print(s)
        print(s_project)
        print(s_main)
        print("Finished testing ShuffleChannels\n")
    elif args.target == 'ShuffleNetV2Block':
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
    elif args.target == 'ShuffleXceptionBlock':
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
        tempx1 = blockx1(tempx0)
        print(tempx0.shape)
        print(tempx1.shape)
        # print(blockx0)
        # print(blockx1)
        print("Finished testing ShuffleXception mode\n")
    elif args.target == 'ChannelSelection':
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
    elif args.target == 'NasBN':
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
    elif args.target == 'ShuffleByConv':
        """ Test Transpose Conv """
        dummy = nd.ones((1, 6, 5, 5))
        for i in range(6):
            dummy[:, i, :, :] = i
        transpose_conv = ShuffleChannelsConv(mid_channel=6 / 2)
        transpose_conv.initialize()
        transpose_conv.transpose_init()
        print(transpose_conv(dummy))
    elif args.target == 'ShuffleNasNetwork':
        if args.fix_arch:
            # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
            # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
            architecture = [0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0]
            scale_ids = [8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6]
            net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                         channels_layout=args.channels_layout,
                                         use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling)
        else:
            net = get_shufflenas_oneshot(use_se=args.use_se, last_conv_after_pooling=args.last_conv_after_pooling,
                                         channels_layout=args.channels_layout)

        """ Test customized initialization """
        net._initialize(force_reinit=True)
        print(net)

        """ Test ShuffleNasOneShot """
        test_data = nd.ones([5, 3, 224, 224])
        for step in range(1):
            if args.fix_arch:
                test_outputs = net(test_data)
                net.summary(test_data)
                net.hybridize()
            else:
                block_choices = net.random_block_choices(select_predefined_block=False, dtype='float32')
                full_channel_mask, _ = net.random_channel_mask(select_all_channels=False, dtype='float32')
                test_outputs = net(test_data, block_choices, full_channel_mask)
                net.summary(test_data, block_choices, full_channel_mask)
        if args.fix_arch:
            if not os.path.exists('./symbols'):
                os.makedirs('./symbols')
            net(test_data)
            net.export("./symbols/ShuffleNas_fixArch", epoch=0)
            flops, model_size = get_flops()
            print("Last conv after pooling: {}, use se: {}".format(args.last_conv_after_pooling, args.use_se))
            print("FLOPS: {}M, # parameters: {}M".format(flops, model_size))
        else:
            if not os.path.exists('./params'):
                os.makedirs('./params')
            net.save_parameters('./params/ShuffleNasOneshot-imagenet-supernet.params')

            """ Test generating random channels """
            epoch_start_cs = 30
            use_all_channels = True if epoch_start_cs != -1 else False
            dtype = 'float16'

            for epoch in range(120):
                if epoch == epoch_start_cs:
                    use_all_channels = False
                for batch in range(1):
                    full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=use_all_channels,
                                                                                 epoch_after_cs=epoch - epoch_start_cs,
                                                                                 dtype=dtype,
                                                                                 ignore_first_two_cs=True)
                    print("Epoch {}: {}".format(epoch, channel_choices))
    else:
        raise ValueError("Unrecognized test case target: {}".format(args.target))


if __name__ == '__main__':
    args = parse_arg()
    do_test()
