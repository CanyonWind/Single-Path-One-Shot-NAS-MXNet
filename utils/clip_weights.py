import sys
import os
import argparse
import mxnet.ndarray as nd
import copy
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
from oneshot_nas_network import get_shufflenas_oneshot


def parse_args():
    parser = argparse.ArgumentParser(description='Verify Two Stream Attention model for IBDI.')
    parser.add_argument('--param-file', type=str, default='../models/oneshot-s+model-0000.params',
                        help='The absolute path of pre-trained model.')
    parser.add_argument('--block-choices', type=str, help='Block choices',
                        default='0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0')
    parser.add_argument('--channel-choices', type=str, help='Channel choices',
                        default='8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='Whether to follow MobileNet V3 last conv after pooling style.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext and ShuffleNas. default is false.')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='The relative path of the output folder.')
    parser.add_argument('--dtype', type=str, default='float16',
                        help='data type for training. default is float32')
    parser.add_argument('--ignored-params', type=str, default='running_mean, running_var',
                        help='the params listed here will not be clipped')
    parser.add_argument('--clip-to', type=float, default=-1,  # 1e-5,
                        help='|parameters| < clip_to will be clipped to clip_to')
    parser.add_argument('--clip-from', type=float, default=1e-5,  # -1,
                        help='|parameters| < clip_from will be clipped to 0')
    opt = parser.parse_args()
    return opt


def parse_str_list(str_list):
    num_list = str_list.split(',')
    return list(map(int, num_list))


def clip_weights():
    architecture = parse_str_list(args.block_choices)
    scale_ids = parse_str_list(args.channel_choices)
    net = get_shufflenas_oneshot(architecture=architecture, n_class=1000, scale_ids=scale_ids,
                                 last_conv_after_pooling=args.last_conv_after_pooling, use_se=args.use_se,
                                 channels_layout=args.channels_layout)
    
    net.cast(args.dtype)
    net.load_parameters(args.param_file)
    param_dict = net.collect_params()
    modified_count = 0

    assert args.clip_to == -1 or args.clip_from == -1
    for param_name in param_dict:
        if 'running' in param_name:
            continue
        param = param_dict[param_name].data().asnumpy()
        param_bak = copy.deepcopy(param)
        if args.clip_to != -1:
            mask = np.abs(param) < args.clip_to
            neg_mask = (param < 0) * mask
            local_count = np.sum(mask)
            if local_count == 0:
                continue
            modified_count += local_count
            param[mask] = args.clip_to
            param[neg_mask] *= -1
            print(param_name)
            print("before clipping")
            print(param_bak[mask])
            print("after clipping")
            print(param[mask])
        if args.clip_from != -1:
            mask = np.abs(param) < args.clip_from
            local_count = np.sum(mask)
            if local_count == 0:
                continue
            modified_count += local_count
            param[mask] = 0
            print(param_name)
            print("before clipping")
            print(param_bak[mask])
            print("after clipping")
            print(param[mask])
        param_dict[param_name].set_data(param)

    print("Totally modified {} weights.".format(modified_count))
    orig_file = args.param_file
    save_file = orig_file[:-7] + '-clip-to-{}.'.format(args.clip_to) + 'params' if args.clip_to != -1 else \
        orig_file[:-7] + '-clip-from-{}.'.format(args.clip_from) + 'params'
    print(save_file)
    net.save_parameters(save_file)


if __name__ == '__main__':
    args = parse_args()
    clip_weights()
