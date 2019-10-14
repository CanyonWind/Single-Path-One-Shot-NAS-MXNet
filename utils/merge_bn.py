
import sys
from mxnet import ndarray as nd
from oneshot_nas_network_nobn import get_shufflenas_oneshot as get_noBN_shufflenas_oneshot

sys.path.append('..')
from oneshot_nas_network import get_shufflenas_oneshot


def merge(conv_w, gamma, beta, running_mean, running_var):
    gamma_over_var = gamma / nd.sqrt(running_var + 1e-5)
    gamma_over_var_expanded = nd.reshape(gamma_over_var, (gamma_over_var.shape[0], 1, 1, 1))
    new_w = gamma_over_var_expanded * nd.cast(conv_w, 'float32')
    new_b = beta - running_mean * gamma_over_var
    return new_w, new_b


def merge_bn(param_file='../models/oneshot-s+model-0000.params'):
    architecture = [0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0]
    scale_ids = [8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6]
    net = get_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                 use_se=True, last_conv_after_pooling=True)
    net.cast('float16')
    net.load_parameters(param_file)
    param_dict = net.collect_params()
    nobn_net = get_noBN_shufflenas_oneshot(architecture=architecture, scale_ids=scale_ids,
                                           use_se=True, last_conv_after_pooling=True)
    nobn_net.initialize()
    nobn_param_dict = nobn_net.collect_params()


    merge_list = []
    param_list = list(param_dict.items())
    nobn_param_list = list(nobn_param_dict.keys())
    for i, key in enumerate(param_dict):
        if 'gamma' in key:
            merge_list.append({'conv_name': param_list[i - 1][0],
                               'bn_name': key[:key.rfind('_')],
                               'gamma': param_list[i][1].data(),
                               'beta': param_list[i + 1][1].data(),
                               'running_mean': param_list[i + 2][1].data(),
                               'running_var': param_list[i + 3][1].data(),
                               })
        if 'batchnorm' not in key:
            nobn_param_dict[key.replace('fix0', 'fix1')].set_data(param_dict[key].data())
            nobn_param_list.remove(key.replace('fix0', 'fix1'))

    for info in merge_list:
        new_w, new_b = merge(param_dict[info['conv_name']].data(), info['gamma'],
                             info['beta'], info['running_mean'], info['running_var'])
        nobn_param_dict[info['conv_name'].replace('fix0', 'fix1')].set_data(new_w)
        nobn_param_dict[info['conv_name'][:-6].replace('fix0', 'fix1') + 'bias'].set_data(new_b)

    nobn_net.save_parameters('../models/oneshot-s+model-noBN-0000.params')


if __name__ == '__main__':
    merge_bn()
