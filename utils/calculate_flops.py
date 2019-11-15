# -*- coding: utf-8 -*-
"""
File Name: calculate_flops.py
Author: liangdepeng
mail: liangdepeng@gmail.com
Cloned from https://github.com/Ldpe2G/DeepLearningForFun/tree/master/MXNet-Python/CalculateFlopsTool

Usage:
   python calculate_flops.py -s symbols/ShuffleNas_fixArch-symbol.json -ds  data,1,3,224,224 -ls prob_label,1,1000 (-norelubn)
"""

import mxnet as mx
import argparse
import numpy as np
import json
import re

import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from micronet_counting import *


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-ds', '--data_shapes', type=str, nargs='+', default=['data,1,3,224,224'],
                        help='data_shapes, format: arg_name,s1,s2,...,sn, example: data,1,3,224,224')
    parser.add_argument('-ls', '--label_shapes', type=str, nargs='+', default=['output,1,1000'],
                        help='label_shapes, format: arg_name,s1,s2,...,sn, example: label,1,1,224,224')
    parser.add_argument('-s', '--symbol_path', type=str,
                        default='../quantization/model/ShuffleNas_fixArch-symbol.json', help='')
    parser.add_argument('-norelubn', action='store_true', help='Whether to calculate relu and bn.')
    return parser.parse_args()


def product(tu):
    """Calculates the product of a tuple"""
    prod = 1
    for x in tu:
        prod = prod * x
    return prod


def get_internal_label_info(internal_sym, label_shapes):
    if label_shapes:
        internal_label_shapes = filter(lambda shape: shape[0] in internal_sym.list_arguments(), label_shapes)
        if internal_label_shapes:
            internal_label_names = [shape[0] for shape in internal_label_shapes]
            return internal_label_names, internal_label_shapes
    return None, None


def get_flops(norelubn=True, size_in_mb=False, mode='wild', micronet_include_bn=False,
              symbol_path='./symbols/ShuffleNas_fixArch-symbol.json',
              data_names=['data'], data_shapes=[('data', (1, 3, 224, 224))],
              label_names=['output'], label_shapes=[('output', (1, 1000))]):
    if mode == 'micronet':
        norelubn = False

    devs = [mx.cpu()]
    sym = mx.sym.load(symbol_path)
    if len(label_names) == 0:
        label_names = None
    model = mx.mod.Module(context=devs, symbol=sym, data_names=data_names, label_names=None)
    model.bind(data_shapes=data_shapes, for_training=False)

    arg_params = model._exec_group.execs[0].arg_dict

    conf = json.loads(sym.tojson())
    nodes = conf["nodes"]

    total_flops = 0.

    # For reusing micronet challenge flop calculator, a list of all ops is needed.
    # each item in the list will be a tuple of (name, namedTuple)
    all_ops = []

    for node in nodes:
        op = node["op"]
        layer_name = node["name"][:-4]
        attrs = None
        if "param" in node:
            attrs = node["param"]
        elif "attrs" in node:
            attrs = node["attrs"]
        else:
            attrs = {}

        if op == 'Convolution':
            internal_sym = sym.get_internals()[layer_name + '_fwd' + '_output']
            internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

            shape_dict = {}
            for k, v in data_shapes:
                shape_dict[k] = v
            if internal_label_shapes != None:
                for k, v in internal_label_shapes:
                    shape_dict[k] = v

            _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
            out_shape = out_shapes[0]

            num_group = 1
            if "num_group" in attrs:
                num_group = int(attrs['num_group'])

            # support conv1d NCW and conv2d NCHW layout
            out_shape_produt = out_shape[2] if len(out_shape) == 3 else out_shape[2] * out_shape[3]
            total_flops += out_shape_produt * product(arg_params[layer_name + '_weight'].shape) * data_shapes[0][1][0]

            if layer_name + "_bias" in arg_params:
                total_flops += product(out_shape)

            del shape_dict

            # ---------------- Add to micronet all_ops -------------------
            # infer input shape
            assert out_shape[2] == out_shape[3]
            if '2' in attrs['stride']:
                input_size = out_shape[2] * 2
            else:
                input_size = out_shape[2]

            # generate (k, k, c_in, c_out) kernel shape from (c_out, c_in, k, k)
            mx_kernel_shape = arg_params[layer_name + '_weight'].shape
            kernel_shape = list(reversed(mx_kernel_shape))

            # others
            strides = [int(i) for i in attrs['stride'][1:-1].split(',')]
            padding = 'same'
            use_bias = layer_name + "_bias" in arg_params

            if int(attrs['num_group']) == 1:
                all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB').
                                replace('_squeeze_', '_').replace('_excitation_', '_'),
                                Conv2D(input_size=input_size,
                                       kernel_shape=kernel_shape,
                                       strides=strides,
                                       padding=padding,
                                       use_bias=use_bias,
                                       activation=None,
                                       sparsity=0)))
            else:
                # Depthwise conv. MXNet is assuming the c_in should be 1, so swapping axis here is processed
                assert kernel_shape[2] == 1 and kernel_shape[3] == int(attrs['num_group'])
                kernel_shape[2] = kernel_shape[3]
                kernel_shape[3] = 1

                all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB'),
                                DepthWiseConv2D(input_size=input_size,
                                                kernel_shape=kernel_shape,
                                                strides=strides,
                                                padding=padding,
                                                use_bias=use_bias,
                                                activation=None)))

        if op == 'Deconvolution':
            input_layer_name = nodes[node["inputs"][0][0]]["name"]

            internal_sym = sym.get_internals()[input_layer_name + '_output']
            internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

            shape_dict = {}
            for k, v in data_shapes:
                shape_dict[k] = v
            if internal_label_shapes != None:
                for k, v in internal_label_shapes:
                    shape_dict[k] = v

            _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
            input_shape = out_shapes[0]

            num_group = 1
            if "num_group" in attrs:
                num_group = int(attrs['num_group'])

            total_flops += input_shape[2] * input_shape[3] * product(arg_params[layer_name + '_weight'].shape) * \
                           data_shapes[0][1][0] / num_group

            del shape_dict

            if layer_name + "_bias" in arg_params:
                internal_sym = sym.get_internals()[layer_name + '_fwd' + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym,
                                                                                      internal_label_shapes)

                shape_dict = {}
                for k, v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k, v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                out_shapes = out_shapes[0]

                total_flops += product(out_shape)

                del shape_dict

        if op == 'FullyConnected':
            total_flops += product(arg_params[layer_name + '_weight'].shape) * data_shapes[0][1][0]

            if layer_name + '_bias' in arg_params:
                num_hidden = int(attrs['num_hidden'])
                total_flops += num_hidden * data_shapes[0][1][0]

        if op == 'Pooling':
            if "global_pool" in attrs and attrs['global_pool'] == 'True':
                input_layer_name = nodes[node["inputs"][0][0]]["name"]

                if input_layer_name == 'data':
                    internal_sym = sym.get_internals()[input_layer_name]
                else:
                    internal_sym = sym.get_internals()[input_layer_name + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

                shape_dict = {}
                for k, v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k, v in internal_label_shapes:
                        shape_dict[k] = v

                _, input_shapes, _ = internal_sym.infer_shape(**shape_dict)
                input_shape = input_shapes[0]

                total_flops += product(input_shape)

                # ---------------- Add to micronet all_ops -------------------
                assert input_shape[2] == input_shape[3]
                all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB'),
                                GlobalAvg(input_size=input_shape[2],
                                          n_channels=input_shape[1])))

            else:
                internal_sym = sym.get_internals()[layer_name + '_fwd' + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

                shape_dict = {}
                for k, v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k, v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                out_shape = out_shapes[0]

                n = '\d+'
                kernel = [int(i) for i in re.findall(n, attrs['kernel'])]

                total_flops += product(out_shape) * product(kernel)

            del shape_dict

        if op == 'Reshape':
            layer_name = node["name"]
            if layer_name[-1] == '0':
                # Channel Shuffle is implemented by reshape-swapaxes-reshape, so that we use the last reshape as
                # the indicator for one whole Channel Shuffle operator
                continue
            internal_sym = sym.get_internals()[layer_name + '_output']
            internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

            shape_dict = {}
            for k, v in data_shapes:
                shape_dict[k] = v
            if internal_label_shapes != None:
                for k, v in internal_label_shapes:
                    shape_dict[k] = v

            _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
            out_shape = out_shapes[0]

            # ---------------- Add to micronet all_ops -------------------
            # As FAQ claimed, Channel Shuffle can be treated as a 1 x 1 conv with same input / output channel C and
            # (C - 1) / C sparsity.
            # infer input shape
            input_size = out_shape[2]

            # generate (k, k, c_in, c_out) kernel shape from (c_out, c_in, k, k)
            kernel_shape = [1, 1, out_shape[1], out_shape[1]]

            # others
            strides = [1, 1]
            padding = 'same'
            use_bias = False

            all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB')[:-9],
                            Conv2D(input_size=input_size,
                                   kernel_shape=kernel_shape,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=use_bias,
                                   activation=None,
                                   sparsity=(out_shape[1] - 1) / out_shape[1])))
        if not norelubn:
            if op == 'Activation':
                if attrs['act_type'] == 'relu':
                    internal_sym = sym.get_internals()[layer_name + '_fwd' + '_output']
                    internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

                    shape_dict = {}
                    for k, v in data_shapes:
                        shape_dict[k] = v
                    if internal_label_shapes != None:
                        for k, v in internal_label_shapes:
                            shape_dict[k] = v

                    _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                    out_shape = out_shapes[0]

                    total_flops += product(out_shape)

                    del shape_dict

                    # ---------------- Add to micronet all_ops -------------------
                    # only nn.Activation('relu') was used.
                    all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB'),
                                    Activation(output_shape=list(out_shape),
                                               activation_name='relu')))
            elif op in ['_plus_scalar', 'clip', '_div_scalar', 'elemwise_mul']:
                # For hard-swish calculation, each related operation is put into the namedTuple Activation too.
                # x * (F.clip(x + 3, 0, 6) / 6.)
                all_ops.append((node["name"][40:].replace('shufflenetblock', 'SNB').
                                replace('hard_swish', '_clip').replace('hard_sigmoid', 'clip').
                                replace('hardswish', 'HSwish').replace('hardsigmoid', 'HSigmoid').
                                replace('_plusscalar', 'plus').replace('_divscalar', 'div').replace('_mul', 'mul'),
                                Activation(output_shape=list(out_shape),
                                           activation_name=op)))

            elif op == 'BatchNorm' or op == 'NasBatchNorm' and micronet_include_bn:
                internal_syms = sym.get_internals()
                internal_sym = sym.get_internals()[layer_name + '_fwd' + '_output']
                internal_label_names, internal_label_shapes = get_internal_label_info(internal_sym, label_shapes)

                shape_dict = {}
                for k, v in data_shapes:
                    shape_dict[k] = v
                if internal_label_shapes != None:
                    for k, v in internal_label_shapes:
                        shape_dict[k] = v

                _, out_shapes, _ = internal_sym.infer_shape(**shape_dict)
                out_shape = out_shapes[0]
                total_flops += product(out_shape) * 3  # mean, variance, (blob - mean) / variance * beta + gamma

    model_size = 0.0
    if label_names == None:
        label_names = list()
    for k, v in arg_params.items():
        if k not in data_names and k not in label_names:
            if size_in_mb:
                model_size += product(v.shape) * np.dtype(v.dtype()).itemsize / 1024 / 1024  # model size in MB
            else:
                model_size += product(v.shape) / 1000000  # number of parameters (Million)

    if mode == 'micronet':
        return all_ops
    else:
        return total_flops / 1000000, model_size


def main():
    args = parse_args()
    data_shapes = list()
    data_names = list()
    if args.data_shapes is not None and len(args.data_shapes) > 0:
        for shape in args.data_shapes:
            items = shape.replace('\'', '').replace('"', '').split(',')
            data_shapes.append((items[0], tuple([int(s) for s in items[1:]])))
            data_names.append(items[0])

    label_shapes = None
    label_names = list()
    if args.label_shapes is not None and len(args.label_shapes) > 0:
        label_shapes = list()
        for shape in args.label_shapes:
            items = shape.replace('\'', '').replace('"', '').split(',')
            label_shapes.append((items[0], tuple([int(s) for s in items[1:]])))
            label_names.append(items[0])
    flops, model_size = get_flops(norelubn=True, size_in_mb=False, symbol_path=args.symbol_path,
                                  data_names=data_names, data_shapes=data_shapes,
                                  label_names=label_names, label_shapes=label_shapes)
    print('flops: ', str(flops), ' MFLOPS')
    print('model size: ', str(model_size), ' M')

    all_ops = get_flops(norelubn=True, size_in_mb=False, symbol_path=args.symbol_path, mode='micronet',
                        data_names=data_names, data_shapes=data_shapes,
                        label_names=label_names, label_shapes=label_shapes)
    for op in all_ops:
        op[0].replace('shufflenetblock', 'SNB')

    counter = MicroNetCounter(all_ops, add_bits_base=32, mul_bits_base=16)

    # Constants
    INPUT_BITS = 16
    ACCUMULATOR_BITS = 32
    PARAMETER_BITS = INPUT_BITS
    SUMMARIZE_BLOCKS = True

    counter.print_summary(0, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=False)


if __name__ == '__main__':
    main()
