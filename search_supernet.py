import mxnet as mx
from mxnet import gluon
from mxnet import nd
import os
import copy
import math
from gluoncv.model_zoo import get_model

from oneshot_nas_network import get_shufflenas_oneshot
from calculate_flops import get_flops
from oneshot_nas_blocks import NasBatchNorm


def generate_random_data_label(ctx=mx.gpu(0)):
    data = nd.random.uniform(-1, 1, shape=(1, 3, 224, 224), ctx=ctx)
    label = None
    return data, label


def get_data(rec_train='~/.mxnet/datasets/imagenet/rec/train.rec', 
             rec_train_idx='~/.mxnet/datasets/imagenet/rec/train.idx',
             rec_val='~/.mxnet/datasets/imagenet/rec/val.rec', 
             rec_val_idx='~/.mxnet/datasets/imagenet/rec/val.idx',
             input_size=224, crop_ratio=0.875, num_workers=4, batch_size=256, num_gpus=0):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]
    batch_size *= max(1, num_gpus)

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn


def get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask,
                 acc_top1=None, acc_top5=None, ctx=mx.cpu(), dtype='float32'):
    val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return top1


def set_nas_bn(net, inference_update_stat=False):
    if isinstance(net, NasBatchNorm):
        net.inference_update_stat = inference_update_stat
    elif len(net._children) != 0:
        for k, v in net._children.items():
            set_nas_bn(v,inference_update_stat=inference_update_stat)
    else:
        return


def update_bn(net, batch_fn, train_data, block_choices, full_channel_mask,
              ctx=mx.cpu(), dtype='float32', batch_size=256, update_bn_images=20000):
    print("Updating BN statistics...")
    set_nas_bn(net, inference_update_stat=True)
    for i, batch in enumerate(train_data):
        if (i + 1) * batch_size >= update_bn_images:
            break
        data, _ = batch_fn(batch, ctx)
        _ = [net(X.astype(dtype, copy=False), block_choices, full_channel_mask) for X in data]
    set_nas_bn(net, inference_update_stat=False)


def search_supernet(net, search_iters=2000, update_bn_images=20000, num_gpus=0, dtype='float32', batch_size=256,
                    flops_constraint=585, parameter_number_constraint=6.9):
    """
    Search within the pre-trained supernet.
    :param net:
    :param search_iters:
    :param update_bn_images:
    :param num_gpus:
    :param dtype:
    :param batch_size:
    :param flops_constraint: MobileNetV3-Large-1.0 [219M], MicroNet Challenge standard MobileNetV2-1.4 [585M]
    :param parameter_number_constraint: MobileNetV3-Large-1.0 [5.4M], MicroNet Challenge standard MobileNetV2-1.4 [6.9M]
    :return:
    """
    # TODO: use a heapq here to store top-5 models
    train_data, val_data, batch_fn = get_data(num_gpus=num_gpus, batch_size=batch_size)
    best_acc, best_acc_flop, best_acc_size = 0, 0, 0
    best_block_choices = None
    best_channel_choices = None
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    for i in range(search_iters):
        print("\nSearching iter: {}".format(i))
        block_choices = net.random_block_choices(select_predefined_block=False, dtype=dtype)
        full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False, dtype=dtype)
        
        # build fix_arch network and calculate flop
        fixarch_net = get_shufflenas_oneshot(block_choices.asnumpy(), channel_choices)
        fixarch_net.initialize()
        if not os.path.exists('./symbols'):
            os.makedirs('./symbols')
        fixarch_net.hybridize()
        dummy_data = nd.ones([1, 3, 224, 224])
        fixarch_net(dummy_data)
        fixarch_net.export("./symbols/ShuffleNas_fixArch", epoch=1)
        flops, model_size = get_flops()  # both in Millions
        normalized_score = flops / flops_constraint + model_size / parameter_number_constraint
        if normalized_score >= 1:
            print("[SKIPPED] Current model normalized score: {}.".format(normalized_score))
            print("[SKIPPED] Block choices:     {}".format(block_choices.asnumpy()))
            print("[SKIPPED] Channel choices:   {}".format(channel_choices))
            print('[SKIPPED] Flops:             {} MFLOPS'.format(flops))
            print('[SKIPPED] # parameters:      {} M'.format(model_size))
            continue

        # Update BN
        update_bn(net, batch_fn, train_data, block_choices, full_channel_mask, ctx=context, dtype=dtype,
                  batch_size=batch_size, update_bn_images=update_bn_images)
        print("BN statistics updated.")
        # Get validation accuracy
        val_acc = get_accuracy(net, val_data, batch_fn, block_choices, full_channel_mask,
                               acc_top1=acc_top1, acc_top5=acc_top5, ctx=context)
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_flop = flops
            best_acc_size = model_size
            best_block_choices = copy.deepcopy(block_choices.asnumpy())
            best_channel_choices = copy.deepcopy(channel_choices)
        print('-' * 40)
        print("Current model normalized score: {}.".format(normalized_score))
        print("Val accuracy:      {}".format(val_acc))
        print("Block choices:     {}".format(block_choices.asnumpy()))
        print("Channel choices:   {}".format(channel_choices))
        print('Flops:             {} MFLOPS'.format(flops))
        print('# parameters:      {} M'.format(model_size))
    
    print('-' * 40)
    print("Current model normalized score: {}.".
          format(best_acc_flop / flops_constraint + best_acc_size / parameter_number_constraint))
    print("Best val accuracy:    {}".format(best_acc))
    print("Block choices:        {}".format(best_block_choices))
    print("Channel choices:      {}".format(best_channel_choices))
    print('Flops:                {} MFLOPS'.format(best_acc_flop))
    print('# parameters:         {} M'.format(best_acc_size))


def main(num_gpus=4, supernet_params='./params/ShuffleNasOneshot-imagenet-supernet.params',
         dtype='float32', batch_size=256):
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    net = get_shufflenas_oneshot()
    net.load_parameters(supernet_params, ctx=context)
    print(net)
    search_supernet(net, search_iters=2000, num_gpus=num_gpus, dtype=dtype,
                    batch_size=batch_size, update_bn_images=20000)


if __name__ == '__main__':
    main(0)

