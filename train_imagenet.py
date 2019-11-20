import argparse, time, logging, os, math, sys
import multiprocessing
from multiprocessing import Value
from ctypes import c_bool

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler

from oneshot_nas_network import get_shufflenas_oneshot
from oneshot_nas_blocks import NasBatchNorm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + '/utils')
from get_distribution import get_shufflenas_flop_param as get_flop_param_forward
from lookup_table import get_flop_params as get_flop_param_lookup
from lookup_table import load_lookup_table


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--rec-train', type=str, default='.//imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-train-idx', type=str, default='./imagenet/rec/train.idx',
                        help='the index of training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                        help='the index of validation data')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use-se', action='store_true',
                        help='use SE layers or not in resnext and ShuffleNas. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--use-all-blocks', action='store_true',
                        help='whether to use all the choice blocks.')
    parser.add_argument('--use-all-channels', action='store_true',
                        help='whether to use all the channels.')
    parser.add_argument('--epoch-start-cs', type=int, default=60,
                        help='Epoch id for starting Channel selection.')
    parser.add_argument('--last-conv-after-pooling', action='store_true',
                        help='Whether to follow MobileNet V3 last conv after pooling style.')
    parser.add_argument('--cs-warm-up', action='store_true',
                        help='Whether to do warm up for Channel Selection so that gradually selects '
                             'larger range of channels')
    parser.add_argument('--channels-layout', type=str, default='OneShot',
                        help='The mode of channels layout: [\'ShuffleNetV2+\', \'OneShot\']')
    parser.add_argument('--ignore-first-two-cs', action='store_true',
                        help='whether to ignore the first two channel selection scales. This will be stable for no'
                             'SE supernet training.')
    parser.add_argument('--reduced-dataset-scale', type=int, default=1,
                        help='How many times the dataset would be reduced, so that in each epoch '
                             'only num_batches / reduced_dataset_scale batches will be trained.')
    parser.add_argument('--train-upper-constraints', type=str, default='',
                        help="training constraints: 1) empty str: no constraint\n"
                             "                      2) a str of 'flops-330-params-4.5', which means searching subnets\n"
                             "                         with less than 330M FLOPs and less than 4.5M parameters.")
    parser.add_argument('--flop-param-method', type=str, default='lookup_table',
                        help='How to calculate flops and params. Choose from [symbol, lookup_table].'
                             'The last one runs hundreds faster than the former one.')
    parser.add_argument('--block-choices', type=str,
                        default='0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2',
                        help='Block choices')
    parser.add_argument('--channel-choices', type=str,
                        default='6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3',
                        help='Channel choices')

    opt = parser.parse_args()
    return opt


def parse_str_list(str_list):
    num_list = str_list.split(',')
    return list(map(int, num_list)) 


def main():
    opt = parse_args()

    filehandler = logging.FileHandler(opt.logging_file)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 1000
    num_training_samples = 1281167

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    # epoch_start_cs controls that before this epoch, use all channels, while, after this epoch, use channel selection.
    if opt.epoch_start_cs != -1:
        opt.use_all_channels = True

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = num_training_samples // batch_size // opt.reduced_dataset_scale

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    model_name = opt.model

    kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
    if opt.use_gn:
        from gluoncv.nn import GroupNorm
        kwargs['norm_layer'] = GroupNorm
    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = opt.use_se

    if opt.last_gamma:
        kwargs['last_gamma'] = True

    optimizer = 'nag'
    optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    if model_name == 'ShuffleNas_fixArch':
        architecture = parse_str_list(opt.block_choices)
        scale_ids = parse_str_list(opt.channel_choices)
        net = get_shufflenas_oneshot(architecture=architecture, n_class=classes, scale_ids=scale_ids, use_se=opt.use_se,
                                     last_conv_after_pooling=opt.last_conv_after_pooling,
                                     channels_layout=opt.channels_layout)
    elif model_name == 'ShuffleNas':
        net = get_shufflenas_oneshot(n_class=classes, use_all_blocks=opt.use_all_blocks, use_se=opt.use_se,
                                     last_conv_after_pooling=opt.last_conv_after_pooling,
                                     channels_layout=opt.channels_layout)
    else:
        net = get_model(model_name, **kwargs)

    net.cast(opt.dtype)
    if opt.resume_params is not '':
        net.load_parameters(opt.resume_params, ctx=context)

    # teacher model for distillation training
    if opt.teacher is not None and opt.hard_weight < 1.0:
        teacher_name = opt.teacher
        teacher = get_model(teacher_name, pretrained=True, classes=classes, ctx=context)
        teacher.cast(opt.dtype)
        distillation = True
    else:
        distillation = False

    # Two functions for reading data from record file or raw images
    def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
        rec_train = os.path.expanduser(rec_train)
        rec_train_idx = os.path.expanduser(rec_train_idx)
        rec_val = os.path.expanduser(rec_val)
        rec_val_idx = os.path.expanduser(rec_val_idx)
        jitter_param = 0.4
        lighting_param = 0.1
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))
        mean_rgb = [123.68, 116.779, 103.939]
        std_rgb = [58.393, 57.12, 57.375]

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

    def get_data_loader(data_dir, batch_size, num_workers):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        jitter_param = 0.4
        lighting_param = 0.1
        input_size = opt.input_size
        crop_ratio = opt.crop_ratio if opt.crop_ratio > 0 else 0.875
        resize = int(math.ceil(input_size / crop_ratio))

        def batch_fn(batch, ctx):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            return data, label

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                        saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

        train_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
        val_data = gluon.data.DataLoader(
            imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        return train_data, val_data, batch_fn

    if opt.use_rec:
        train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_train_idx,
                                                    opt.rec_val, opt.rec_val_idx,
                                                    batch_size, num_workers)
    else:
        train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

    if opt.mixup:
        train_metric = mx.metric.RMSE()
    else:
        train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = opt.save_frequency
    if opt.save_dir and save_frequency:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_frequency = 0

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            res.append(lam*y1 + (1-lam)*y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            smoothed.append(res)
        return smoothed

    def make_divisible(x, divisible_by=8):
        return int(np.ceil(x * 1. / divisible_by) * divisible_by)

    def set_nas_bn(net, inference_update_stat=False):
        if isinstance(net, NasBatchNorm):
            net.inference_update_stat = inference_update_stat
        elif len(net._children) != 0:
            for k, v in net._children.items():
                set_nas_bn(v, inference_update_stat=inference_update_stat)
        else:
            return

    def update_bn(net, batch_fn, train_data, block_choices, full_channel_mask,
                  ctx=[mx.cpu()], dtype='float32', batch_size=256, update_bn_images=20000):
        train_data.reset()
        # Updating bn needs the model to be float32
        net.cast('float32')
        full_channel_masks = [full_channel_mask.as_in_context(ctx_i) for ctx_i in ctx]
        set_nas_bn(net, inference_update_stat=True)
        for i, batch in enumerate(train_data):
            if (i + 1) * batch_size * len(ctx) >= update_bn_images:
                break
            data, _ = batch_fn(batch, ctx)
            _ = [net(X.astype('float32', copy=False), block_choices.astype('float32', copy=False),
                     channel_mask.astype('float32', copy=False)) for X, channel_mask in zip(data, full_channel_masks)]
        set_nas_bn(net, inference_update_stat=False)
        net.cast(dtype)

    def test(ctx, val_data, epoch):
        if model_name == 'ShuffleNas':
            # For evaluating validation accuracy, random select block and channels and update bn stats
            block_choices = net.random_block_choices(select_predefined_block=False, dtype=opt.dtype)
            if opt.cs_warm_up:
                # TODO: edit in the issue, readme and medium article that
                #  bn stat needs to be updated before verifying val acc
                full_channel_mask, channel_choices = net.random_channel_mask(select_all_channels=False,
                                                                             epoch_after_cs=epoch - opt.epoch_start_cs,
                                                                             dtype=opt.dtype,
                                                                             ignore_first_two_cs=opt.ignore_first_two_cs)
            else:
                full_channel_mask, _ = net.random_channel_mask(select_all_channels=False,
                                                               dtype=opt.dtype,
                                                               ignore_first_two_cs=opt.ignore_first_two_cs)
            update_bn(net, batch_fn, train_data, block_choices, full_channel_mask, ctx,
                      dtype=opt.dtype, batch_size=batch_size)
        else:
            block_choices, full_channel_mask = None, None

        if opt.use_rec:
            val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            if model_name == 'ShuffleNas':
                full_channel_masks = [full_channel_mask.as_in_context(ctx_i) for ctx_i in ctx]
                outputs = [net(X.astype(opt.dtype, copy=False), block_choices, channel_mask)
                           for X, channel_mask in zip(data, full_channel_masks)]
            else:
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return 1-top1, 1-top5

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.resume_params is '':
            if 'ShuffleNas' in model_name:
                net._initialize(ctx=ctx)
            else:
                net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        if opt.resume_states is not '':
            trainer.load_states(opt.resume_states)

        if opt.label_smoothing or opt.mixup:
            sparse_label_loss = False
        else:
            sparse_label_loss = True
        if distillation:
            L = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=opt.temperature,
                                                                 hard_weight=opt.hard_weight,
                                                                 sparse_label=sparse_label_loss)
        else:
            L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

        best_val_score = 1

        def train_epoch(pool=None, pool_lock=None, shared_finished_flag=None, use_pool=False):
            btic = time.time()
            for i, batch in enumerate(train_data):
                if i == num_batches:
                    if use_pool:
                        shared_finished_flag.value = True
                    return
                data, label = batch_fn(batch, ctx)

                if opt.mixup:
                    lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                    if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                        lam = 1
                    data = [lam*X + (1-lam)*X[::-1] for X in data]

                    if opt.label_smoothing:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label, classes, lam, eta)

                elif opt.label_smoothing:
                    hard_label = label
                    label = smooth(label, classes)

                if distillation:
                    teacher_prob = [nd.softmax(teacher(X.astype(opt.dtype, copy=False)) / opt.temperature) \
                                    for X in data]

                with ag.record():
                    if model_name == 'ShuffleNas' and use_pool:
                        cand = None
                        while cand is None:
                            if len(pool) > 0:
                                with pool_lock:
                                    cand = pool.pop()
                                    logger.debug('[Trainer]' + '-' * 40)
                                    logger.debug("Time: {}".format(time.time()))
                                    logger.debug("Block choice: {}".format(cand['block_list']))
                                    logger.debug("Channel choice: {}".format(cand['channel_list']))
                                    logger.debug("Flop: {}M, param: {}M".format(cand['flops'], cand['model_size']))
                            else:
                                time.sleep(1)

                        full_channel_masks = [cand['channel'].as_in_context(ctx_i) for ctx_i in ctx]
                        outputs = [net(X.astype(opt.dtype, copy=False), cand['block'], channel_mask)
                                   for X, channel_mask in zip(data, full_channel_masks)]
                    elif model_name == 'ShuffleNas':
                        block_choices = net.random_block_choices(select_predefined_block=False, dtype=opt.dtype)
                        if opt.cs_warm_up:
                            full_channel_mask, channel_choices = net.random_channel_mask(
                                select_all_channels=opt.use_all_channels,
                                epoch_after_cs=epoch - opt.epoch_start_cs,
                                dtype=opt.dtype,
                                ignore_first_two_cs=opt.ignore_first_two_cs)
                        else:
                            full_channel_mask, channel_choices = net.random_channel_mask(
                                select_all_channels=opt.use_all_channels,
                                dtype=opt.dtype,
                                ignore_first_two_cs=opt.ignore_first_two_cs)

                        full_channel_masks = [full_channel_mask.as_in_context(ctx_i) for ctx_i in ctx]
                        outputs = [net(X.astype(opt.dtype, copy=False), block_choices, channel_mask)
                                   for X, channel_mask in zip(data, full_channel_masks)]
                    else:
                        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]

                    if distillation:
                        loss = [L(yhat.astype('float32', copy=False),
                                  y.astype('float32', copy=False),
                                  p.astype('float32', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob)]
                    else:
                        loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size, ignore_stale_grad=True)

                if opt.mixup:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                    for out in outputs]
                    train_metric.update(label, output_softmax)
                else:
                    if opt.label_smoothing:
                        train_metric.update(hard_label, outputs)
                    else:
                        train_metric.update(label, outputs)

                if opt.log_interval and not (i+1)%opt.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                                epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                                train_metric_name, train_metric_score, trainer.learning_rate))
                    btic = time.time()
            return

        def pool_maintainer(pool, pool_lock, shared_finished_flag, upper_flops=sys.maxsize, upper_params=sys.maxsize):
            lookup_table = None
            if opt.flop_param_method == 'lookup_table':
                lookup_table = load_lookup_table(opt.use_se, opt.last_conv_after_pooling,
                                                 opt.channels_layout, nas_root='./')
            while True:
                if shared_finished_flag.value:
                    break
                if len(pool) < 5:
                    candidate = dict()
                    block_choices, block_choices_list = net.random_block_choices(select_predefined_block=False,
                                                                                 dtype=opt.dtype,
                                                                                 return_choice_list=True)
                    if opt.cs_warm_up:
                        full_channel_mask, channel_choices_list = net.random_channel_mask(
                            select_all_channels=opt.use_all_channels,
                            epoch_after_cs=epoch - opt.epoch_start_cs,
                            dtype=opt.dtype,
                            ignore_first_two_cs=opt.ignore_first_two_cs,)
                    else:
                        full_channel_mask, channel_choices_list = net.random_channel_mask(
                            select_all_channels=opt.use_all_channels,
                            dtype=opt.dtype,
                            ignore_first_two_cs=opt.ignore_first_two_cs)

                    if opt.flop_param_method == 'symbol':
                        flops, model_size, _, _ = \
                            get_flop_param_forward(block_choices_list, channel_choices_list,
                                                 use_se=opt.use_se, last_conv_after_pooling=opt.last_conv_after_pooling,
                                                 channels_layout=opt.channels_layout)
                    elif opt.flop_param_method == 'lookup_table':
                        flops, model_size = get_flop_param_lookup(block_choices_list, channel_choices_list, lookup_table)
                    else:
                        raise ValueError('Unrecognized flop param calculation method: {}'.format(opt.flop_param_method))

                    candidate['block'] = block_choices
                    candidate['channel'] = full_channel_mask
                    candidate['block_list'] = block_choices_list
                    candidate['channel_list'] = channel_choices_list
                    candidate['flops'] = flops
                    candidate['model_size'] = model_size

                    if flops > upper_flops or model_size > upper_params:
                        continue

                    with pool_lock:
                        pool.append(candidate)
                        logger.debug("[Maintainer] Add one good candidate. currently pool size: {}".format(len(pool)))

        manager = multiprocessing.Manager()
        cand_pool = manager.list()
        p_lock = manager.Lock()
        for epoch in range(opt.resume_epoch, opt.num_epochs):
            if epoch >= opt.epoch_start_cs:
                opt.use_all_channels = False
            tic = time.time()
            if opt.use_rec:
                train_data.reset()
            train_metric.reset()

            if model_name == 'ShuffleNas' and opt.train_upper_constraints:
                constraints = opt.train_upper_constraints.split('-')
                # opt.train_upper_constraints = 'flops-300-params-4.5'
                assert len(constraints) == 4 and constraints[0] == 'flops' and constraints[2] == 'params'
                upper_flops = float(constraints[1]) if float(constraints[1]) != 0 else sys.maxsize
                upper_params = float(constraints[3]) if float(constraints[3]) != 0 else sys.maxsize
                finished = Value(c_bool, False)
                logger.debug("===== DEBUG ======\n"
                             "Train SuperNet with Flops less than {}, params less than {}"
                             .format(upper_flops, upper_params))
                pool_process = multiprocessing.Process(target=pool_maintainer,
                                                       args=[cand_pool, p_lock, finished, upper_flops, upper_params])
                pool_process.start()
                train_epoch(pool=cand_pool, pool_lock=p_lock, shared_finished_flag=finished, use_pool=True)
                pool_process.join()
            else:
                logger.debug("===== DEBUG ======\n"
                             "Train SuperNet with no constraint")
                train_epoch()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * num_batches / (time.time() - tic))

            err_top1_val, err_top5_val = test(ctx, val_data, epoch)

            logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time()-tic))
            logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f' % (epoch, err_top1_val, err_top5_val))

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
                trainer.save_states('%s/%.4f-imagenet-%s-%d-best.states' % (save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, opt.num_epochs-1))
            trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, opt.num_epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
        if distillation:
            teacher.hybridize(static_alloc=True, static_shape=True)
    print(net)
    train(context)
   

if __name__ == '__main__':
    main()
