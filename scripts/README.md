## Explaination

- [train_oneshot.sh](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/scripts/train_oneshot.sh) for the model with **original** B/C choices, no se or last-conv-after-pooling.
- [train_oneshot+.sh](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/scripts/train_oneshot%2B.sh) for the model with **original** B/C choices, se and last-conv-after-pooling.
- [train_oneshot-s+.sh](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/scripts/train_oneshot-s%2B.sh) for the model with **searched** B/C choices, se and last-conv-after-pooling.
- [train_oneshot-s+_mobilenetv3.sh](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/scripts/train_oneshot-s%2B_mobilenetv3.sh) for the model with **searched** B/C choices, **ShuffleNetV2+ channels layout**, se and conv-after-last-pooling. This model has both fewer FLOPs and parameter amount than MobileNetV3

## Usage
**Training stage**

Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
# For the paper's searched fixed-structure model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --mode hybrid --lr 1.3 --wd 0.00003 --lr-mode cosine --dtype float16\
    --num-epochs 360 --batch-size 256 --num-gpus 4 -j 16 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --model ShuffleNas_fixArch \
    --block-choices '0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2' \
    --channel-choices '6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3' \
    --channels-layout OneShot \
    --save-dir params_shufflenas_oneshot+ --logging-file ./logs/shufflenas_oneshot+.log

# For supernet model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --mode imperative --lr 0.65 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 120 --batch-size 64 --num-gpus 1 -j 16 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --model ShuffleNas \
    --epoch-start-cs 60 --cs-warm-up --channels-layout OneShot \
    --save-dir params_shufflenas_supernet --logging-file ./logs/shufflenas_supernet.log
```

**Searching stage**

```shell
# Save a toy model of supernet model param, or put a well-trained supernet model under ./params/ folder and skip this step
python oneshot_nas_network.py

# do genetic search
python search_supernet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --batch-size 128 --num-gpus 4 -j 8 \
    --supernet_params ./params/ShuffleNasOneshot-imagenet-supernet.params \
    --dtype float32 --shuffle-train False \
    --search-mode genetic --comparison-model SinglePathOneShot \
    --topk 3 --search_iters 10 --update_bn_images 20000\
    --population_size 500 --retain_length 100 \
    --random_select 0.1 --mutate_chance 0.1
