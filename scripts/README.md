## Usage
**Training stage**

Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
# For the paper's searched fixed-structure model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode hybrid \
    --lr 0.5 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 240 --batch-size 256 --num-gpus 4 -j 8 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_fixarch --logging-file shufflenas_fixarch.log

# For supernet model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas --mode imperative \
    --lr 0.25 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 120 --batch-size 128 --num-gpus 4 -j 4 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_supernet --logging-file shufflenas_supernet.log \
    --epoch-start-cs 60 --cs-warm-up
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
