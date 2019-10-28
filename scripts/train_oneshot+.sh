export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --mode hybrid --lr 1.3 --wd 0.00003 --lr-mode cosine --dtype float16\
    --num-epochs 360 --batch-size 256 --num-gpus 4 -j 16 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --model ShuffleNas_fixArch \
    --block-choices '0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2' \
    --channel-choices '6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3' \
    --use-se --last-conv-after-pooling --channels-layout OneShot \
    --save-dir params_shufflenas_oneshot+ --logging-file ./logs/shufflenas_oneshot+.log
