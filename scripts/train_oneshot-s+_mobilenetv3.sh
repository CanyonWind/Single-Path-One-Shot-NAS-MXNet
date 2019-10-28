export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --mode hybrid --lr 1.3 --wd 0.00003 --lr-mode cosine --dtype float16\
    --num-epochs 360 --batch-size 256 --num-gpus 4 -j 16 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --model ShuffleNas_fixArch \
    --block-choices '0, 0, 0, 1, 0, 0, 1, 0, 3, 2, 0, 1, 2, 2, 1, 2, 0, 0, 2, 0' \
    --channel-choices '8, 7, 6, 8, 5, 7, 3, 4, 2, 4, 2, 3, 4, 5, 6, 6, 3, 3, 4, 6' \
    --use-se --channels-layout ShuffleNetV2+ \
    --save-dir params_shufflenas_oneshot+_genetic --logging-file ./logs/shufflenas_oneshot+_genetic.log
