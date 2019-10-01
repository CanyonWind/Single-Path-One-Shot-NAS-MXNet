export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode hybrid \
    --lr 0.4875 --wd 0.00003 --lr-mode cosine --dtype float16\
    --num-epochs 360 --batch-size 128 --num-gpus 3 -j 8 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --use-se --last-conv-after-pooling \
    --save-dir params_shufflenas_oneshot+ --logging-file shufflenas_oneshot+.log
