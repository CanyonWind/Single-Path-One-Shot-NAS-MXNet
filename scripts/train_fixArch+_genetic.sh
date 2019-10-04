export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode hybrid \
    --lr 1.3 --wd 0.00003 --lr-mode cosine --dtype float16\
    --num-epochs 360 --batch-size 256 --num-gpus 4 -j 8 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --use-se --last-conv-after-pooling \
    --save-dir params_shufflenas_oneshot+_genetic --logging-file ./logs/shufflenas_oneshot+_genetic.log
