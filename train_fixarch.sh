export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode hybrid \
    --lr 0.5 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 240 --batch-size 256 --num-gpus 4 -j 8 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_fixarch --logging-file shufflenas_fixarch.log \
    --use-se --last-conv-after-pooling
