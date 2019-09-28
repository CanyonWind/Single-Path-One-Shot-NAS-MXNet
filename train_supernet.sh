export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas --mode imperative \
    --lr 0.25 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 120 --batch-size 128 --num-gpus 4 -j 4 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_supernet --logging-file shufflenas_supernet.log \
    --epoch-start-cs 0 --use-se
