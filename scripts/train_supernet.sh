export MXNET_SAFE_ACCUMULATION=1

python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --mode imperative --lr 0.65 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 120 --batch-size 64 --num-gpus 1 -j 16 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --model ShuffleNas \
    --epoch-start-cs 60 --cs-warm-up --channels-layout OneShot \
    --save-dir params_shufflenas_supernet --logging-file ./logs/shufflenas_supernet.log
