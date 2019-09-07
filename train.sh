python train_imagenet.py \
    --rec-train ./imagenet/rec/train.rec --rec-train-idx ./imagenet/rec/train.idx \
    --rec-val ./imagenet/rec/val.rec --rec-val-idx ./imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode imperative \
    --lr 0.05 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 150 --batch-size 192 --num-gpus 4 -j 64 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --save-dir params_shufflenas_fixarch --logging-file shufflenas_fixarch.log
