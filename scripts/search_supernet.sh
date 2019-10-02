python search_supernet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --batch-size 128 --num-gpus 4 --num-workers 8 \
    --supernet-params ./params/ShuffleNasOneshot-imagenet-supernet.params \
    --dtype float32 --shuffle-train False \
    --search-mode genetic --comparison-model SinglePathOneShot \
    --topk 3 --search-iters 20 --update-bn-images 20000\
    --population-size 50 --retain-length 10 \
    --random-select 0.1 --mutate-chance 0.1
