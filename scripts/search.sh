python search_supernet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --batch-size 128 --num-gpus 4 -j 8 \
    --supernet_params ./params/ShuffleNasOneshot-imagenet-supernet.params \
    --dtype float32 --shuffle-train False --use-se \
    --search-mode genetic --comparison-model SinglePathOneShot \
    --topk 3 --search_iters 10 --update_bn_images 20000\
    --population_size 500 --retain_length 100 \
    --random_select 0.1 --mutate_chance 0.1
