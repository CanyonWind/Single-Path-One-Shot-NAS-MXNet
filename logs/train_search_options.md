Detailed options:
```
train_imagenet.py [-h]   [--data-dir DATA_DIR] [--rec-train REC_TRAIN]
                         [--rec-train-idx REC_TRAIN_IDX] [--rec-val REC_VAL]
                         [--rec-val-idx REC_VAL_IDX] [--use-rec]
                         [--batch-size BATCH_SIZE] [--dtype DTYPE]
                         [--num-gpus NUM_GPUS] [-j NUM_WORKERS]
                         [--num-epochs NUM_EPOCHS] [--lr LR]
                         [--momentum MOMENTUM] [--wd WD] [--lr-mode LR_MODE]
                         [--lr-decay LR_DECAY]
                         [--lr-decay-period LR_DECAY_PERIOD]
                         [--lr-decay-epoch LR_DECAY_EPOCH]
                         [--warmup-lr WARMUP_LR]
                         [--warmup-epochs WARMUP_EPOCHS] [--last-gamma]
                         [--mode MODE] --model MODEL [--input-size INPUT_SIZE]
                         [--crop-ratio CROP_RATIO] [--use-pretrained]
                         [--use-se] [--mixup] [--mixup-alpha MIXUP_ALPHA]
                         [--mixup-off-epoch MIXUP_OFF_EPOCH]
                         [--label-smoothing] [--no-wd] [--teacher TEACHER]
                         [--temperature TEMPERATURE]
                         [--hard-weight HARD_WEIGHT] [--batch-norm]
                         [--save-frequency SAVE_FREQUENCY]
                         [--save-dir SAVE_DIR] [--resume-epoch RESUME_EPOCH]
                         [--resume-params RESUME_PARAMS]
                         [--resume-states RESUME_STATES]
                         [--log-interval LOG_INTERVAL]
                         [--logging-file LOGGING_FILE] [--use-gn]
                         [--use-all-blocks] [--use-all-channels]
                         [--epoch-start-cs EPOCH_START_CS]
                         [--last-conv-after-pooling] [--cs-warm-up]
                         [--channels-layout CHANNELS_LAYOUT]
                         [--ignore-first-two-cs]
                         [--reduced-dataset-scale REDUCED_DATASET_SCALE]
                         [--block-choices BLOCK_CHOICES]
                         [--channel-choices CHANNEL_CHOICES]

Train a model for image classification.

NAS related arguments:
  -h, --help            show this help message and exit
  --warmup-epochs       WARMUP_EPOCHS
                        number of warmup epochs.
  --model               MODEL
                        type of model to use. see vision_model for options.
  --use-se              use SE layers or not in resnext and ShuffleNas.
                        default is false.
  --use-all-blocks      whether to use all the choice blocks.
  --use-all-channels    whether to use all the channels.
  --epoch-start-cs      EPOCH_START_CS
                        Epoch id for starting Channel selection.
  --last-conv-after-pooling
                        Whether to follow MobileNet V3 last conv after pooling
                        style.
  --cs-warm-up          Whether to do warm up for Channel Selection so that
                        gradually selects larger range of channels
  --channels-layout     CHANNELS_LAYOUT
                        The mode of channels layout: ['ShuffleNetV2+',
                        'OneShot']
  --ignore-first-two-cs
                        whether to ignore the first two channel selection
                        scales. This will be stable for noSE supernet
                        training.
  --reduced-dataset-scale REDUCED_DATASET_SCALE
                        How many times the dataset would be reduced, so that
                        in each epoch only num_batches / reduced_dataset_scale
                        batches will be trained.
  --block-choices       BLOCK_CHOICES
                        Block choices
  --channel-choices     CHANNEL_CHOICES
                        Channel choices
```

```
search_supernet.py [-h]   [--rec-train REC_TRAIN]
                          [--rec-train-idx REC_TRAIN_IDX] [--rec-val REC_VAL]
                          [--rec-val-idx REC_VAL_IDX]
                          [--input-size INPUT_SIZE] [--crop-ratio CROP_RATIO]
                          [--num-workers NUM_WORKERS]
                          [--batch-size BATCH_SIZE] [--dtype DTYPE]
                          [--shuffle-train SHUFFLE_TRAIN]
                          [--num-gpus NUM_GPUS] [--use-se]
                          [--last-conv-after-pooling]
                          [--supernet-params SUPERNET_PARAMS]
                          [--search-mode SEARCH_MODE]
                          [--comparison-model COMPARISON_MODEL] [--topk TOPK]
                          [--search-iters SEARCH_ITERS]
                          [--update-bn-images UPDATE_BN_IMAGES]
                          [--search-target SEARCH_TARGET]
                          [--flop-max FLOP_MAX] [--param-max PARAM_MAX]
                          [--score-acc-ratio SCORE_ACC_RATIO]
                          [--fixed-block-choices FIXED_BLOCK_CHOICES]
                          [--fixed-channel-choices FIXED_CHANNEL_CHOICES]
                          [--population-size POPULATION_SIZE]
                          [--retain-length RETAIN_LENGTH]
                          [--random-select RANDOM_SELECT]
                          [--mutate-chance MUTATE_CHANCE]

Search on a pretrained supernet.

optional arguments:
  -h, --help            show this help message and exit
  --shuffle-train       SHUFFLE_TRAIN
                        whether to do shuffle in training data for BN update
  --num-gpus            NUM_GPUS   
                        number of gpus to use
  --use-se              use SE layers or not in resnext and ShuffleNas
  --last-conv-after-pooling
                        whether to follow MobileNet V3 last conv after pooling
                        style
  --supernet-params     SUPERNET_PARAMS
                        supernet parameter directory
  --search-mode         SEARCH_MODE
                        search mode, options: ['random', 'genetic']
  --comparison-model    COMPARISON_MODEL
                        model to compare with when searching, options:
                        ['MobileNetV3_large', 'MobileNetV2_1.4',
                        'SinglePathOneShot', 'ShuffleNetV2+_medium']
  --topk                TOPK           
                        save top k models
  --search-iters        SEARCH_ITERS
                        how many search iterations
  --update-bn-images    UPDATE_BN_IMAGES
                        How many images to update the BN statistics.
  --search-target       SEARCH_TARGET
                        searching target, options: ['acc',
                        'balanced_flop_acc']
  --flop-max            FLOP_MAX
                        The maximum ratio to the comparison model's flop. So
                        that the searched model's FLOP willalways <
                        comparison_model_FLOP * args.flop_max. -1 means
                        unbounded.
  --param-max           PARAM_MAX
                        The maximum ratio to the comparison model's # param.
                        So that the searched model's # paramwill always <
                        comparison_model_#param * args.param_max. -1 means
                        unbounded
  --score-acc-ratio     SCORE_ACC_RATIO
                        Normalized_MicroNet_score/acc_weight for fitness. The
                        evolver will search for the modelwith highest balanced
                        score (-micronet_score * args.score_acc_ratio + acc).
  --fixed-block-choices FIXED_BLOCK_CHOICES
                        Block choices. It should be a str ofblock_ids
                        separated with comma : '0, 0, 3, 1, 1, 1, 0, 0, 2, 0,
                        2, 1, 1, 0, 2, 0, 2, 1, 3, 2'
  --fixed-channel-choices FIXED_CHANNEL_CHOICES
                        Channel choices. It should be a str ofchannel_ids
                        separated by comma: '6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7,
                        5, 4, 6, 7, 4, 4, 5, 4, 3'
  --population-size     POPULATION_SIZE
                        the size of population to keep during searching
  --retain-length       RETAIN_LENGTH
                        how many items to keep after fitness
  --random-select       RANDOM_SELECT
                        probability of a rejected network remaining in the
                        population
  --mutate-chance       MUTATE_CHANCE
                        probability a network will be randomly mutated

```
