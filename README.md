# [Single Path One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS networks  **MXNet (Gluon)** implementation, modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot). **For training:** It supports the fixed structure model, the supernet model with block & channel selection and the ShuffleNetV2+ style SE. **For searching:** It supports both genetic and random search with BN statistics update and the FLOP + # parameters constraint.

**10/09/2019 Update:** A searched model Oneshot+_searched, with the block choices and channel choices searched by this repo's implementation, ShuffleNetV2+ style SE and MobileNetV3 last convolution block design, reaches the **new highest** top-1 & top-5 accuracies with the **new lowest** [Google MicroNet Challenge Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html). Check [here](https://github.com/CanyonWind/oneshot_nas#results) for comparison.

**09/30/2019 Update:** A customized model Oneshot+, with the block choices and channel choices provided from paper, ShuffleNetV2+ style SE and MobileNetV3 last convolution block design, reaches the highest top-1 & top-5 accuracies with the lowest [Google MicroNet Challenge Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html). Check [here](https://github.com/CanyonWind/oneshot_nas#results) for comparison.

## Prerequisites
Download the ImageNet dataset, reorgnize the raw data and create MXNet RecordIO files (or just put the validation images in its corresponding class folder) by following [this script](https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset). 

And set up the environments.
```shell
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
python3 -m venv env

source env/bin/activate
pip install -r requirements.txt
```

## Comparison to the official release 
- Support both fixed-structure model and supernet uniform selection model.
- SE is available for both fixed-structure and supernet models.
- Fixed-structure model can be hybridized, hence (hopefully) also be accelerated.
- Support both random block selection and random channel selection.
- A full functioning FLOP calculator is provided.
- Genetic and random search with BN statistics update and FLOP & # parameters constraint are provided.

## Results
 
![alt text](./images/search_supernet.gif)


| Model                  | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  841.9M  |  15.4M  |  62.90   |   84.49   | 7.09 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |
|    OneShot+_searched |  322M |  3.2M |  **75.48**   |   **92.59**   | **1.9334** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot+ |  297M |  3.7M |  **75.24**   |   **92.58**   | **1.9937** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot |  328M |  3.4M |  74.02   |   91.60   | 2 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch.sh) | [log](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/logs/shufflenas_oneshot.log) |
|    OneShot (official) |  328M |  3.4M |  74.9   |   92.0   | 2 | - | - |
|    FBNet-B|  295M|  4.5M |  74.1   |   -   | 2.19 | - | - |
|    MnasNet|  317M |  4.2M |  74.0   |  91.8   | 2.20 | - | - |
|    MobileNetV3 Large|	 **217M** |	5.4M |	75.2|	- | 2.25 | - | - |
|    DARTS|  574M|  4.7M |  73.3   |   91.3  | 3.13 | - | - |
|    NASNET-A|  564M |  5.3M |  74.0   |   91.6   | 3.28 | - | - |
|    PNASNET|  588M |  5.1M |  74.2   |   91.9   | 3.29 | - | - |
|    MobileNetV2 (1.4) |	585M |	6.9M |	74.7 |	- | 3.81 | - | - |

 
## Usage
**Training stage**

Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
# For the paper's searched fixed-structure model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas_fixArch --mode hybrid \
    --lr 0.5 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 240 --batch-size 256 --num-gpus 4 -j 8 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_fixarch --logging-file shufflenas_fixarch.log

# For supernet model
python train_imagenet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --model ShuffleNas --mode imperative \
    --lr 0.25 --wd 0.00004 --lr-mode cosine --dtype float16\
    --num-epochs 120 --batch-size 128 --num-gpus 4 -j 4 \
    --label-smoothing --no-wd --warmup-epochs 10 --use-rec \
    --save-dir params_shufflenas_supernet --logging-file shufflenas_supernet.log \
    --epoch-start-cs 60 --cs-warm-up
```

**Searching stage**

```shell
# Save a toy model of supernet model param, or put a well-trained supernet model under ./params/ folder and skip this step
python oneshot_nas_network.py

# do genetic search
python search_supernet.py \
    --rec-train ~/imagenet/rec/train.rec --rec-train-idx ~/imagenet/rec/train.idx \
    --rec-val ~/imagenet/rec/val.rec --rec-val-idx ~/imagenet/rec/val.idx \
    --batch-size 128 --num-gpus 4 -j 8 \
    --supernet_params ./params/ShuffleNasOneshot-imagenet-supernet.params \
    --dtype float32 --shuffle-train False \
    --search-mode genetic --comparison-model SinglePathOneShot \
    --topk 3 --search_iters 10 --update_bn_images 20000\
    --population_size 500 --retain_length 100 \
    --random_select 0.1 --mutate_chance 0.1
```

## Roadmap
- [x] Implement the fixed architecture model from the official pytorch release.
- [x] Implement the random block selection and channel selection.
- [x] Verify conv kernel gradients would be be updated according to ChannelSelector 
- [x] Make the fixed architecture model hybridizable.
- [x] Train a tiny model on Imagenet to verify the feasibility.
- [x] Modify the open source MXNet FLOP calculator to support BN
- [x] Verify that this repo's implementation shares the same # parameters and # FLOPs with the official one.
- [x] Add SE and hard swish in the model (on/off can be controlled by --use-se)
- [x] Add MobileNetV3 style last conv (on/off can be controlled by --last-conv-after-pooling)
- [x] Train the official fixed architecture model on Imagenet
- [x] Train the official uniform selection supernet model on Imagenet
    - [x] Add --use-all-blocks, --use-all-channels and --epoch-start-cs options for the supernet training.
    - [x] Add channel selection warm up: after epoch_start_cs, the channel selection range will be gradually increased.
    - [x] Train the supernet with --use-se and --last-conv-after-pooling --cs-warm-up
- [x] Build the evolution algorithm to search within the pretrained supernet model.
    - [x] Build random search
    - [x] update BN before calculating the validation accuracy for each choice
        - [x] Build and do unit test on the customized BN for updating moving mean & variance during inference
        - [x] Replace nn.batchnorm with the customized BN
    - [x] Evolution algorithm 
    - [x] Evolution algorithm with flop and # parameters constraint(s)
- [ ] Train with heuristic constraint --> To limit unuseful subnet training
    - [ ] Do offiline calculation for each pair of (block, channels) and build an efficient heuristic flops estimator.
    - [ ] In training stage, the pair of choices that doen't reach heuristic constraint will be ignored.
- [ ] Two stage searching
    - [ ] Do Block search firstly
    - [ ] Based on the best searched blocks, do channel search
- [ ] Search and train an ShufflNetV2+ style se-subnet.
    - [ ] **In progress:** Search within the pretrained se-supernet
    - [ ] Train the searched se-subnet
- [ ] Estimate each (block, # channel) combination cpu & gpu latency
- [ ] Quantization


## Citation
If you use these models in your research, please cite:


    @article{guo2019single,
            title={Single path one-shot neural architecture search with uniform sampling},
            author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
            journal={arXiv preprint arXiv:1904.00420},
            year={2019}
    }
