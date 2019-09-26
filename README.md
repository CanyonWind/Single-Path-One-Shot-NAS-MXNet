# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS networks  **MXNet (Gluon)** implementation, modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series/tree/master/OneShot). **For training:** It supports both fixed structure model, the supernet model with block & channel selection and SE. **For searching:** It supports both random search and genetic search with BN statistics update and [normalized FLOP + # parameters constraint](https://micronet-challenge.github.io/scoring_and_submission.html)

## Prerequisites
Download the ImageNet dataset, reorgnize the raw data and create MXNet RecordIO files (or just put the validation images in its corresponding class folder) by following [this script](https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset)

## Comparison to the official release 
- Support both fixed-structure model and supernet uniform selection model.
- SE is available for both fixed-structure and supernet models.
- Fixed-structure model can be hybridized, hence (hopefully) also be accelerated.
- Support both random block selection and random channel selection.
- A full functioning FLOP calculator is provided.
- A naive random search with BN statistics update and FLOP & # parameters constraint is provided.
 
 
## Usage
**Training stage**

Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
# For fixed-structure model
sh ./train_fixarch.sh

# For supernet model
sh ./train_supernet.sh
```

**Searching stage**

```shell
# Save a toy model of supernet model param, or put a well-trained supernet model under ./params/ folder and skip this step
python oneshot_nas_network.py

# do random search
python search_supernet.py
```


## Roadmap
- [x] Implement the fixed architecture model from the official pytorch release.
- [x] Implement the random block selection and channel selection.
- [x] Verify conv kernel gradients would be be updated according to ChannelSelector 
- [x] Make the fixed architecture model hybridizable.
- [x] Train a tiny model on Imagenet to verify the feasibility.
- [x] Modify the open source MXNet FLOP calculator to support BN
- [x] Verify that this repo's implementation shares the same # parameters and # FLOPs with the official one.
- [x] Add SE in the model (on/off can be controlled by --use-se)
- [x] Add MobileNetV3 style last conv (on/off can be controlled by --last-conv-after-pooling)
- [ ] **In progress:** Train the official fixed architecture model on Imagenet
- [x] **In progress:** Train the official uniform selection supernet model on Imagenet
    - [x] Add --use-all-blocks, --use-all-channels and --epoch-start-cs options for the supernet training.
    - [x] Add channel selection warm up: after epoch_start_cs, the channel selection range will be gradually increased.
    - [x] Train the supernet with --use-se and --last-conv-after-pooling --cs-warm-up
- [ ] **In progress:** Build the evolution algorithm to search within the pretrained supernet model.
    - [x] Build random search
    - [x] update BN before calculating the validation accuracy for each choice
        - [x] Build and do unit test on the customized BN for updating moving mean & variance during inference
        - [x] Replace nn.batchnorm with the customized BN
    - [x] Evolution algorithm 
    - [ ] Evolution algorithm with flop / # parameters constraint(s)


## Results

**Original implementation**

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  328M (345M with ReLU and BN) |  3.4M |  **-**   |   -   |
|    NASNET-A|  564M |  5.3M |  74.0   |   91.6   |
|    PNASNET|  588M |  5.1M |  74.2   |   91.9   |
|    MnasNet|  317M |  4.2M |  74.0   |  91.8   |
|    DARTS|  574M|  4.7M |  73.3   |   91.3  |
|    FBNet-B|  295M|  4.5M |  74.1   |   -   |

**Last conv channels reducted model**

Beacuse of a mistyping, a small model with 1/10 last conv channels was trained. Provided anyhow.  

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot (customized) |  (embarrassing large)M |  1.93M |  **68.74**   |   -   |
|    MobileNet V3 Small 0.75 | 44M | 2.4M | 65.4 | - |
|    Mnas Small | 65.1M | 1.9M | 64.9 | - |
|    [MobileNet V2 0.5](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet#imagenet--checkpoints) | 97.2M | 1.95M | 65.4 | - |


## Citation
If you use these models in your research, please cite:


    @article{guo2019single,
            title={Single path one-shot neural architecture search with uniform sampling},
            author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
            journal={arXiv preprint arXiv:1904.00420},
            year={2019}
    }
