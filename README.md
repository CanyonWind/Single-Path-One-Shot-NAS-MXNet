# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS searched networks implementation by MXNet (Gluon), modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series).

Support both pre-defined fixed structure net and random structure supernet with block selection and channel selection.

## Prerequisites
Download the ImageNet dataset, move validation images to labeled subfolders and(or) create MXNet RecordIO files. To do these, you can use the following script:
https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset

## Comparison to the official release 
- Support both fixed-structure model and supernet uniform selection model
- Fixed-structure model can be hybridized and (hopefully) accelerated
- Support both random block selection and random channel selection
- Fuse the original "Shufflenet" and "Shuffle_Xception" blocks into one "ShuffleNetBlock"
- Add another customized super tiny model with 1.9M parameters and 67.02% top-1 accuracy.

## Usage
Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

Train:
```shell
sh ./train_fixArch.sh
```

## Results

**Original implementation**

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  -M |  3.4M |  **-**   |   -   |
|    NASNET-A|  564M |  5.3M |  26.0   |   8.4   |
|    PNASNET|  588M |  5.1M |  25.8   |   8.1   |
|    MnasNet|  317M |  4.2M |  26.0   |  8.2   |
|    DARTS|  574M|  4.7M |  26.7   |   8.7  |
|    FBNet-B|  295M|  4.5M |  25.9   |   -   |

**Customized super tiny model**

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot (customized) |  -M |  1.93M |  **67.02**   |   -   |
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
