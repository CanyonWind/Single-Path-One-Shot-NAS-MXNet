# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS searched networks implementation by MXNet (Gluon), modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series).

No fix branch

## Requirements
Download the ImageNet dataset, move validation images to labeled subfolders and(or) create MXNet RecordIO files. To do these, you can use the following script:
https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset

## Usage
Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

Train:
```shell
python train_imagenet.py --model ShuffleNas_fixArch
```

## Results


| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  -M |  -M |  **-**   |   -   |
|    NASNET-A|  564M |  5.3M |  26.0   |   8.4   |
|    PNASNET|  588M |  5.1M |  25.8   |   8.1   |
|    MnasNet|  317M |  4.2M |  26.0   |  8.2   |
|    DARTS|  574M|  4.7M |  26.7   |   8.7  |
|    FBNet-B|  295M|  4.5M |  25.9   |   -   |

## Citation
If you use these models in your research, please cite:


    @article{guo2019single,
            title={Single path one-shot neural architecture search with uniform sampling},
            author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
            journal={arXiv preprint arXiv:1904.00420},
            year={2019}
    }
