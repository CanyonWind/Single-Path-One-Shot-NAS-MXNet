# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS searched networks implementation by **MXNet (Gluon)**, modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series). It supports both pre-defined fixed structure model and the supernet model with block selection and channel selection.

## Prerequisites
Download the ImageNet dataset, reorgnize the raw data and create MXNet RecordIO files (or just put the validation images in its corresponding class folder) by following [this script](https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset)

## Comparison to the official release 
- Support both fixed-structure model and supernet uniform selection model.
- Fixed-structure model can be hybridized, hence (hopefully) also be accelerated.
- Support both random block selection and random channel selection.
- Add a customized tiny model with 1.9M parameters and 67.02% top-1 accuracy.
- A full functioning FLOP calculator is provided.

## Roadmap
- [x] Implement the fixed architecture model from the official pytorch release.
- [x] Implement the random block selection and channel selection.
- [x] Make the fixed architecture model hybridizable.
- [x] Train a tiny model on Imagenet to verify the feasibility.
- [x] Modify the open source MXNet FLOP calculator to support BN
- [x] Verify that this repo's implementation shares the same # parameters and # FLOPs with the official one.
- [ ] **In progress:** Train the official fixed architecture model on Imagenet
- [ ] **TODO:** Train the official uniform selection supernet model on Imagenet
- [ ] **TODO:** Build the evolution algorithm to search within the pretrained supernet model.


## Usage
Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
sh ./train_fixarch.sh

# For flop calculator, save the symobilc model first. Then call the calculator. You can choose to ignore relu & bn or not with the flag -norelubn
python oneshot_nas_network.py 
python calculate_flops.py -norelubn
```

## Results

**Original implementation**

| Model                  | FLOPs | #Params   | Top-1 | Top-5 |
| :--------------------- | :---: | :------:  | :---: | :---: |
|    OneShot |  328M (345M with ReLU and BN) |  3.4M |  **-**   |   -   |
|    NASNET-A|  564M |  5.3M |  26.0   |   8.4   |
|    PNASNET|  588M |  5.1M |  25.8   |   8.1   |
|    MnasNet|  317M |  4.2M |  26.0   |  8.2   |
|    DARTS|  574M|  4.7M |  26.7   |   8.7  |
|    FBNet-B|  295M|  4.5M |  25.9   |   -   |

**Customized tiny model**

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
