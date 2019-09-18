# [One-Shot NAS](https://arxiv.org/abs/1904.00420)
This repository contains single path one-shot NAS networks  **MXNet (Gluon)** implementation, modified from
[the official pytorch implementation](https://github.com/megvii-model/ShuffleNet-Series). It supports both pre-defined fixed structure model and the supernet model with block selection and channel selection.

## Prerequisites
Download the ImageNet dataset, reorgnize the raw data and create MXNet RecordIO files (or just put the validation images in its corresponding class folder) by following [this script](https://gluon-cv.mxnet.io/build/examples_datasets/imagenet.html#prepare-the-imagenet-dataset)

## Comparison to the official release 
- Support both fixed-structure model and supernet uniform selection model.
- Fixed-structure model can be hybridized, hence (hopefully) also be accelerated.
- Support both random block selection and random channel selection.
- A full functioning FLOP calculator is provided.
- Add a last conv channels reducted model with 1.9M parameters and 68.74% top-1 accuracy.


## Roadmap
- [x] Implement the fixed architecture model from the official pytorch release.
- [x] Implement the random block selection and channel selection.
- [x] Verify conv kernel gradients would be be updated according to ChannelSelector 
- [x] Make the fixed architecture model hybridizable.
- [x] Train a tiny model on Imagenet to verify the feasibility.
- [x] Modify the open source MXNet FLOP calculator to support BN
- [x] Verify that this repo's implementation shares the same # parameters and # FLOPs with the official one.
- [ ] **In progress:** Train the official fixed architecture model on Imagenet
- [ ] **TODO:** Train the official uniform selection supernet model on Imagenet
    - [x] Seems like training with random block & channel selection from scratch is hard to converge, add --use-all-blocks and --use-all-channels options for the supernet training.
- [ ] **In progress:** Build the evolution algorithm to search within the pretrained supernet model.
    - [x] Build random search
    - [ ] update BN before calculating the validation accuracy for each choice
        - [x] Build and do unit test on the customized BN for updating moving mean & variance during inference
        - [x] Replace nn.batchnorm with the customized BN
    - [ ] Evolution algorithm 
    - [ ] Evolution algorithm with flop / # parameters constraint(s)


## Usage
Use [the GluonCV official ImageNet training script](https://gluon-cv.mxnet.io/build/examples_classification/dive_deep_imagenet.html#sphx-glr-download-build-examples-classification-dive-deep-imagenet-py)
to do the training. A slightly modified version is included in this repo.

```shell
sh ./train_fixarch.sh
```

For the flop calculator, save the symobilc model first. Then call the calculator. You can choose to ignore ReLU and BN or not with the flag -norelubn
```python
python oneshot_nas_network.py 
python calculate_flops.py -norelubn
```

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
