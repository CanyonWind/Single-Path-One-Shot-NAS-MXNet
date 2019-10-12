# [MXNet Quantization Benchmark](https://github.com/apache/incubator-mxnet/tree/master/example/quantization)

| Model | Source | Dataset | FP32 Accuracy (top-1/top-5)| INT8 Accuracy (top-1/top-5)|
|:---|:---|---|:---:|:---:|
| [ResNet18-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  |70.15%/89.38%|69.92%/89.30%|
| [ResNet50-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 76.34%/93.13%  |  76.06%/92.99% |
| [ResNet101-V1](#3)  | [Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | [Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)  | 77.33%/93.59%  | 77.07%/93.47%  |
|[Squeezenet 1.0](#4)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|56.98%/79.20%|56.79%/79.47%|
|[MobileNet 1.0](#5)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|72.23%/90.64%|72.06%/90.53%|
|[MobileNetV2 1.0](#6)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|70.27%/89.62%|69.82%/89.35%|
|[Inception V3](#7)|[Gluon-CV](https://gluon-cv.mxnet.io/model_zoo/classification.html)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|77.76%/93.83% |78.05%/93.91% |
|[ResNet152-V2](#8)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/resnet/152-layers/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|76.65%/93.07%|76.25%/92.89%|
|[Inception-BN](#9)|[MXNet ModelZoo](http://data.mxnet.io/models/imagenet/inception-bn/)|[Validation Dataset](http://data.mxnet.io/data/val_256_q90.rec)|72.28%/90.63%|72.02%/90.53%|
| [SSD-VGG16](#10) | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | VOC2007/2012  | 0.8366 mAP  | 0.8357 mAP  |
| [SSD-VGG16](#10) | [example/ssd](https://github.com/apache/incubator-mxnet/tree/master/example/ssd)  | COCO2014  | 0.2552 mAP  | 0.253 mAP  |
