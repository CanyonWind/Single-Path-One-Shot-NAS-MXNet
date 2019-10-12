# Run Inference

1. Install MXNet **1.5.0** version with MKL backend
```
# if use CPU version
pip3 install mxnet-mkl==1.5.0

# if use GPU version with CUDA 10.1
pip3 install mxnet-cu101mkl==1.5.0
```

2. Download imagenet validation dataset `val.rec` from ... and put it in `MicroNetChallenge/data` folder

3. Launch Inference
```
# Launch Inference
python3 inference.py --param-file=./model/oneshot-s+model-quantized-0000.params --symbol-file=./model/oneshot-s+model-quantized-symbol.json --dataset=./data/val.rec
```

3. If everything works fine, the result should be like:
```
INFO:logger:('accuracy', 0.75721875)
INFO:logger:('top_k_accuracy_5', 0.92878125)
```
