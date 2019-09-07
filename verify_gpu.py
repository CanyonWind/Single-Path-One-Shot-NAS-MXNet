import mxnet as mx
a = mx.nd.ones((2, 3), mx.gpu())
b = a * 2 + 1
b.asnumpy()

print("How many gpus: {}".format(mx.context.num_gpus()))
