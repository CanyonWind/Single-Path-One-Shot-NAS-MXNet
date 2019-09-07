import mxnet as mx
from mxnet import nd
import random


def random_block_choices(num_of_block_choices=4, select_predefined_block=False, ctx=mx.cpu()):
    if select_predefined_block:
        block_choices = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    else:
        block_number = sum([4, 4, 8, 4])
        block_choices = []
        for i in range(block_number):
            block_choices.append(random.randint(0, num_of_block_choices - 1))
    return nd.array(block_choices, ctx)


def main():
    # ctx = [mx.gpu(i) for i in range(4)]
    ctx = mx.gpu(0)
    random_block_choices(ctx=ctx)


if __name__ == '__main__':
    main()
