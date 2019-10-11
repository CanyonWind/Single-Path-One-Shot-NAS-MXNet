
# Abstract

Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this work, we provided an open-sourced weight sharing Neural Architecture Search (NAS) pipeline, which can be **trained and searched on ImageNet totally within 60 GPU hours** (on 4 V100 GPUS) **in the exploration space of about 32^20 choices**.

This implementation searched a new state-of-the-art subnet model outperforming Single Path One Shot, FBNet,MnasNet, DARTS, NASNET, PNASNET by a good margin in all factors of FLOPS, # of parameters and Top-1 accuracy. Also for considering the MicroNet Challenge Σ score, without any quantization, it outperforms MobileNet V2, V3, ShuffleNet V1, V2, V2+ too.

# Our approach

Our approach is **mainly based on the Single Path One Shot NAS in combination of Squeeze and Excitation (SE), ShuffleNet V2+ and MobileNet V3**. Like the original paper, we searched for the choice blocks and block channels with multiple FLOPs and # of parameters constraints. In this section, we will elaborate the modifications from the original paper.

## Supernet Structure Design

For each ShuffleNasBlock (SNasB), four choice blocks were explored, ShuffleNetBlock-3x3 (SNB-3), SNB-5, SNB-7 and ShuffleXceptionBlock-3x3 (SXB-3). Within each block, eight channel choices are avialable: [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] x (BlockOutputChannel / 2). So each SNasB block explores 32 possible choices and there are 20 blocks in this implementation, counting for totaly 32^20 design choices.

We also applied the SE, ShuffleNet V2+ SE layout and the MobileNet V3 last convolution block design in the supernet. Finally, the supernet contains 15.4 Million trainable parameters and the subnet FLOPs ranges from 168M to 841M.

## Supernet Training

Unlike what the original paper did, in the training stage, we didn't apply uniform distribution from the begining. We train the supernet totaly 120 epochs. In the first 60 epochs doing Block selection only and, for the upcoming 60 epochs, we used Channel Selection warm-up which gradually allows the supernet to be trained with larger range of channel choices.

``` python
   # Supernet sampling schedule: during channel selection warm-up, apply more epochs for [0.2, 0.4, 0.6, 0.8, 1.0] channel choices
   1 - 60 epochs:          Only block selection (BS)
   61 epoch:               [1.8, 2.0] + BS
   62 epoch:               [1.6, 1.8, 2.0] + BS
   63 epoch:               [1.4, 1.6, 1.8, 2.0] + BS
   64 epoch:               [1.2, 1.4, 1.6, 1.8, 2.0] + BS
   65 - 66 epochs:         [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS
   67 - 69 epochs:         [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS
   70 - 73 epochs:         [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS 
   74 - 78 epochs:         [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS 
   79 - 120 epochs:        [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS
```

The reason why we did this in the supernet training is that, during our experiments, we found:
> 1. For supernet without SE, doing block selection alone in the first n epochs and using Channel Selection warm-up are necessary to make the model converge. Otherwise it wouldn't converge at all.
> 2. For supernet with SE, Channel Selection with the full choices (0.2 ~ 2.0) can be used at the first epoch and it converges.
>    - However, doing Channel Selection from the first epoch seems like harming the accuracy. Compared to the same se-supernet with first n epoch block selection alone and --cs-warm-up, the channel selection from-scratch se-supernet only reached ~33% training accuracy and the warmed up se-supernet reaches ~44%, both at ~70th epoch.
>    - Another thing is that validation accuracy in the channel selection from-scratch se-supernet is always under 1%, while the warmed up se-supernet's validation accuracy looks reasonably increasing from 0.1% to 63%.

## Subnet Searching

Different from the paper, we jointly searched for the Block choices and Channel Choices in the supernet at the same time. It means that, for each instance in the population of our genetic algorithm, it contains 20 Block choices gene and 20 Channel choices gene. We were aiming to find a combination of these two which optimizing for the both and being Complementary.

The details of our genetic search are also not identical to the original paper, here is how we implemented:

> Initial population P0 = 50
> 
> random_select_ratio = 0.1
> 
> mutate_ratio = 0.1

For each sampled subnet structure passing the constraint, like most weight sharing NAS approaches did, we updated the BN statistics firstly with 20,000 fixed (or random, this doesn't influence much) training set images and then evalute this subnet ImageNet validation accuracy as the indicator for its performance.


## Subnet Training

For the final searched model, we build and train it from scratch. No previous supernet weights are reused in subnet.

As for the hyperparameters. We modified the GluonCV official ImageNet training script to support both supernet training and subnet training. We trained both models with initial learning rate 1.3, weight decay 0.00003, cosine learning rate scheduler, 4 GPUs each with batch size 256, label smoothing and no weight decay for BN beta gamma. Supernet was trained 120 epochs and subnet was trained 360 epochs. 


# Results

## Supernet Training
![alt text](../images/Supernet.png)

As we claimed in [here], we did Block selection only in the first 60 epochs and strating from 61 epoch, we gradually allow larger range of channel choices to be sampled and used. To explain the accuracy drop between [60, 70] epochs, we need to understand there is a difficulty in evaluating the supernet performance. During training, we sampled a random combination of Block Choices and Channel Choices per batch and it results in that we only trained (120 epochs * 1,280,000 images per epoch) / 1024 batch size -> 150,000 possible subnet structures over 32^20. 

If we also sample random Blocks and Channels for validation set per batch, each validation batch subnet structure is highly unlikely being seen during training. So we tried to fix the Block & Channel choices (do random block selection but all channel choices are set to be maximum) for the validation set between epoch [60, 70]. It doesn't work well so that we change back to do random samplingfor both. This random sampling for validation set surprisingly worked well. 

## Supernet Searching
![alt text](../images/search_supernet.gif)

We tried both random search, random selecting 250 qualified instance to evaluate their performance, and genetic search. The genetic method easily found a better subnet structure over the random selection.

## Searched Model Profiling

|op_name                                   | inp_size  |  kernel_size   | Cin |  Cout |  params(MBytes) |   mults(M)  |   adds(M)  |    MFLOPS| 
|:-----                                   | :-----:  | :-----:   | :-----:|  :-----: |  :-----: |   :-----:  |   :-----: |   :-----:| 
|first_conv_fwd                          |        224|             3|             3 |           16|           0.000 |     5.419|      5.218 |    10.637|
|hardswish0__plusscalar0                 |        112|            -1|            16 |           16|           0.000 |     0.000|      0.201 |     0.201|
|hardswish0__clip                        |        112|            -1|            16 |           16|           0.000 |     0.201|      0.000 |     0.201|
|hardswish0__divscalar0                  |        112|            -1|            16 |           16|           0.000 |     0.201|      0.000 |     0.201|
|hardswish0__mul0                        |        112|            -1|            16 |           16|           0.000 |     0.201|      0.000 |     0.201|
|SNB0_conv3_fwd                          |        112|             3|            16 |            1|           0.000 |     0.452|      0.401 |     0.853|
|SNB0_conv4_fwd                          |         56|             1|            16 |           16|           0.000 |     0.803|      0.753 |     1.555|
|SNB0_relu2_fwd                          |         56|            -1|            16 |           16|           0.000 |     0.050|      0.000 |     0.050|
|SNB0_conv0_fwd                          |        112|             1|            16 |           57|           0.001 |    11.440|     10.725 |    22.165|
|SNB0_relu0_fwd                          |        112|            -1|            57 |           57|           0.000 |     0.715|      0.000 |     0.715|
|SNB0_conv1_fwd                          |        112|             3|            57 |            1|           0.001 |     1.609|      1.430 |     3.039|
|SNB0_conv2_fwd                          |         56|             1|            57 |           48|           0.003 |     8.580|      8.430 |    17.010|
|SNB0_relu1_fwd                          |         56|            -1|            48 |           48|           0.000 |     0.151|      0.000 |     0.151|
|SNB1_shufflechannels0                   |         56|             1|            64 |           64|           0.000 |     0.201|      0.000 |     0.201|
|SNB1_conv0_fwd                          |         56|             1|            32 |           51|           0.002 |     5.118|      4.958 |    10.076|
|SNB1_relu0_fwd                          |         56|            -1|            51 |           51|           0.000 |     0.160|      0.000 |     0.160|
|SNB1_conv1_fwd                          |         56|             3|            51 |            1|           0.000 |     1.439|      1.279 |     2.719|
|SNB1_conv2_fwd                          |         56|             1|            51 |           32|           0.002 |     5.118|      5.018 |    10.136|
|SNB1_relu1_fwd                          |         56|            -1|            32 |           32|           0.000 |     0.100|      0.000 |     0.100|
|SNB2_shufflechannels0                   |         56|             1|            64 |           64|           0.000 |     0.201|      0.000 |     0.201|
|SNB2_conv0_fwd                          |         56|             1|            32 |           44|           0.001 |     4.415|      4.278 |     8.693|
|SNB2_relu0_fwd                          |         56|            -1|            44 |           44|           0.000 |     0.138|      0.000 |     0.138|
|SNB2_conv1_fwd                          |         56|             3|            44 |            1|           0.000 |     1.242|      1.104 |     2.346|
|SNB2_conv2_fwd                          |         56|             1|            44 |           32|           0.001 |     4.415|      4.315 |     8.731|
|SNB2_relu1_fwd                          |         56|            -1|            32 |           32|           0.000 |     0.100|      0.000 |     0.100|
|SNB3_shufflechannels0                   |         56|             1|            64 |           64|           0.000 |     0.201|      0.000 |     0.201|
|SNB3_conv0_fwd                          |         56|             1|            32 |           57|           0.002 |     5.720|      5.541 |    11.261|
|SNB3_relu0_fwd                          |         56|            -1|            57 |           57|           0.000 |     0.179|      0.000 |     0.179|
|SNB3_conv1_fwd                          |         56|             5|            57 |            1|           0.001 |     4.469|      4.290 |     8.759|
|SNB3_conv2_fwd                          |         56|             1|            57 |           32|           0.002 |     5.720|      5.620 |    11.340|
|SNB3_relu1_fwd                          |         56|            -1|            32 |           32|           0.000 |     0.100|      0.000 |     0.100|
|SNB4_conv3_fwd                          |         56|             3|            64 |            1|           0.001 |     0.452|      0.401 |     0.853|
|SNB4_conv4_fwd                          |         28|             1|            64 |           64|           0.004 |     3.211|      3.161 |     6.372|
|SNB4_hardswish2__plusscalar0            |         28|            -1|            64 |           64|           0.000 |     0.000|      0.050 |     0.050|
|SNB4_hardswish2__clip                   |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB4_hardswish2__divscalar0             |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB4_hardswish2__mul0                   |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB4_conv0_fwd                          |         56|             1|            64 |           96|           0.006 |    19.268|     18.967 |    38.234|
|SNB4_hardswish0__plusscalar0            |         56|            -1|            96 |           96|           0.000 |     0.000|      0.301 |     0.301|
|SNB4_hardswish0__clip                   |         56|            -1|            96 |           96|           0.000 |     0.301|      0.000 |     0.301|
|SNB4_hardswish0__divscalar0             |         56|            -1|            96 |           96|           0.000 |     0.301|      0.000 |     0.301|
|SNB4_hardswish0__mul0                   |         56|            -1|            96 |           96|           0.000 |     0.301|      0.000 |     0.301|
|SNB4_conv1_fwd                          |         56|             3|            96 |            1|           0.001 |     0.677|      0.602 |     1.279|
|SNB4_conv2_fwd                          |         28|             1|            96 |           96|           0.009 |     7.225|      7.150 |    14.375|
|SNB4_hardswish1__plusscalar0            |         28|            -1|            96 |           96|           0.000 |     0.000|      0.075 |     0.075|
|SNB4_hardswish1__clip                   |         28|            -1|            96 |           96|           0.000 |     0.075|      0.000 |     0.075|
|SNB4_hardswish1__divscalar0             |         28|            -1|            96 |           96|           0.000 |     0.075|      0.000 |     0.075|
|SNB4_hardswish1__mul0                   |         28|            -1|            96 |           96|           0.000 |     0.075|      0.000 |     0.075|
|SNB5_shufflechannels0                   |         28|             1|           160 |          160|           0.002 |     0.125|     -0.000 |     0.125|
|SNB5_conv0_fwd                          |         28|             1|            80 |          128|           0.010 |     8.028|      7.928 |    15.956|
|SNB5_hardswish0__plusscalar0            |         28|            -1|           128 |          128|           0.000 |     0.000|      0.100 |     0.100|
|SNB5_hardswish0__clip                   |         28|            -1|           128 |          128|           0.000 |     0.100|      0.000 |     0.100|
|SNB5_hardswish0__divscalar0             |         28|            -1|           128 |          128|           0.000 |     0.100|      0.000 |     0.100|
|SNB5_hardswish0__mul0                   |         28|            -1|           128 |          128|           0.000 |     0.100|      0.000 |     0.100|
|SNB5_conv1_fwd                          |         28|             3|           128 |            1|           0.001 |     0.903|      0.803 |     1.706|
|SNB5_conv2_fwd                          |         28|             1|           128 |           80|           0.010 |     8.028|      7.965 |    15.994|
|SNB5_hardswish1__plusscalar0            |         28|            -1|            80 |           80|           0.000 |     0.000|      0.063 |     0.063|
|SNB5_hardswish1__clip                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB5_hardswish1__divscalar0             |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB5_hardswish1__mul0                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB6_shufflechannels0                   |         28|             1|           160 |          160|           0.002 |     0.125|     -0.000 |     0.125|
|SNB6_conv0_fwd                          |         28|             1|            80 |           64|           0.005 |     4.014|      3.964 |     7.978|
|SNB6_hardswish0__plusscalar0            |         28|            -1|            64 |           64|           0.000 |     0.000|      0.050 |     0.050|
|SNB6_hardswish0__clip                   |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB6_hardswish0__divscalar0             |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB6_hardswish0__mul0                   |         28|            -1|            64 |           64|           0.000 |     0.050|      0.000 |     0.050|
|SNB6_conv1_fwd                          |         28|             5|            64 |            1|           0.002 |     1.254|      1.204 |     2.459|
|SNB6_conv2_fwd                          |         28|             1|            64 |           80|           0.005 |     4.014|      3.951 |     7.965|
|SNB6_hardswish1__plusscalar0            |         28|            -1|            80 |           80|           0.000 |     0.000|      0.063 |     0.063|
|SNB6_hardswish1__clip                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB6_hardswish1__divscalar0             |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB6_hardswish1__mul0                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_shufflechannels0                   |         28|             1|           160 |          160|           0.002 |     0.125|     -0.000 |     0.125|
|SNB7_conv0_fwd                          |         28|             1|            80 |           80|           0.006 |     5.018|      4.955 |     9.972|
|SNB7_hardswish0__plusscalar0            |         28|            -1|            80 |           80|           0.000 |     0.000|      0.063 |     0.063|
|SNB7_hardswish0__clip                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_hardswish0__divscalar0             |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_hardswish0__mul0                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_conv1_fwd                          |         28|             3|            80 |            1|           0.001 |     0.564|      0.502 |     1.066|
|SNB7_conv2_fwd                          |         28|             1|            80 |           80|           0.006 |     5.018|      4.955 |     9.972|
|SNB7_hardswish1__plusscalar0            |         28|            -1|            80 |           80|           0.000 |     0.000|      0.063 |     0.063|
|SNB7_hardswish1__clip                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_hardswish1__divscalar0             |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB7_hardswish1__mul0                   |         28|            -1|            80 |           80|           0.000 |     0.063|      0.000 |     0.063|
|SNB8_conv6_fwd                          |         28|             3|           160 |            1|           0.001 |     0.282|      0.251 |     0.533|
|SNB8_conv7_fwd                          |         14|             1|           160 |          160|           0.026 |     5.018|      4.986 |    10.004|
|SNB8_hardswish3__plusscalar0            |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB8_hardswish3__clip                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_hardswish3__divscalar0             |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_hardswish3__mul0                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_conv0_fwd                          |         28|             3|           160 |            1|           0.001 |     0.282|      0.251 |     0.533|
|SNB8_conv1_fwd                          |         14|             1|           160 |           96|           0.015 |     3.011|      2.992 |     6.002|
|SNB8_hardswish0__plusscalar0            |         14|            -1|            96 |           96|           0.000 |     0.000|      0.019 |     0.019|
|SNB8_hardswish0__clip                   |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_hardswish0__divscalar0             |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_hardswish0__mul0                   |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_conv2_fwd                          |         14|             3|            96 |            1|           0.001 |     0.169|      0.151 |     0.320|
|SNB8_conv3_fwd                          |         14|             1|            96 |           96|           0.009 |     1.806|      1.788 |     3.594|
|SNB8_hardswish1__plusscalar0            |         14|            -1|            96 |           96|           0.000 |     0.000|      0.019 |     0.019|
|SNB8_hardswish1__clip                   |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_hardswish1__divscalar0             |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_hardswish1__mul0                   |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB8_conv4_fwd                          |         14|             3|            96 |            1|           0.001 |     0.169|      0.151 |     0.320|
|SNB8_conv5_fwd                          |         14|             1|            96 |          160|           0.015 |     3.011|      2.979 |     5.990|
|SNB8_hardswish2__plusscalar0            |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB8_hardswish2__clip                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_hardswish2__divscalar0             |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_hardswish2__mul0                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB8_se0_pool0_fwd                      |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB8_se0_conv_squeeze_fwd               |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB8_se0_relu0_fwd                      |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB8_se0_conv_excitation_fwd            |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB8_se0_hardsigmoid0__plusscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB8_se0_hardsigmoid0__clip             |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB8_se0_hardsigmoid0__divscalar0       |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB9_shufflechannels0                   |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB9_conv0_fwd                          |         14|             1|           160 |          160|           0.026 |     5.018|      4.986 |    10.004|
|SNB9_hardswish0__plusscalar0            |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB9_hardswish0__clip                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_hardswish0__divscalar0             |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_hardswish0__mul0                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_conv1_fwd                          |         14|             7|           160 |            1|           0.008 |     1.537|      1.505 |     3.042|
|SNB9_conv2_fwd                          |         14|             1|           160 |          160|           0.026 |     5.018|      4.986 |    10.004|
|SNB9_hardswish1__plusscalar0            |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB9_hardswish1__clip                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_hardswish1__divscalar0             |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_hardswish1__mul0                   |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB9_se0_pool0_fwd                      |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB9_se0_conv_squeeze_fwd               |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB9_se0_relu0_fwd                      |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB9_se0_conv_excitation_fwd            |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB9_se0_hardsigmoid0__plusscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB9_se0_hardsigmoid0__clip             |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB9_se0_hardsigmoid0__divscalar0       |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB10_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB10_conv0_fwd                         |         14|             1|           160 |           96|           0.015 |     3.011|      2.992 |     6.002|
|SNB10_hardswish0__plusscalar0           |         14|            -1|            96 |           96|           0.000 |     0.000|      0.019 |     0.019|
|SNB10_hardswish0__clip                  |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB10_hardswish0__divscalar0            |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB10_hardswish0__mul0                  |         14|            -1|            96 |           96|           0.000 |     0.019|      0.000 |     0.019|
|SNB10_conv1_fwd                         |         14|             3|            96 |            1|           0.001 |     0.169|      0.151 |     0.320|
|SNB10_conv2_fwd                         |         14|             1|            96 |          160|           0.015 |     3.011|      2.979 |     5.990|
|SNB10_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB10_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB10_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB10_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB10_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB10_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB10_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB10_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB10_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB10_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB10_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB11_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB11_conv0_fwd                         |         14|             1|           160 |          128|           0.020 |     4.014|      3.989 |     8.003|
|SNB11_hardswish0__plusscalar0           |         14|            -1|           128 |          128|           0.000 |     0.000|      0.025 |     0.025|
|SNB11_hardswish0__clip                  |         14|            -1|           128 |          128|           0.000 |     0.025|      0.000 |     0.025|
|SNB11_hardswish0__divscalar0            |         14|            -1|           128 |          128|           0.000 |     0.025|      0.000 |     0.025|
|SNB11_hardswish0__mul0                  |         14|            -1|           128 |          128|           0.000 |     0.025|      0.000 |     0.025|
|SNB11_conv1_fwd                         |         14|             5|           128 |            1|           0.003 |     0.627|      0.602 |     1.229|
|SNB11_conv2_fwd                         |         14|             1|           128 |          160|           0.020 |     4.014|      3.983 |     7.997|
|SNB11_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB11_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB11_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB11_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB11_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB11_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB11_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB11_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB11_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB11_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB11_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB12_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB12_conv0_fwd                         |         14|             1|           160 |          160|           0.026 |     5.018|      4.986 |    10.004|
|SNB12_hardswish0__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB12_hardswish0__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_hardswish0__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_hardswish0__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_conv1_fwd                         |         14|             7|           160 |            1|           0.008 |     1.537|      1.505 |     3.042|
|SNB12_conv2_fwd                         |         14|             1|           160 |          160|           0.026 |     5.018|      4.986 |    10.004|
|SNB12_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB12_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB12_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB12_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB12_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB12_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB12_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB12_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB12_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB13_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB13_conv0_fwd                         |         14|             1|           160 |          192|           0.031 |     6.021|      5.983 |    12.005|
|SNB13_hardswish0__plusscalar0           |         14|            -1|           192 |          192|           0.000 |     0.000|      0.038 |     0.038|
|SNB13_hardswish0__clip                  |         14|            -1|           192 |          192|           0.000 |     0.038|      0.000 |     0.038|
|SNB13_hardswish0__divscalar0            |         14|            -1|           192 |          192|           0.000 |     0.038|      0.000 |     0.038|
|SNB13_hardswish0__mul0                  |         14|            -1|           192 |          192|           0.000 |     0.038|      0.000 |     0.038|
|SNB13_conv1_fwd                         |         14|             7|           192 |            1|           0.009 |     1.844|      1.806 |     3.650|
|SNB13_conv2_fwd                         |         14|             1|           192 |          160|           0.031 |     6.021|      5.990 |    12.011|
|SNB13_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB13_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB13_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB13_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB13_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB13_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB13_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB13_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB13_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB13_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB13_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB14_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB14_conv0_fwd                         |         14|             1|           160 |          224|           0.036 |     7.025|      6.981 |    14.005|
|SNB14_hardswish0__plusscalar0           |         14|            -1|           224 |          224|           0.000 |     0.000|      0.044 |     0.044|
|SNB14_hardswish0__clip                  |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB14_hardswish0__divscalar0            |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB14_hardswish0__mul0                  |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB14_conv1_fwd                         |         14|             5|           224 |            1|           0.006 |     1.098|      1.054 |     2.151|
|SNB14_conv2_fwd                         |         14|             1|           224 |          160|           0.036 |     7.025|      6.993 |    14.018|
|SNB14_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB14_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB14_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB14_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB14_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB14_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB14_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB14_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB14_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB14_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB14_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB15_shufflechannels0                  |         14|             1|           320 |          320|           0.007 |     0.063|      0.000 |     0.063|
|SNB15_conv0_fwd                         |         14|             1|           160 |          224|           0.036 |     7.025|      6.981 |    14.005|
|SNB15_hardswish0__plusscalar0           |         14|            -1|           224 |          224|           0.000 |     0.000|      0.044 |     0.044|
|SNB15_hardswish0__clip                  |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB15_hardswish0__divscalar0            |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB15_hardswish0__mul0                  |         14|            -1|           224 |          224|           0.000 |     0.044|      0.000 |     0.044|
|SNB15_conv1_fwd                         |         14|             7|           224 |            1|           0.011 |     2.151|      2.107 |     4.259|
|SNB15_conv2_fwd                         |         14|             1|           224 |          160|           0.036 |     7.025|      6.993 |    14.018|
|SNB15_hardswish1__plusscalar0           |         14|            -1|           160 |          160|           0.000 |     0.000|      0.031 |     0.031|
|SNB15_hardswish1__clip                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB15_hardswish1__divscalar0            |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB15_hardswish1__mul0                  |         14|            -1|           160 |          160|           0.000 |     0.031|      0.000 |     0.031|
|SNB15_se0_pool0_fwd                     |         14|            -1|           160 |            1|           0.000 |     0.000|      0.031 |     0.031|
|SNB15_se0_conv_squeeze_fwd              |          1|             1|           160 |           40|           0.006 |     0.006|      0.006 |     0.013|
|SNB15_se0_relu0_fwd                     |          1|            -1|            40 |           40|           0.000 |     0.000|      0.000 |     0.000|
|SNB15_se0_conv_excitation_fwd           |          1|             1|            40 |          160|           0.007 |     0.006|      0.006 |     0.013|
|SNB15_se0_hardsigmoid0__plusscalar0     |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB15_se0_hardsigmoid0__clip            |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB15_se0_hardsigmoid0__divscalar0      |          1|            -1|           160 |          160|           0.000 |     0.000|      0.000 |     0.000|
|SNB16_conv3_fwd                         |         14|             3|           320 |            1|           0.003 |     0.141|      0.125 |     0.267|
|SNB16_conv4_fwd                         |          7|             1|           320 |          320|           0.102 |     5.018|      5.002 |    10.020|
|SNB16_hardswish2__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB16_hardswish2__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_hardswish2__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_hardswish2__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_conv0_fwd                         |         14|             1|           320 |          256|           0.082 |    16.056|     16.006 |    32.062|
|SNB16_hardswish0__plusscalar0           |         14|            -1|           256 |          256|           0.000 |     0.000|      0.050 |     0.050|
|SNB16_hardswish0__clip                  |         14|            -1|           256 |          256|           0.000 |     0.050|      0.000 |     0.050|
|SNB16_hardswish0__divscalar0            |         14|            -1|           256 |          256|           0.000 |     0.050|      0.000 |     0.050|
|SNB16_hardswish0__mul0                  |         14|            -1|           256 |          256|           0.000 |     0.050|      0.000 |     0.050|
|SNB16_conv1_fwd                         |         14|             3|           256 |            1|           0.002 |     0.113|      0.100 |     0.213|
|SNB16_conv2_fwd                         |          7|             1|           256 |          320|           0.082 |     4.014|      3.998 |     8.012|
|SNB16_hardswish1__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB16_hardswish1__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_hardswish1__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_hardswish1__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB16_se0_pool0_fwd                     |          7|            -1|           320 |            1|           0.000 |     0.000|      0.015 |     0.016|
|SNB16_se0_conv_squeeze_fwd              |          1|             1|           320 |           80|           0.026 |     0.026|      0.026 |     0.051|
|SNB16_se0_relu0_fwd                     |          1|            -1|            80 |           80|           0.000 |     0.000|      0.000 |     0.000|
|SNB16_se0_conv_excitation_fwd           |          1|             1|            80 |          320|           0.026 |     0.026|      0.026 |     0.051|
|SNB16_se0_hardsigmoid0__plusscalar0     |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB16_se0_hardsigmoid0__clip            |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB16_se0_hardsigmoid0__divscalar0      |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB17_shufflechannels0                  |          7|             1|           640 |          640|           0.026 |     0.031|      0.000 |     0.031|
|SNB17_conv0_fwd                         |          7|             1|           320 |          256|           0.082 |     4.014|      4.002 |     8.016|
|SNB17_hardswish0__plusscalar0           |          7|            -1|           256 |          256|           0.000 |     0.000|      0.013 |     0.013|
|SNB17_hardswish0__clip                  |          7|            -1|           256 |          256|           0.000 |     0.013|      0.000 |     0.013|
|SNB17_hardswish0__divscalar0            |          7|            -1|           256 |          256|           0.000 |     0.013|      0.000 |     0.013|
|SNB17_hardswish0__mul0                  |          7|            -1|           256 |          256|           0.000 |     0.013|      0.000 |     0.013|
|SNB17_conv1_fwd                         |          7|             3|           256 |            1|           0.002 |     0.113|      0.100 |     0.213|
|SNB17_conv2_fwd                         |          7|             1|           256 |          320|           0.082 |     4.014|      3.998 |     8.012|
|SNB17_hardswish1__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB17_hardswish1__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB17_hardswish1__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB17_hardswish1__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB17_se0_pool0_fwd                     |          7|            -1|           320 |            1|           0.000 |     0.000|      0.015 |     0.016|
|SNB17_se0_conv_squeeze_fwd              |          1|             1|           320 |           80|           0.026 |     0.026|      0.026 |     0.051|
|SNB17_se0_relu0_fwd                     |          1|            -1|            80 |           80|           0.000 |     0.000|      0.000 |     0.000|
|SNB17_se0_conv_excitation_fwd           |          1|             1|            80 |          320|           0.026 |     0.026|      0.026 |     0.051|
|SNB17_se0_hardsigmoid0__plusscalar0     |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB17_se0_hardsigmoid0__clip            |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB17_se0_hardsigmoid0__divscalar0      |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB18_shufflechannels0                  |          7|             1|           640 |          640|           0.026 |     0.031|      0.000 |     0.031|
|SNB18_conv0_fwd                         |          7|             1|           320 |          320|           0.102 |     5.018|      5.002 |    10.020|
|SNB18_hardswish0__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB18_hardswish0__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_hardswish0__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_hardswish0__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_conv1_fwd                         |          7|             7|           320 |            1|           0.016 |     0.768|      0.753 |     1.521|
|SNB18_conv2_fwd                         |          7|             1|           320 |          320|           0.102 |     5.018|      5.002 |    10.020|
|SNB18_hardswish1__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB18_hardswish1__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_hardswish1__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_hardswish1__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB18_se0_pool0_fwd                     |          7|            -1|           320 |            1|           0.000 |     0.000|      0.015 |     0.016|
|SNB18_se0_conv_squeeze_fwd              |          1|             1|           320 |           80|           0.026 |     0.026|      0.026 |     0.051|
|SNB18_se0_relu0_fwd                     |          1|            -1|            80 |           80|           0.000 |     0.000|      0.000 |     0.000|
|SNB18_se0_conv_excitation_fwd           |          1|             1|            80 |          320|           0.026 |     0.026|      0.026 |     0.051|
|SNB18_se0_hardsigmoid0__plusscalar0     |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB18_se0_hardsigmoid0__clip            |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB18_se0_hardsigmoid0__divscalar0      |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB19_shufflechannels0                  |          7|             1|           640 |          640|           0.026 |     0.031|      0.000 |     0.031|
|SNB19_conv0_fwd                         |          7|             1|           320 |          448|           0.143 |     7.025|      7.003 |    14.027|
|SNB19_hardswish0__plusscalar0           |          7|            -1|           448 |          448|           0.000 |     0.000|      0.022 |     0.022|
|SNB19_hardswish0__clip                  |          7|            -1|           448 |          448|           0.000 |     0.022|      0.000 |     0.022|
|SNB19_hardswish0__divscalar0            |          7|            -1|           448 |          448|           0.000 |     0.022|      0.000 |     0.022|
|SNB19_hardswish0__mul0                  |          7|            -1|           448 |          448|           0.000 |     0.022|      0.000 |     0.022|
|SNB19_conv1_fwd                         |          7|             3|           448 |            1|           0.004 |     0.198|      0.176 |     0.373|
|SNB19_conv2_fwd                         |          7|             1|           448 |          320|           0.143 |     7.025|      7.009 |    14.034|
|SNB19_hardswish1__plusscalar0           |          7|            -1|           320 |          320|           0.000 |     0.000|      0.016 |     0.016|
|SNB19_hardswish1__clip                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB19_hardswish1__divscalar0            |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB19_hardswish1__mul0                  |          7|            -1|           320 |          320|           0.000 |     0.016|      0.000 |     0.016|
|SNB19_se0_pool0_fwd                     |          7|            -1|           320 |            1|           0.000 |     0.000|      0.015 |     0.016|
|SNB19_se0_conv_squeeze_fwd              |          1|             1|           320 |           80|           0.026 |     0.026|      0.026 |     0.051|
|SNB19_se0_relu0_fwd                     |          1|            -1|            80 |           80|           0.000 |     0.000|      0.000 |     0.000|
|SNB19_se0_conv_excitation_fwd           |          1|             1|            80 |          320|           0.026 |     0.026|      0.026 |     0.051|
|SNB19_se0_hardsigmoid0__plusscalar0     |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB19_se0_hardsigmoid0__clip            |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|SNB19_se0_hardsigmoid0__divscalar0      |          1|            -1|           320 |          320|           0.000 |     0.000|      0.000 |     0.000|
|pool0_fwd                               |          7|            -1|           640 |            1|           0.000 |     0.001|      0.031 |     0.031|
|conv_fc_fwd                             |          1|             1|           640 |         1024|           0.656 |     0.655|      0.655 |     1.311|
|hardswish1__plusscalar0                 |          1|            -1|          1024 |         1024|           0.000 |     0.000|      0.001 |     0.001|
|hardswish1__clip                        |          1|            -1|          1024 |         1024|           0.000 |     0.001|      0.000 |     0.001|
|hardswish1__divscalar0                  |          1|            -1|          1024 |         1024|           0.000 |     0.001|      0.000 |     0.001|
|hardswish1__mul0                        |          1|            -1|          1024 |         1024|           0.000 |     0.001|      0.000 |     0.001|
|output                                  |          1|             1|          1024 |         1000|           1.025 |     1.024|      1.024 |     2.048|
|total                                   |           |              |               |             |           3.652 |     299.6|      288.3 |     587.9|

## Searched Model Performance

| Model   | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  1684M  |  15.4M  |  62.9   |   84.5   | 3.67 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |
|    **OneShot-S+ int8** |  148M |  0.95M |  **75.?**   |   **92.?**   | **0.26** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+ float16** |  438M |  1.85M |  **75.5**   |   **92.7**   | **0.64** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+** |  588M |  3.65M |  **75.5**   |   **92.7**   | **1.03** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot (our) |  656M |  3.5M |  74.0   |   91.6   | 1.05 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch.sh) | [log](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/logs/shufflenas_oneshot.log) |
|    OneShot (paper) |  656M |  3.5M |  74.9   |   92.0   | 1.05 | - | - |
|    MnasNet|  634M |  4.2M |  74.0   |  91.8   | 1.15 | - | - |
|    MobileNetV3 Large|	 **434M** |	5.4M |	75.2|	- | 1.15 | - | - |
|    FBNet-B|  590M|  4.5M |  74.1   |   -   | 1.16 | - | - |
|    DARTS|  1148M|  4.7M |  73.3   |   91.3  | 1.66 | - | - |
|    NASNET-A|  1128M |  5.3M |  74.0   |   91.6   | 1.73 | - | - |
|    PNASNET|  1176M |  5.1M |  74.2   |   91.9   | 1.74 | - | - |
|    MobileNetV2 (1.4) |	1170M |	6.9M |	74.7 |	- | 2.00 | - | - |
 


# Summary
In this work, we provided an state-of-the-art open-sourced weight sharing Neural Architecture Search (NAS) pipeline, which can be trained and searched on ImageNet totally within 60 GPU hours (on 4 V100 GPUS) and the exporation space is about 32^20. The model searched by this implementation outperforms Single Path One Shot, FBNet, MnasNet, DARTS, NASNET, PNASNET by a good margin in all factors of FLOPS, # of parameters and Top-1 accuracy. Also for considering the MicroNet Challenge Σ score, without any quantization, it outperforms MobileNet V2, V3, ShuffleNet V1, V2, V2+.
