
# Abstract

Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider. In this work, we provided an open-sourced weight sharing Neural Architecture Search (NAS) pipeline, which can be **trained and searched on ImageNet totally within 60 GPU hours** (on 4 V100 GPUS) **in the exploration space of about 32^20 choices**.

This implementation searched a new state-of-the-art subnet model outperforming Single Path One Shot, FBNet,MnasNet, DARTS, NASNET, PNASNET by a good margin in all factors of FLOPS, # of parameters and Top-1 accuracy. Also for considering the MicroNet Challenge Σ score, without any quantization, it outperforms MobileNet V2, V3, ShuffleNet V1, V2, V2+ too.

| Model   | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  1684M  |  15.4M  |  62.9   |   84.5   | 3.67 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |
|    **OneShot-S+ int8** |  148M |  0.95M |  **75.0**   |   **92.0**   | **0.26** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+ float16** |  438M |  1.85M |  **75.7**   |   **92.9**   | **0.64** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+** |  588M |  3.65M |  **75.7**   |   **92.9**   | **1.03** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot (our) |  656M |  3.5M |  74.0   |   91.6   | 1.05 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch.sh) | [log](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/logs/shufflenas_oneshot.log) |
|    OneShot (paper) |  656M |  3.5M |  74.9   |   92.0   | 1.05 | - | - |
|    MnasNet|  634M |  4.2M |  74.0   |  91.8   | 1.15 | - | - |
|    MobileNetV3 Large|	 **434M** |	5.4M |	75.2|	- | 1.15 | - | - |
|    FBNet-B|  590M|  4.5M |  74.1   |   -   | 1.16 | - | - |
|    DARTS|  1148M|  4.7M |  73.3   |   91.3  | 1.66 | - | - |
|    NASNET-A|  1128M |  5.3M |  74.0   |   91.6   | 1.73 | - | - |
|    PNASNET|  1176M |  5.1M |  74.2   |   91.9   | 1.74 | - | - |
|    MobileNetV2 (1.4) |	1170M |	6.9M |	74.7 |	- | 2.00 | - | - |

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

|op_name                                 |  quantizable      |   inp_size|   kernel_size|           Cin|          Cout|       params(M)|   mults(M)|    adds(M)|     MFLOPS|
|:-----                                   | :-----: | :-----:  | :-----:   | :-----:|  :-----: |  :-----: |   :-----:  |   :-----: |   :-----:| 
|First conv                                   | False          |        224|            -1|             3|            16|           0.000|      5.419|      5.218|     10.637|
|HSwish                                   | False          |        112|            -1|            16|            16|           0.000|      0.603|      0.201|      0.804|
|SNB0                                   | False          |        112|            -1|            16|            64|           0.005|     23.800|     21.739|     45.539|
|SNB1                                   | False          |         56|            -1|            64|            64|           0.004|     12.136|     11.255|     23.391|
|SNB2                                   | False          |         56|            -1|            64|            64|           0.002|     10.511|      9.697|     20.208|
|SNB3                                   | False          |         56|            -1|            64|            64|           0.005|     16.389|     15.451|     31.840|
|SNB4                                   | False          |         56|            -1|            64|            160|           0.021|     32.111|     30.707|     62.818|
|SNB5                                   | False          |         28|            -1|           160|            160|           0.023|     17.573|     16.859|     34.432|
|SNB6                                   | False          |         28|            -1|           160|            160|           0.014|      9.746|      9.232|     18.978|
|SNB7                                   | False          |         28|            -1|           160|            160|           0.015|     11.103|     10.538|     21.641|
|SNB8                                   | False          |         28|            -1|           160|           320|           0.082|     14.060|     13.692|     27.752|
|SNB9                                   | False          |         14|            -1|           320|           320|           0.080|     11.834|     11.582|     23.416|
|SNB10                                   | False          |         14|            -1|           320|           320|           0.051|      6.416|      6.215|     12.631|
|SNB11                                   | False          |         14|            -1|           320|           320|           0.063|      8.898|      8.673|     17.571|
|SNB12                                   | False          |         14|            -1|           320|           320|           0.080|     11.834|     11.582|     23.416|
|SNB13                                   | False          |         14|            -1|           320|           320|           0.091|     14.168|     13.891|     28.059|
|SNB14                                   | False          |         14|            -1|           320|           320|           0.098|     15.448|     15.146|     30.594|
|SNB15                                   | False          |         14|            -1|           320|           320|           0.103|     16.501|     16.199|     32.700|
|SNB16                                   | False          |         14|            -1|           320|           320|           0.323|     25.640|     25.380|     51.020|
|SNB17                                   | False          |          7|            -1|           640|           640|           0.244|      8.311|      8.196|     16.507|
|SNB18                                   | False          |          7|            -1|           640|           640|           0.298|     10.983|     10.856|     21.839|
|SNB19                                   | False          |          7|            -1|           640|           640|           0.368|     14.445|     14.293|     28.738|
|GAP                                   | False          |          7|            -1|           640|             640|           0.000|      0.001|      0.031|      0.032|
|Last conv                                   | False          |          1|            -1|           640|          1024|           0.656|      0.655|      0.655|      1.310|
|HSwish                                   | False          |          1|            -1|          1024|          1024|           0.000|      0.003|      0.001|      0.004|
|Classifier                                   | False          |          1|            -1|          1024|          1000|           1.025|      1.024|      1.024|      2.048|
|total_quant                             | True           |           |              |              |              |           3.520|    292.820|    286.218|    579.038|
|total_no_quant                          | False          |           |              |              |              |           0.132|      6.801|      2.105|      8.905|
|total                                   | False          |           |              |              |              |           3.652|    299.621|    288.323|    587.943|

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
