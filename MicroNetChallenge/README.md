
# Abstract

Designing convolutional neural networks (CNN) for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant efforts have been dedicated to design and improve mobile CNNs on all dimensions, it is very difficult to manually balance these trade-offs when there are so many architectural possibilities to consider[[1]](https://arxiv.org/pdf/1807.11626.pdf). In this work, we provided an open-sourced weight sharing Neural Architecture Search (NAS) pipeline, which can be **trained and searched on ImageNet totally within `60` GPU hours** (on 4 V100 GPUs, including supernet training, supernet searching and the searched best subnet training) **in the exploration space of about `32^20` choices**.

This implementation has searched a new state-of-the-art subnet model which **outperforms** other NAS searched models like `Single Path One Shot, FBNet, MnasNet, DARTS, NASNET, PNASNET` by a good margin in all factors of FLOPS, number of parameters and Top-1 / Top-5 accuracies. Also for considering [the MicroNet Challenge Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html), before any quantization, it **outperforms** other popular base models like `MobileNet V2, V3, ShuffleNet V1, V2, V2+` too. For `float16`, OneShot-S+ score is `0.64` and `0.28` for `int8`. Check [here](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/tree/master/MicroNetChallenge#searched-model-profiling) for profiling and score calculation. 

**To verify the model's performance**, please refer to the [inference readme](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/MicroNetChallenge/INFERENCE_README.md).

| Model   | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  1684M  |  15.4M  |  62.90   |   84.49   | 3.67 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |
|    **OneShot-S+ int8** |  154M |  1.01M |  **75.00***   |   **92.00***   | **0.28** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+ float16** |  438M |  1.85M |  **75.75**   |   **92.77**   | **0.64** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+** |  588M |  3.65M |  **75.74**   |   **92.77**   | **1.03** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot (our) |  656M |  3.5M |  74.02   |   91.60   | 1.05 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch.sh) | [log](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/logs/shufflenas_oneshot.log) |
|    OneShot (paper) |  656M |  3.5M |  74.9   |   92.0   | 1.05 | - | - |
|    MnasNet|  634M |  4.2M |  74.0   |  91.8   | 1.15 | - | - |
|    MobileNetV3 Large|     **434M** |    5.4M |    75.2|    - | 1.15 | - | - |
|    FBNet-B|  590M|  4.5M |  74.1   |   -   | 1.16 | - | - |
|    DARTS|  1148M|  4.7M |  73.3   |   91.3  | 1.66 | - | - |
|    NASNET-A|  1128M |  5.3M |  74.0   |   91.6   | 1.73 | - | - |
|    PNASNET|  1176M |  5.1M |  74.2   |   91.9   | 1.74 | - | - |
|    MobileNetV2 (1.4) |    1170M |    6.9M |    74.7 |    - | 2.00 | - | - |

*For the int8 performance, please refer to [this section](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/MicroNetChallenge/README.md#searched-model-performance).

# Our approach

Our approach is **mainly based on the Single Path One Shot NAS in the combination of Squeeze and Excitation (SE), ShuffleNet V2+ and MobileNet V3**. Like the original paper, we searched for the choice blocks and block channels with multiple FLOPs and parameter amount constraints. In this section, we will elaborate on the modifications from the original paper.

## Supernet Structure Design

For each `ShuffleNasBlock`, four choice blocks were explored, `ShuffleNetBlock-3x3 (SNB-3)`, `SNB-5`, `SNB-7` and `ShuffleXceptionBlock-3x3 (SXB-3)`. Within each block, eight channel choices are avialable: `[0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] * (BlockOutputChannel / 2)`. So each `ShuffleNasBlock` explores `32` possible choices and there are `20` blocks in this implementation, counting for totaly `32^20` design choices.

We also applied the SE, ShuffleNet V2+ SE layout and the MobileNet V3 last convolution block design in the supernet. Finally, the supernet contains `15.4` Million trainable parameters and the possible subnet FLOPs range from `336M` to `1682M`.

## Supernet Training

Unlike what the original paper did, in the training stage, we didn't apply uniform distribution from the beginning. We train the supernet totally `120` epochs. In the first `60` epochs doing Block selection only and, for the upcoming `60` epochs, we used **Channel Selection Warm-up** which gradually allows the supernet to be trained with a larger range of channel choices.

``` python
   # Supernet sampling schedule: during channel selection warm-up
   1 - 60 epochs:          Only block selection (BS), Channels are set to maximum (here [2.0])
   61 epoch:               [1.8, 2.0] + BS
   62 epoch:               [1.6, 1.8, 2.0] + BS
   63 epoch:               [1.4, 1.6, 1.8, 2.0] + BS
   64 epoch:               [1.2, 1.4, 1.6, 1.8, 2.0] + BS
   65 - 66 epochs:         [1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS
   67 - 69 epochs:         [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS
   70 - 73 epochs:         [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0] + BS 
```

The reason why we did this in the supernet training is that during our experiments we found, **for supernet without SE**, doing Block Selection from beginning works well, nevertheless doing Channel Selection from the beginning will cause the network not converging at all. The Channel Selection range needs to be gradually enlarged otherwise it will crash with free-fall drop accuracy. And the range can only be allowed for `(0.6 ~ 2.0)`. Smaller channel scales will make the network crashing too. **For supernet with SE**, Channel Selection with the full choices `(0.2 ~ 2.0)` can be used from the beginning and it converges. However, doing this seems like harming accuracy. Compared to the same se-supernet with Channel Selection warm-up, the Channel Selection from scratch model has been always left behind `10%` training accuracy during the whole procedure. 

## Subnet Searching

Different from the paper, we **jointly searched** for the Block choices and Channel Choices in the supernet at the same time. It means that for each instance in the population of our genetic algorithm it contains `20` Block choice genes and `20` Channel choice genes. We were aiming to find a combination of these two which optimizing for each other and being complementary.

For each qualified subnet structure (has lower `Σ Normalized Scores` than the baseline OneShot searched model), like most weight sharing NAS approaches did, we updated the BN statistics firstly with `20,000` fixed training set images and then evaluate this subnet ImageNet validation accuracy as the indicator for its performance.


## Subnet Training

For the final searched model, we **build and train it from scratch**. No previous supernet weights are reused in the subnet.

As for the hyperparameters. We modified the GluonCV official ImageNet training script to support both supernet training and subnet training. We trained both models with initial learning rate `1.3`, weight decay `0.00003`, cosine learning rate scheduler, 4 GPUs each with batch size `256`, label smoothing and no weight decay for BN beta gamma. Supernet was trained `120` epochs and subnet was trained `360` epochs. 


# Results

## Supernet Training

| Model   | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  1684M  |  15.4M  |  62.9   |   84.5   | 3.67 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |

## Supernet Searching
![alt text](../images/search_supernet.gif)

We tried both random search, randomly selecting 250 qualified instances to evaluate their performance, and genetic search. The genetic method easily found a better subnet structure over the random selection.

## Searched Model Profiling

A detailed op to op profiling can be found [here](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/MicroNetChallenge/detailed_profiling.md).

|op_name                                 |  quantizable      |   inp_size|   kernel_size|           Cin|          Cout|       params(M)|   mults(M)|    adds(M)|     MFLOPS|
|:-----                                   | :-----: | :-----:  | :-----:   | :-----:|  :-----: |  :-----: |   :-----:  |   :-----: |   :-----:| 
|First conv                                   | True          |        224|            3|             3|            16|           0.000|      5.419|      5.218|     10.637|
|HSwish                                   | False          |        112|            -1|            16|            16|           0.000|      0.603|      0.201|      0.804|
|SNB-3x3                                   | Mixed          |        112|            3|            16|            64|           0.005|     23.800|     21.739|     45.539|
|SNB-3x3                                   | Mixed          |         56|            3|            64|            64|           0.004|     12.136|     11.255|     23.391|
|SNB-3x3                                   | Mixed          |         56|            3|            64|            64|           0.002|     10.511|      9.697|     20.208|
|SNB-5x5                                   | Mixed          |         56|            5|            64|            64|           0.005|     16.389|     15.451|     31.840|
|SNB-3x3                                   | Mixed          |         56|            3|            64|            160|           0.021|     32.111|     30.707|     62.818|
|SNB-3x3                                   | Mixed          |         28|            3|           160|            160|           0.023|     17.573|     16.859|     34.432|
|SNB-5x5                                   | Mixed          |         28|            5|           160|            160|           0.014|      9.746|      9.232|     18.978|
|SNB-3x3                                   | Mixed          |         28|            3|           160|            160|           0.015|     11.103|     10.538|     21.641|
|SXB-3x3                                   | Mixed          |         28|            3|           160|           320|           0.082|     14.060|     13.692|     27.752|
|SNB-7x7                                   | Mixed          |         14|            7|           320|           320|           0.080|     11.834|     11.582|     23.416|
|SNB-3x3                                   | Mixed          |         14|            3|           320|           320|           0.051|      6.416|      6.215|     12.631|
|SNB-5x5                                   | Mixed          |         14|            5|           320|           320|           0.063|      8.898|      8.673|     17.571|
|SNB-7x7                                   | Mixed          |         14|            7|           320|           320|           0.080|     11.834|     11.582|     23.416|
|SNB-7x7                                   | Mixed          |         14|            7|           320|           320|           0.091|     14.168|     13.891|     28.059|
|SNB-5x5                                   | Mixed          |         14|            5|           320|           320|           0.098|     15.448|     15.146|     30.594|
|SNB-7x7                                   | Mixed          |         14|            7|           320|           320|           0.103|     16.501|     16.199|     32.700|
|SNB-3x3                                   | Mixed          |         14|            3|           320|           320|           0.323|     25.640|     25.380|     51.020|
|SNB-3x3                                   | Mixed          |          7|            3|           640|           640|           0.244|      8.311|      8.196|     16.507|
|SNB-7x7                                   | Mixed          |          7|            7|           640|           640|           0.298|     10.983|     10.856|     21.839|
|SNB-3x3                                   | Mixed          |          7|            3|           640|           640|           0.368|     14.445|     14.293|     28.738|
|GAP                                   | False          |          7|            -1|           640|             640|           0.000|      0.001|      0.031|      0.032|
|Last conv                                   | True          |          1|            1|           640|          1024|           0.656|      0.655|      0.655|      1.310|
|HSwish                                   | False          |          1|            -1|          1024|          1024|           0.000|      0.003|      0.001|      0.004|
|Classifier                                   | True          |          1|            1|          1024|          1000|           1.025|      1.024|      1.024|      2.048|
|Total_quant                             | True           |           |              |              |              |           3.520|      292.8|      286.2|      579.0|
|Total_no_quant                          | False          |           |              |              |              |           0.132|      8.553|      2.105|      10.66|
|Total                                   |           |           |              |              |              |                3.652|      301.4|      288.3|      589.6|

```python
Float32 scores: 
param score: 3.652 / 6.9 = 0.529
flop score: 589.6 / 1170 = 0.504
Σ scores: 1.03

Float16 scores: 
param score: 3.652 / 2 / 6.9 = 0.265
flop score: (301.4 / 2 + 288.3) / 1170 = 0.375
Σ scores: 0.640

Int8 scores: 
param score: (3.52 / 4 + 0.132) / 6.9 = 0.147
flop score: ((292.8 / 4 + 8.55) + (286.2 / 4 + 2.11)) / 1170 = 0.133
Σ scores: 0.280
```

## Searched Model Performance

| Model   | FLOPs | # of Params   | Top - 1 | Top - 5 | [Σ Normalized Scores](https://micronet-challenge.github.io/scoring_and_submission.html) | Scripts | Logs |
| :--------------------- | :-----: | :------:  | :-----: | :-----: | :---------------------: | :-----: |  :-----: | 
|    OneShot+ Supernet |  1684M  |  15.4M  |  62.90   |   84.49   | 3.67 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_supernet.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_supernet.log) |
|    **OneShot-S+ int8** |  154M |  1.01M |  **75.00***   |   **92.00***   | **0.28** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+ float16** |  438M |  1.85M |  **75.75**   |   **92.77**   | **0.64** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    **OneShot-S+** |  588M |  3.65M |  **75.74**   |   **92.77**   | **1.03** | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch%2B.sh) | [log](https://github.com/CanyonWind/oneshot_nas/blob/master/logs/shufflenas_oneshot%2B.log) |
|    OneShot (our) |  656M |  3.5M |  74.02   |   91.60   | 1.05 | [script](https://github.com/CanyonWind/oneshot_nas/blob/master/scripts/train_fixArch.sh) | [log](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/logs/shufflenas_oneshot.log) |
|    OneShot (paper) |  656M |  3.5M |  74.9   |   92.0   | 1.05 | - | - |
|    MnasNet|  634M |  4.2M |  74.0   |  91.8   | 1.15 | - | - |
|    MobileNetV3 Large|     **434M** |    5.4M |    75.2|    - | 1.15 | - | - |
|    FBNet-B|  590M|  4.5M |  74.1   |   -   | 1.16 | - | - |
|    DARTS|  1148M|  4.7M |  73.3   |   91.3  | 1.66 | - | - |
|    NASNET-A|  1128M |  5.3M |  74.0   |   91.6   | 1.73 | - | - |
|    PNASNET|  1176M |  5.1M |  74.2   |   91.9   | 1.74 | - | - |
|    MobileNetV2 (1.4) |    1170M |    6.9M |    74.7 |    - | 2.00 | - | - |
 
*The int8 quantized OneShot-S+ model has been provided. But because of [this issue](https://github.com/apache/incubator-mxnet/issues/16424), it seems like the MXNet MKL quantization backend has some bugs and there are people from Intel working on fixing them. According to the [MXNet quantization benchmark](https://github.com/CanyonWind/MXNet-Single-Path-One-Shot-NAS/blob/master/MicroNetChallenge/mxnet_quantization_benchmark.md), **no model's Top-1 accuracy drops more than `0.5%`**. So we are claiming that, as long as the MXNet MKL bug has been fixed, our Oneshot-S+ model has a very high chance to pass the `75.0%` criterion for MicroNet Challenge. If the bug cannot be fixed in time (before the MicroNet Challenge result release date), we have to submit with the float16 score. 

# Summary
In this work, we provided a state-of-the-art open-sourced weight sharing Neural Architecture Search (NAS) pipeline, which can be trained and searched on ImageNet totally within `60` GPU hours (on 4 V100 GPUS) and the exploration space is about `32^20`. The model searched by this implementation outperforms the other NAS searched models, such as `Single Path One Shot, FBNet, MnasNet, DARTS, NASNET, PNASNET` by a good margin in all factors of FLOPS, # of parameters and Top-1 accuracy. Also for considering the MicroNet Challenge Σ score, without any quantization, it outperforms `MobileNet V2, V3, ShuffleNet V1, V2, V2+`.

We have not tried to use more aggressive weight / channel pruning or more complex low-bit quantization methods, because, if we want to take full advantage of them, most compression methods and low-bit quantization models require custom hardware. However, in general practical situations, we need to build / design a model that meets hardware constraints, but not build the hardware architecture based on the algorithm. The good thing about the MicroNet Challenge is, of course, that because of Google's backing, excellent quantization algorithms along with hardware can be put into usage easier in the near future. But that is the next step. Focusing on what we currently have, we believe that our direction - design optimal searching space and search for further optimized network structures - is more suitable for direct application. Based on these NAS searched models and other efficient base networks such as `MobileNet series`, `ShuffleNet series` and `EfficientNet`, in the future also combined with the compression / low-bit methods which already supported by the hardware, the edge-side neural network will be better serving the industry and people's everyday life.
