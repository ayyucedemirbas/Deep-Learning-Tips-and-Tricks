# Atrous Convolution

* As we add more number of convolution layers to the neural network, we enlarge receptive field of feature maps and capture long range contextual information of the image. However, while doing that, we need to reduce spatial resolution by max-pooling or striding. 
There are 2 main reasons for that:
   - Without striding and pooling, we cannot enlarge receptive field sufficiently and learn long range contextual features.
   -  We need to decrease spatial size if we increase the number of channels to create a balance in terms of computational complexity. 

* This situation triggers 2 main problems:
  - Reduced resolution due to consecutive striding and max pooling causes important low level details to be lost, which affects
   segmentation of object boundaries and small scale pieces adversely. This is called as decimation of detailed information.
  - As we go deeper in neural networks, extracted features become more generalized and abstract, which is not useful to dense
   prediction tasks like segmentation. 

* Atrous convolution can solve both of these problems. By adding holes between filter weights, it enlarges filter kernels before 
convolution operation. The number of holes is controlled by dilation rate. This approach helps to extract denser feature maps and 
increase receptive field rapidly without any need of pooling or striding.

* This means that we do not have to use max-pooling or striding in convolution layers for the rapid increase in receptive field,
  instead we can use dilated convolution. It can enlarge the receptive field with same proportion, but it can also preserve spatial
  resolution at the same time.
  
* Apart from these, it also allows us to control how densely feature maps are computed in convolution backbone. At this point, the
  phrase *"the generation of dense/denser feature maps"* seems confusing, but this is related to the proportion of input resolution to
  final feature map resolution. Let's assume that input image is $256 \times 256$, and final feature maps to be fed into dense layer are
  of $16 \times 16$. In that case, the ratio is equal to $16$. If we apply one max-pooling and downsample it into $8 \times 8$ maps,
  spatial density of new feature responses would be lower. Hence, denser features can be interpreted as smaller input-output spatial
  ratio. 

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Convolutional-Neural-Networks/blob/main/Atrous%20Convolution/images/dilated%20convolution.png" width="800" height="300" />
</p>

* The usage of atrous convolution with spatial pyramid pooling, called as ASPP module, was proposed in DeepLabV2, but it was also incorporated in next version, which is DeepLabV3. Atrous convolution layers in this module are initialized with different dilation 
rates and applied in parallel to capture multi-scale information. Detection and segmentation of objects at multiple scales are some issues encountered in deep convolution architectures. ASPP module is capable of extracting multi-scale context, so can alleviate this problem.

## ResNet Summary:

* All ResNet architectures of version-1 start with a convolution layer of $7 \times 7$ kernel and $3 \times 3$ max-pooling. Both of them downsample the image by stride of $2$. At the end of these two layers, the proportion of input resolution to extracted feature maps becomes $4$.
  
* Then, $4$ convolutional blocks come in. The structure and the size of filters in these blocks tend to vary depending on type of ResNet architecture. In ResNet-18/34, each block accomodates $2$ number $3 \times 3$ convolution layers, whereas larger ResNet models uses bottleneck-block of $3$ consecutive convolution layers.

* The common thing that all Res-Net variations have is that feature map resolution is reduced by half per block. In this case, the output of entire network has $64$ input-output resolution ratio. 

## DeepLabV3

* DeepLabV3 actually relies on the combination of ResNet and ASPP module to alleviate its reduced feature map resolution problem. ASPP module consists of $4$ parallel convolution layers followed by batch-norm and relu activation. First $3$ of these layers utilizes $3 \times 3$ kernel with dilation rate of $6$, $12$, and $18$, whereas last convolution adopts $1 \times 1$ kernel. For all of them, $256$ filters are used with same padding.

* The main problem in ASPP module is the degeneration: As dilation rate gets larger, filter weights of its convolution layers are surpassed. In other words, fewer number of kernel weights are applied to valid image context. This is expressed in the paper with the following words: *"As sampling rate becomes larger, the number of valid filter weights (the weights that are applied to valid feature region instead of padded zeros) becomes smaller"*. To solve this problem and cover global feature representatives to model, we insert image pooling module aside ASPP module, which is composed of average pooling, 1x1 convolution with 256 filters, batch-norm layer and upsampling operator.

<p align="center">
  <img src="https://github.com/GoktugGuvercin/Convolutional-Neural-Networks/blob/main/Atrous%20Convolution/images/DeepLabV3%20Encoder.png" width="800" height="500" />
</p>



