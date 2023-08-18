# Deep Learning Tips & Tricks


**Architecture Design and Basics:**
   - Choose an appropriate architecture based on the problem: CNNs for image tasks, RNNs/LSTMs/GRUs for sequential data, Transformers for sequences and attention-based tasks, etc.
   - Start with simpler architectures and gradually increase complexity if needed.
   - Model selection should be driven by the size of your dataset, available resources, and problem domain.

**Convolutional Neural Networks (CNNs):**
   - Larger filter sizes (7x7, 5x5) capture more complex features but are computationally expensive.
   - Smaller filter sizes (3x3) with multiple layers can achieve similar receptive fields with fewer parameters.
   - Padding helps retain spatial dimensions and is essential for preserving information at the edges.

**Smaller or Larger Kernel Size in CNNs:**

- Convolution layers with 5x5 or 7x7 kernels tend to increase receptive field faster than their 3x3 versions. However,
  many state of the art architectures prefer to use 3x3 convolution layers. To reach enough receptive field compared to
  larger kernels, they generally deploy multiple layers. In the below, you can see the receptive field covered by convolution
  layers with different kernel sizes

- Receptive Field Comparison:
  * 2 consecutive 3x3 conv layers = 1 5x5 conv layer
  * 3 consecutive 3x3 conv layers = 1 7x7 Conv Layer

- In fact, using multiple conv layers with small kernel instead of single layer with larger kernel introduces some advantages:

  * Going through multiple non-linear activations instead of single one, this makes decision function in the model more
    discriminative

  * Number of parameters and computations would be decreased in small kernels. Working with larger kernels especially in
    3D domain like medical imaging is a bad decision. Small kernels are highly recommended at this point.
  

**Receptive Field and Layer Depth:**
   - Decide the number of layers based on the problem's complexity and desired receptive field.
   - Receptive field ratio guides the depth of the network, balancing between local and global features.

**Striding and Pooling:**
   - Strided convolution and pooling operations help to increase receptive field, thereby learning long-range dependencies.
   - While doing these, they also reduce spatial dimension of input. In that way, computational cost would be decreased.
   - However, reduced feature maps start to loose fine-grained information (low-level details) like boundaries and edges of organs.
     This decreases the quality of segmentation; hence, using skip connection can be good alternative to handle this problem.

**Atrous (Dilated) Convolutions:**
   - Atrous convolution is capable of increasing receptive field of feature maps without reducing resolution, so it is suitable for           retaining fine-grained details in segmentation tasks.

**Pooling and Segmentation:**
   - Pooling extracts geometrically invariant features but might affect segmentation tasks negatively.
   - Skip connections help preserve spatial details while still leveraging hierarchical features.

**Batch Normalization (Batch-Norm):**
   - Batch normalization stabilizes training by normalizing inputs to each layer, improving convergence and generalization.
   - Usually placed before activation functions (convolution, ReLU) for better performance.

**Depth-wise Separable Convolutions:**
   - Depth-wise separable convolutions reduce computational complexity by splitting spatial and channel convolutions.
   - Suitable for scenarios with limited computational resources.

**SPP Layer (Spatial Pyramid Pooling):**
   - SPP layers capture information at multiple scales, improving object detection by handling various object sizes.

**Residual Networks (ResNets) and Skip Connections:**
   - Residual connections facilitate gradient flow and alleviate vanishing gradient issues.
   - Skip connections help build deeper networks without suffering from degradation problems.

**Topological and Geometric Insights:**
   - Understand the topology of the data to guide architecture design.
   - Geometric invariance might not always be desirable, especially in tasks like segmentation.

**Backpropagation and Optimization:**
   - Use activation functions (ReLU, Leaky ReLU) to avoid vanishing gradients and improve convergence.
   - Employ optimization techniques like Adam, RMSProp, or SGD with momentum for faster convergence.
   - Learning rate scheduling can help in balancing exploration and exploitation during training.

**Regularization and Dropout:**
   - Regularization techniques like L2 regularization, dropout, and batch normalization help prevent overfitting.
   - Apply dropout in moderation to prevent underfitting.

 **Hyperparameter Tuning:**
   - Experiment with learning rates, batch sizes, and network architectures.
   - Utilize techniques like random search or Bayesian optimization for efficient hyperparameter tuning.

 **Data Augmentation:**
   - Apply data augmentation techniques (rotation, cropping, flipping) to increase the diversity of the training data.
   - Augmentation helps improve model generalization.

 **Transfer Learning and Pretrained Models:**
   - Utilize pretrained models and fine-tuning to leverage features learned on large datasets.
   - Adapt pretrained models to your specific task to improve convergence and performance.
