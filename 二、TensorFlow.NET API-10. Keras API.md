# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 10. Keras API

### 10.1 Keras 简要介绍

2015年11月9日，Google 在 GitHub 上开源了 TensorFlow，同一年的3月，Keras 发布，支持 "TensorFlow", "Theano" 和"CNTK" 等 backend (后端)。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608022347132.png" alt="1608022347132" style="zoom:80%;" />

Keras 的出现大大降低了深度学习的入门门槛，易用性的同时也并不降低灵活性，我们从 Keras 官网的文字可以总结出 Keras 的主要特点：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608104090066.png" alt="1608104090066" style="zoom:80%;" />

- 为人类习惯设计的深度学习

  Keras是为人类思维习惯而非机器逻辑设计的API。Keras 始终遵循最少学习成本的准则，它提供了简单易用的API，尽量减少经常使用的深度学习方法需要的开发量，并且提供了清晰可快速定位排查的错误消息机制，同时，它还具有大量完善的文档和开发指南。

- 创意的快速实现和灵活性

  Keras 是 [Kaggle](https://www.kaggle.com/) 上前5名获胜团队中最常用的深度学习框架，因为Keras的快速模型编写方式，使得进行创新的实验变得更加容易，所以它能使你比对手更快地尝试更多的想法。欧洲核子研究组织（CERN），美国国家航空航天局（NASA），美国国立卫生研究院（NIH）和世界上许多其他科学组织都在使用Keras（包括大型强子对撞机也在使用Keras）。Keras具有底层灵活性，可以开发任意的研究思路和创意想法，同时提供可选的高级便利功能，以加快实验的整体周期。

- 万亿级的机器学习部署

  [Keras](https://www.tensorflow.org/) 建立在 [TensorFlow 2.0](https://www.tensorflow.org/) 之上，是一个行业领先的框架，可以扩展到大型GPU集群或整个 [TPU云端](https://cloud.google.com/tpu)。这不仅是可能的，而且很容易实现。 

- 模型适用多平台部署

  充分利用 TensorFlow 框架的全面部署功能，你可以将 Keras 模型导出为 JavaScript 方式直接在浏览器中运行，也可以导出为 TF Lite 方式在iOS、Android 和嵌入式设备上运行，也可以通过 Web API  在 Web 上轻松部署 Keras 模型。 

- 丰富的生态系统

  Keras 紧密连接于 TensorFlow 2.x 生态系统的核心部分，涵盖了机器学习工作流程的每个步骤，从数据管理到超参数训练再到部署解决方案。 



2019年3月7日凌晨， 谷歌一年一度的 TensorFlow 开发者大会（TensorFlow DEV SUMMIT 2019）在加州举行。这次大会发布了  TensorFlow 2.0 Alpha 版，同时 TensorFlow 的 Logo 也变成了流行的扁平化设计。

 <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/196414261.gif" alt="img" style="zoom:80%;" /> 

同时伴随 Alpha 版本的更新，TensorFlow 团队表达了对 Keras 更深的爱，Keras 整合进了 tf.keras 高层API，成为了 TensorFlow 的一部分。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608105508138.png" alt="1608105508138" style="zoom:80%;" />

2020年，Keras之父  **弗朗索瓦.肖莱(Francois Chollet)**  正式加入 Google 人工智能小组，同年的 TensorFlow 开发者峰会（TensorFlow DEV SUMMIT 2020）上，Keras（tf.keras） 正式内置为 TensorFlow 的核心 API，得到 TensorFlow 的全面支持，并成为 TensorFlow 官方推荐的首选深度学习建模工具。自此，TensorFlow 和 Keras 完美地结合，成为大家入门深度学习的首选API。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608106027998.png" alt="1608106027998" style="zoom:67%;" />





### 10.2 模型（Model）与层（Layer） 

模型（Model）与层（Layer） 是 Keras 的2个最基本的概念。深度学习中的模型一般由各种层组合而成，层（Layer） 将各种常用的计算节点和变量进行了封装， 而模型则将各种层进行组织和连接，并打包成一个整体，描述了如何将输入数据通过各种层以及运算而得到输出。

Keras 在 `Tensorflow.Keras.Layers` 下内置了丰富的深度学习常用的各种功能的预定义层， 例如：

Layers.Dense

Layers.Flatten

Layers.RNN

Layers.BatchNormalization

Layers.Dropout

Layers.Conv2D

Layers.MaxPooling2D

Layers.Conv1D

Layers.Embedding

Layers.GRU

Layers.LSTM

Layers.Bidirectional

Layers.InputLayer

……等等

如果上述的内置网络层无法满足你的需求，Keras 也允许我们自定义层，通过继承 Tensorflow.Keras.Engine.Layer 基类构建自定义的网络层。 



**常用的网络层（Layer）** 

下面选择一些常用的内置模型层进行简单的介绍：

**基础层**

- Dense：全连接层，或称MLP。逻辑上等价于这样一个函数：权重W为m*n的矩阵，输入x为n维向量，激活函数Activation，偏置bias，输出向量out为m维向量，函数如下，out=Activation(Wx+bias)，即一个线性变化加一个非线性变化产生输出。
- Activation：激活函数层。一般放在Dense层后面，等价于在Dense层中指定激活函数，激活函数可以提高模型的非线性表达能力。
- Dropout：随机丢弃层。训练期间以一定几率将输入置0（等同于按照一定的概率将神经网络单元暂时从网络中丢弃），是一种正则化的手段， 可以作为CNN中防止过拟合提高训练效果的一个大神器 。
- BatchNormalization：批标准化层。通过线性变换将输入批次缩放平移到稳定的均值和标准差，可以增强模型对输入不同分布的适应性，加快模型训练速度，有轻微正则化效果。一般在激活函数之前使用。
- SpatialDropout2D：空间随机置零层。训练期间以一定几率将整个特征图置0，是一种正则化手段，有利于避免特征图之间过高的相关性。
- InputLayer：输入层。通常使用Functional API方式构建模型时作为第一层。
- DenseFeature：特征转换层，用于接收一个特征列表并产生一个全连接层。
- Flatten：展平层，用于将多维张量展开压平成一维。
- Reshape：形状变换层，改变张量的形状。
- Concatenate：拼接层，将多个张量在某个维度上拼接。
- Add：加法层。
- Subtract： 减法层。
- Maximum：取最大值层。
- Minimum：取最小值层。

**卷积网络相关层**

- Conv1D：普通一维卷积，常用于文本和时间序列。该层创建一个卷积内核，该卷积内核在单个空间（或时间）维度上与该层输入进行卷积，以生成输出张量。 
- Conv2D：普通二维卷积，常用于图像的空间卷积。该层创建一个卷积内核，该卷积内核与该层输入进行卷积以产生输出张量。
- Conv3D：普通三维卷积，常用于视频或体积上的空间卷积。该层创建一个卷积内核，该卷积内核与该层输入进行卷积以产生输出张量。
- SeparableConv2D：二维深度可分离卷积层。不同于普通卷积同时对区域和通道操作，深度可分离卷积先操作区域，再操作通道。即先对每个通道做独立卷积操作区域，再用1乘1卷积跨通道组合操作通道。直观上，可以理解为将卷积内核分解为两个较小内核的一种方式，或者是Inception块的一种极端版本。参数个数 = 输入通道数×卷积核尺寸 + 输入通道数×1×1×输出通道数。深度可分离卷积的参数数量一般远小于普通卷积，效果一般也更好。
- DepthwiseConv2D：二维深度卷积层。仅有SeparableConv2D前半部分操作，即只操作区域，不操作通道，一般输出通道数和输入通道数相同，但也可以通过设置depth_multiplier让输出通道为输入通道的若干倍数。输出通道数 = 输入通道数 × depth_multiplier，参数个数 = 输入通道数×卷积核尺寸× depth_multiplier。
- Conv2DTranspose：二维卷积转置层，也称反向卷积层。并非卷积的逆操作，但在卷积核相同的情况下，当其输入尺寸是卷积操作输出尺寸的情况下，卷积转置的输出尺寸恰好是卷积操作的输入尺寸。
- LocallyConnected2D: 二维局部连接层。类似Conv2D，唯一的差别是没有空间上的权值共享，也就是说，在输入的每个不同色块上应用了一组不同的过滤器，所以其参数个数远高于二维卷积。
- MaxPool2D: 二维最大池化层，也称作下采样层。池化层无可训练参数，主要作用是降维。
- AveragePooling2D: 二维平均池化层。执行空间数据的平均池化操作。 
- GlobalMaxPool2D: 全局最大池化层。每个通道仅保留一个值，一般从卷积层过渡到全连接层时使用，是Flatten的替代方案。
- GlobalAveragePooling2D: 全局平均池化层。空间数据的全局平均池化操作，每个通道仅保留一个值。

**循环网络相关层**

- Embedding：嵌入层。将正整数（索引）转换为固定大小的密集向量，例如 `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`，该层只能用作模型中的第一层。这是一种比One-Hot更加有效的对离散特征进行编码的方法,一般用于将输入中的单词映射为稠密向量。嵌入层的参数需要学习。
- LSTM：长短记忆循环网络层 -Hochreiter 1997 ，最普遍使用的循环网络层。具有携带轨道、遗忘门、更新门、输出门，可以较为有效地缓解梯度消失问题，从而能够适用长期记忆依赖问题。设置return_sequences = True时可以返回各个中间步骤输出，否则只返回最终输出。
- GRU：门控循环单元 - Cho et al. 2014 。LSTM的低配版，不具有携带轨道，参数数量少于LSTM，训练速度更快。
- SimpleRNN：简单循环网络层。完全连接的RNN，其中输出将反馈到输入。容易存在梯度消失，不能够适用长期依赖问题。一般较少使用。
- ConvLSTM2D：卷积长短记忆循环网络层。结构上类似LSTM，但对输入的转换操作和对状态的递归转换操作都是卷积运算。
- Bidirectional：双向循环网络包装器，RNN的双向包装。可以将LSTM，GRU等层包装成双向循环网络，从而增强特征提取能力。
- RNN：RNN基本层。接受一个循环网络单元或一个循环单元列表，通过调用 Tensorflow.Keras.Layers.RNN 类的构造函数在序列上进行迭代从而转换成循环网络层。
- Attention：Dot-product类型注意力机制层，也称 Luong-style  关注。可以用于构建注意力模型。
- AdditiveAttention：Additive类型注意力机制层，也称 Bahdanau-style 关注。可以用于构建注意力模型。
- TimeDistributed：时间分布包装器。包装后可以将Dense、Conv2D等一层网络层作用到每一个输入的时间片段上。



**自定义的网络层（Layer）** 

如果现有的这些层无法满足你的要求，我们可以通过继承 `Tensorflow.Keras.Engine.Layer` 来编写自己的自定义层。

自定义层需要继承 `Tensorflow.Keras.Engine.Layer` 类，重新实现 `Layer构造函数` ，并重写 `build` 和 `call` 这2个方法（build方法一般定义Layer需要被训练的参数，call方法一般定义正向传播运算逻辑），你也可以添加自定义的方法。如果要让自定义的 Layer 通过 Functional API 组合成模型时可以被保存成 h5 模型，需要自定义get_config方法。

自定义层的模板如下：

//TODO：下述代码需要转换为C#方式---------------------------------------------------------------------------

```python
class MyLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # 初始化代码

    def build(self, input_shape):     # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状。
        # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        self.variable_0 = self.add_weight(...)
        self.variable_1 = self.add_weight(...)

    def call(self, inputs):
        # 模型调用的代码（处理输入并返回输出）
        return output
```

//TODO：-------------------------------------------------------------------------------------------------------------------



下面是一个简化的全连接层的范例，类似 Dense，此代码在 `build` 方法中创建两个变量，并在 `call` 方法中使用创建的变量进行运算： 

//TODO：下述代码需要转换为C#方式---------------------------------------------------------------------------

```python
class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):     # 这里 input_shape 是第一次运行call()时参数inputs的形状
        self.w = self.add_variable(name='w',
            shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b',
            shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred
```

//TODO：-------------------------------------------------------------------------------------------------------------------



在定义模型的时候，我们便可以如同 Keras 中的其他层一样，调用我们自定义的层 `LinearLayer`： 

//TODO：下述代码需要转换为C#方式---------------------------------------------------------------------------

```python
class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(units=1)

    def call(self, inputs):
        output = self.layer(inputs)
        return output
```

//TODO：-------------------------------------------------------------------------------------------------------------------



介绍完 Keras 中的层，我们来看下 Keras 中模型的结构，以及如何自定义模型。



**自定义的模型（Model）**

我们先来了解下 Keras 模型类的示意图：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608257005411.png" alt="1608257005411" style="zoom:80%;" />

自定义模型需要继承 Tensorflow.Keras.Engine.Model 类，然后在构造函数中初始化所需要的层（可以使用 Keras 的层或者继承 Layer 进行自定义层），并重载 call() 方法进行模型的调用，建立输入和输出之间的函数关系。

我们可以通过自定义模型类的方式编写简单的线性模型 `y_pred = a * X + b`  ，实例代码如下：















































### 10.3 Keras 常用训练和参数类 API





### 10.4 Keras 建立模型的3种方式

//keras建模的一般步骤 先画示意图















