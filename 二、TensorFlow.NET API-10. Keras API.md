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

我们可以通过自定义模型类的方式编写简单的线性模型 `y_pred = a * X + b`  ，完整的实例代码如下：

```c#
using NumSharp;
using System;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace LinearRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            TFNET tfnet = new TFNET();
            tfnet.Run();
        }
    }
    public class TFNET
    {
        public void Run()
        {
            Tensor X = tf.constant(new[,] { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } });
            Tensor y = tf.constant(new[,] { { 10.0f }, { 20.0f } });

            var model = new Linear(new ModelArgs());
            var optimizer = keras.optimizers.SGD(learning_rate: 0.01f);
            foreach (var step in range(20))//20 iterations for test
            {
                using var g = tf.GradientTape();
                var y_pred = model.Apply(X);//using "y_pred = model.Apply(X)" replace "y_pred = a * X + b"
                var loss = tf.reduce_mean(tf.square(y_pred - y));
                var grads = g.gradient(loss, model.trainable_variables);
                optimizer.apply_gradients(zip(grads, model.trainable_variables.Select(x => x as ResourceVariable)));
                print($"step: {step},loss: {loss.numpy()}");
            }
            print(model.trainable_variables.ToArray());
            Console.ReadKey();
        }
    }
    public class Linear : Model
    {
        Layer dense;
        public Linear(ModelArgs args) : base(args)
        {
            dense = keras.layers.Dense(1, activation: null,
                kernel_initializer: tf.zeros_initializer, bias_initializer: tf.zeros_initializer);
            StackLayers(dense);
        }
        // Set forward pass
        protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
        {
            var outputs = dense.Apply(inputs);
            return outputs;
        }
    }
}
```

这里，我们没有显式地声明 `a` 和 `b` 两个变量并写出 `y_pred = a * X + b` 这一线性变换，而是建立了一个继承了 `tf.keras.Model` 的模型类 `Linear` 。这个类在初始化部分实例化了一个 **全连接层** （ `keras.layers.Dense` ），并在 call 方法中对这个层进行调用，实现了线性变换的计算。 



你可以通过扫二维码 或者 访问代码链接，进行该 “LinearRegression_CustomModel” 完整控制台程序代码的下载。



代码下载URL：

https://github.com/SciSharp/TensorFlow.NET-Tutorials/blob/master/PracticeCode/2.10%20Keras%20API/LinearRegression_CustomModel/LinearRegression/Program.cs



二维码链接：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608273760088.png" alt="1608273760088" style="zoom:80%;" />



通过运行代码，我们可以得到正确的 Loss 下降的过程数据，同时训练完成后，终端打印出了最终 dense 中的 kernel 和 bias 的值，如下所示：

```
step: 0,loss: 250
step: 1,loss: 2.4100037
step: 2,loss: 0.85786396
step: 3,loss: 0.8334713
step: 4,loss: 0.8188195
step: 5,loss: 0.80448306
step: 6,loss: 0.7903991
step: 7,loss: 0.7765587
step: 8,loss: 0.7629635
step: 9,loss: 0.74960434
step: 10,loss: 0.73648137
step: 11,loss: 0.7235875
step: 12,loss: 0.71091765
step: 13,loss: 0.69847053
step: 14,loss: 0.6862406
step: 15,loss: 0.67422575
step: 16,loss: 0.66242313
step: 17,loss: 0.65082455
step: 18,loss: 0.6394285
step: 19,loss: 0.62823385
[tf.Variable: 'kernel:0' shape=(3, 1), dtype=float32, numpy=[[0.8266835],
[1.2731746],
[1.7196654]], tf.Variable: 'bias:0' shape=(1), dtype=float32, numpy=[0.44649106]]
```



**模型（Model） 常用 API 概述**

**① 模型类**

```c#
Tensorflow.Keras.Engine.Model(ModelArgs args)
```

Tensorflow.Keras.Engine.Model() 为模型类，通过将网络层组合成为一个对象，实现训练和推理功能。模型类的参数为 ModelArgs 参数列表类，包含 Inputs (模型的输入)  和 Outputs (模型的输出) 。



**② 模型类的 summary 方法**

```c#
Model.summary(int line_length = -1, float[] positions = null)
```

打印网络模型的内容摘要，包含网络层结构和参数描述等。

一个简单的3层网络打印出的模型示例如下：

```
Model: "mnist_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 784)]             0         
_________________________________________________________________
dense (Dense)                (None, 64)                50240     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```

**参数**

- **line_length**：int 类型，打印的总行数，默认值 -1 代表打印所有内容；
- **positions**： float[] 类型，指定每一行中日志元素的相对或绝对位置 ，默认值 null ；



**③ Sequential 类**

```c#
Tensorflow.Keras.Engine.Sequential(SequentialArgs args)
```

Sequential 类继承 Model 类，将线性的网络层堆叠到一个模型中，并提供训练和推理的方法；



**④ Sequential 类的 add 方法**

```c#
Sequential.add(Layer layer)
```

在网络模型的顶部插入一个网络层；



**④ Sequential 类的 pop 方法**

```c#
Sequential.pop()
```

删除网络模型的最后一个网络层；



**接下来是模型(Model)训练相关的API：**



**⑥ compile 编译方法 **

```c#
Model.compile(ILossFunc loss, OptimizerV2 optimizer, string[] metrics)
```

配置模型的训练参数。

**参数**

- **optimizer**：优化器，参数类型为 优化器名称的字符串 或 优化器实例；
- **loss**：损失函数，参数类型为 损失函数体 或 `Tensorflow.Keras.Losses.Loss` 实例 ；
- **metrics**：模型训练和测试过程中，采用的评估指标列表，列表元素类型为 评估函数名称的字符串、评估函数体 或 `Tensorflow.Keras.Metrics.Metric` 实例；
- **loss_weights** ：可选参数，类型为 列表 或 字典，指定权重系数以加权模型的不同输出中的 Loss 贡献度；
- **weighted_metric**：可选参数，类型为 列表，在训练和测试期间要通过sample_weight或class_weight评估和加权的指标列表；
- **run_eagerly**：可选参数，类型为 bool布尔类型，默认为 False ，如果设置为 True ，则模型结构将不会被`tf.function` 修饰和调用；
-  **steps_per_execution**：可选参数，类型为 整数Int，默认值为 1，表示每个 `tf.function` 调用中要运行的批处理数。 



**⑦ fit 训练方法**

```c#
Model.fit(NDArray x, NDArray y,
            int batch_size = -1,
            int epochs = 1,
            int verbose = 1,
            float validation_split = 0f,
            bool shuffle = true,
            int initial_epoch = 0,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
```

以固定的轮数或迭代方式 训练模型。

**参数**

- **x**：输入数据，类型为 Numpy数组 或 Tensor；
- **y**：标签数据，类型和 x 保持一致；
- **batch_size**：批次大小，类型为 整数int，指定每次梯度更新时每个样本批次的数量；
- **epochs**：训练轮数，类型为 整数int，表示模型训练的迭代轮数；
- **verbose**：详细度，类型为 整数 0 ，1 或 2 ，设置终端输出信息的详细程度。0 为最简模式，1 为进度信息简单显示，2 为每个训练周期的数据详细显示；
- **validation_split**：验证集划分比，类型为 0~1之间的浮点型float，自动从训练数据集中划分出该比例的数据作为验证集。模型将剥离开训练数据的这一部分，不对其进行训练，并且将在每个轮次结束时评估此验证集数据的损失和其它模型评估函数；
- **shuffle**：数据集乱序，类型为 布尔bool类型，默认值为true，表示每次训练前都将该批次中的数据随机打乱；
- **initial_epoch**：周期起点，类型为 整数int，默认值为 0，表示训练开始的轮次起点（常用于重启恢复之前训练中的模型）；
- **max_queue_size**：队列最大容量，类型为整数int，默认值为 10，设置数据生成器队列的最大容量；
- **workers**：进程数，类型为整数int，默认值为 1。设置使用基于进程的线程时，要启动的最大进程数。如果未指定，`workers` 则默认为1。如果为0，将在主线程上执行数据生成器；
- **use_multiprocessing**：是否使用异步线程，类型为 布尔bool类型，默认为 false。如果为`True`，需要使用基于进程的线程；



**⑧ evaluate 评价方法**

```c#
Model.evaluate(NDArray x, NDArray y,
            int batch_size = -1,
            int verbose = 1,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false,
            bool return_dict = false)
```

返回测试集上评估的模型损失和精度值。

**参数**

- **x**：输入的测试数据，类型为 Numpy数组 或 Tensor；
- **y**：测试的标签数据，类型和 x 保持一致；
- **batch_size**：批次大小，类型为 整数int，指定每次计算时每个样本批次的数量；
- **verbose**：详细度，类型为 整数 0 或 1  ，设置终端输出信息的详细程度。0 为最简模式，1 为进度信息简单显示；
- **steps**：迭代步数，类型为 整数，默认值为 -1。设置评估周期内的迭代次数（样本批次数量）；
- **max_queue_size**：队列最大容量，类型为整数int，默认值为 10，设置数据生成器队列的最大容量；
- **workers**：进程数，类型为整数int，默认值为 1。设置使用基于进程的线程时，要启动的最大进程数。如果未指定，`workers` 则默认为1。如果为0，将在主线程上执行数据生成器；
- **use_multiprocessing**：是否使用异步线程，类型为 布尔bool类型，默认为 false。如果为`True`，需要使用基于进程的线程；
- **return_dict**：返回字典，类型为 布尔bool类型，默认为 false。如果为`True`，返回字典类型的 loss 和 metric 结果，字典的 key 为数据的名称；如果为`False`，则正常返回列表类型的结果；



**⑨ predict 预测方法**

```c#
Model.predict(Tensor x,
            int batch_size = 32,
            int verbose = 0,
            int steps = -1,
            int max_queue_size = 10,
            int workers = 1,
            bool use_multiprocessing = false)
```

生成输入样本的预测输出的结果。

**参数**

- **x**：输入的测试数据，类型为 Tensor；
- **batch_size**：每批样品数，类型为整数int，如果未指定，`batch_size`则默认为32，该参数指定了每次计算时每个样本批次的数量；
- **verbose**：详细度，类型为 整数 0 或 1  ，默认值为 0，该参数设置终端输出信息的详细程度。0 为最简模式，1 为进度信息简单显示；
- **steps**：迭代步数，类型为 整数，默认值为 -1。设置预测周期完成内的迭代次数（样本批次数量）；
- **max_queue_size**：队列最大容量，类型为整数int，默认值为 10，设置数据生成器队列的最大容量；
- **workers**：进程数，类型为整数int，默认值为 1。设置使用基于进程的线程时，要启动的最大进程数。如果未指定，`workers` 则默认为1。如果为0，将在主线程上执行数据生成器；
- **use_multiprocessing**：是否使用异步线程，类型为 布尔 bool 类型，默认为 false。如果为`True`，需要使用基于进程的线程；

**返回值**

返回 Tensor 类型的预测数组。



**最后一块是模型(Model)保存和载入相关的API：**



**⑩ save 模型保存方法**

```c#
Model.save(string filepath,
            bool overwrite = true,
            bool include_optimizer = true,
            string save_format = "tf",
            SaveOptions options = null)
```

 将模型保存到 Tensorflow 的 SavedModel 或单个 HDF5 文件。 

**参数**

- **filepath**：模型文件路径，类型为字符串，模型保存的 SavedModel 或 H5 文件的路径；
- **overwrite**：是否覆盖，类型为 布尔 bool，设置为 true 以默认方式覆盖目标位置上的任何现有文件，设置为 false 则向用户提供手动提示；
- **include_optimizer**：是否包含优化器信息，类型为 布尔 bool，如果为 True，则将 optimizer 优化器的状态保存在一起；
- **save_format**：保存格式，类型为字符串string，可选择 `"tf"` 或 `"h5"` ，指示是否将模型保存到 Tensorflow SavedModel 或 HDF5，在 TF 2.X 中默认为 “tf”；
- **SaveOptions**：保存到 SaveModel 的选项，类型为 `Tensorflow.ModelSaving.SaveOptions` 对象，用于指定保存到 SavedModel 的选项；



**⑪ load_model 载入模型功能**

//TODO

加载通过 `model.save()` 方法保存的模型。

https://keras.io/api/models/model_saving_apis/

//TODO

Tensorflow.keras.models.save_model 方法

[`get_weights` 方法](https://keras.io/api/models/model_saving_apis/#getweights-method)

[`set_weights` 方法](https://keras.io/api/models/model_saving_apis/#setweights-method)

[`save_weights` 方法](https://keras.io/api/models/model_saving_apis/#saveweights-method)

[`load_weights` 方法](https://keras.io/api/models/model_saving_apis/#loadweights-method)

[`get_config` 方法](https://keras.io/api/models/model_saving_apis/#getconfig-method)

[`from_config` 方法](https://keras.io/api/models/model_saving_apis/#fromconfig-method)

[`model_from_config` 功能](https://keras.io/api/models/model_saving_apis/#modelfromconfig-function)

[`to_json` 方法](https://keras.io/api/models/model_saving_apis/#tojson-method)

[`model_from_json` 功能](https://keras.io/api/models/model_saving_apis/#modelfromjson-function)

[`clone_model` 功能](https://keras.io/api/models/model_saving_apis/#clonemodel-function)





### 10.3 Keras 常用 API 说明

#### 10.3.1 回调函数 Callbacks API

keras 的回调函数是一个类，用在 Model.fit() 中作为参数，可以在训练的各个阶段（例如，在训练的开始或结束时，在单个 epoch 处理之前或之后等）执行一定的操作。

您可以使用回调来处理这些操作：

- 每个批次（batch）训练后写入 TensorBoard 日志以监控指标
- 定期将模型保存到本地文件
- 提前结束训练
- 训练期间查看模型的内部状态和统计信息
- ...（更多其它的功能）

大部分时候，keras.callbacks 子模块中定义的回调函数类已经足够使用了，如果有特殊的需要，我们也可以通过对 keras.callbacks.Callbacks 执行子类化构造自定义的回调函数。 



**基础用法**

你可以将回调列表作为 callbacks 参数传递给模型的 Model.fit() 方法，这样就可以在训练的每个阶段自动调用回调列表定义的相关方法，代码如下：

//TODO 下述代码为 Python，需要转换为 C#

```py
my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(dataset, epochs=10, callbacks=my_callbacks)
```



**已定义的回调类**

我们来简单看下 callbacks 子模块中已定义好的回调函数类，主要有下述几种类型：

- Base Callback class

  keras.callbacks.Callback() 是用于建立新回调的抽象基类，所有回调函数都继承于 keras.callbacks.Callbacks基类，拥有params和model这两个属性。其中 params 是一个dict，记录了训练相关参数 (例如 verbosity, batch size, number of epochs 等等)，model 即当前关联模型的引用；

- ModelCheckpoint

  该回调以特定频率保存 Keras 模型或者模型权重。 `ModelCheckpoint` 回调与 `model.fit()` 训练结合使用，用于以一定间隔（达到最佳性能或每个epoch结束时）保存模型或权重（在 checkpoint 文件中），这样就可以在后续加载模型或权重，以从保存的状态中恢复继续训练。

- TensorBoard

  TensorBoard 是 TensorFlow 自带的可视化工具，该回调可以为 Tensorboard 可视化保存日志信息。包括 Metrics 指标 summary 图、训练图的可视化、激活直方图、采样分析和模型参数可视化等；

- EarlyStopping

  当被监控指标在设定的若干个epoch后没有提升，则提前终止训练。例如，假设训练的目的是使损失（loss）最小化。这样，要监视的指标设置为 `'loss'`，模式设置为 `'min'`。每个 `model.fit()` 训练循环都会检查每个epoch结束时的损失是否不再递减，同时考虑到 `min_delta` 和 `patience`（如果适用的话），一旦发现它不再减少， `model.stop_training `则标记为“真”，训练提前终止；

- LearningRateScheduler

  学习率控制器。给定学习率 lr 和 epoch 的函数关系 schedule ，该回调会根据该函数关系在每个 epoch 前更新学习率， 并将更新后的学习率应用于优化器；

- ReduceLROnPlateau

  当设置的指标停止改善时，自动降低学习率。在常见的深度学习场景中，一旦发现学习停滞不再优化，通常可以将学习率降低2-10倍，以使模型继续优化训练。此回调自动进行这个过程，它监视训练的过程，如果没有发现持续 “ patience ” 轮数（ patience 为可设置的参数）的改善，则学习率会自动按照设定的因子降低；

- RemoteMonitor

  该回调用于将事件流传输到服务器端进行显示；

- LambdaCallback

  使用 callbacks.LambdaCallback 编写较为简单的自定义回调函数，该回调是使用匿名函数构造的；

- TerminateOnNaN

  该回调用于训练过程中遇到 loss 为 NaN 时自动终止训练；

- CSVLogger

  该回调将每次 epoch 后的 logs 结果以 streams 流格式记录到本地 CSV 文件； 

- ProgbarLogger

  将每个 epoch 结束后的 logs 结果打印到标准输出流中。



#### 10.3.2 数据集预处理 Dataset preprocessing

Keras 中的数据集预处理组件，位于 Tensorflow.Keras.Preprocessing 类中。主要作用是帮助你载入本地磁盘中的数据，转换成 `tf.data.Dataset` 的对象，以供模型训练使用。

常用的数据集预处理功能分为：图像数据预处理、时间数据预处理和文本数据预处理，我们来逐个看下各数据集种类的预处理的函数。



**图像数据集预处理**

常用函数为 image_dataset_from_directory ，实现从本地磁盘载入图像文件并生成一个 `tf.data.Dataset` 对象的功能。

```c#
IDatasetV2 image_dataset_from_directory(string directory,
            string labels = "inferred",
            string label_mode = "int",
            string[] class_names = null,
            string color_mode = "rgb",
            int batch_size = 32,
            TensorShape image_size = null,
            bool shuffle = true,
            int? seed = null,
            float validation_split = 0.2f,
            string subset = null,
            string interpolation = "bilinear",
            bool follow_links = false)
```

例如，假如你有2个文件夹，代表 class_a 和 class_b 这2种类别的图像，每个文件夹包含10,000个来自不同类别的图像文件，并且你想训练一个图像分类的模型，你的训练数据文件夹结构如下：

```c#
training_data/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
    ......
......a_image_N.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
    ......
......b_image_N.jpg
```

你可以直接调用函数 image_dataset_from_directory ，传入参数  `文件夹路径 directory` 和 `labels='inferred' ` ，该函数运行后返回 `tf.data.Dataset` 对象，实现从子目录`class_a`和`class_b`生成批次图像的功能，同时生成标签0和1（0对应于`class_a`和1对应于`class_b`）。 



参数说明：

- **directory**：数据所在的**目录**。如果 `labels` 被设置为 “inferred” ，则它应包含子目录，每个子目录都包含一个类的图像。否则，目录结构将被忽略。
- **labels**：  “inferred” ，图像关联的类别标签（标签是从目录结构生成的），或者是整数标签的列表/元组，其大小与目录中找到的图像文件数量相同。标签的编号是根据图像文件路径中的字母和数字的顺序进行排序。
- **label_mode**：“int”  表示标签被编码为整数（例如，用于`sparse_categorical_crossentropy` 类型的Loss）。“categorical” 是指标签被编码为分类矢量（例如，用于`categorical_crossentropy` 类型的 Loss）。“binary” 表示将标签（只能有2个）编码为 `float32` 标量，其值为0或1（例如表示`binary_crossentropy` 类型的 Loss）。“None”（无标签）。
- **class_names**：仅在 `labels` 被设置为 “inferred” 时有效。这是类别标签名称的明确列表（必须与子目录的名称匹配）。用于自定义控制类的顺序（否则使用字母数字顺序）。
- **color_mode**：“grayscale”，“ rgb”，“ rgba” 之一。默认值：“ rgb”。图像将被转换为具有1、3或4个通道。
- **batch_size**：数据批处理的大小。默认值：32
- **image_size**：从磁盘读取图像后将图像调整的大小。默认为`(256, 256)`。由于数据管道方式处理的批次图像必须全部具有相同大小，因此必须设置该图像的尺寸。
- **shuffle**：是否随机打乱数据。默认值：True。如果设置为False，则按字母数字顺序对数据进行排序。
- **seed**：用于随机排列和转换的随机种子。
- **validation_split**：设置介于0和1之间的浮点数，用于分割出一部分数据供验证集使用。
- **subset**：“ training ”或“ validation ”之一。仅在 `validation_split` 设置时使用。
- **interpolation**：字符串，调整图像大小时使用的图像插值算法。默认为 `bilinear`。支持 `bilinear`，`nearest`，`bicubic`， `area`，`lanczos3`，`lanczos5`，`gaussian`，`mitchellcubic` 。
- **follow_links**：是否访问符号链接指向的子目录。默认为False。



**时间数据集预处理**

//TODO `timeseries_dataset_from_array` 功能

https://keras.io/api/preprocessing/timeseries/#timeseries_dataset_from_array-function



**文本数据集预处理**

常用函数为 text_dataset_from_directory，实现从本地磁盘载入 txt 文本文件并生成一个 `tf.data.Dataset` 对象的功能。

```c#
IDatasetV2 text_dataset_from_directory(string directory,
            string labels = "inferred",
            string label_mode = "int",
            string[] class_names = null,
            int batch_size = 32,
            bool shuffle = true,
            int? seed = null,
            float validation_split = 0.2f,
            string subset = null)
```

例如，假如你有2个文件夹，代表 class_a 和 class_b 这2种类别的文本数据，每个文件夹包含10,000个来自不同类别的文本文件，并且你想训练一个文本分类的模型，你的训练数据文件夹结构如下：

```c#
training_data/
...class_a/
......a_text_1.txt
......a_text_2.txt
    ......
......a_text_N.txt
...class_b/
......b_text_1.txt
......b_text_2.txt
    ......
......b_text_N.txt
```

你可以直接调用函数 text_dataset_from_directory ，传入参数  `文件夹路径 directory` 和 `labels='inferred' ` ，该函数运行后返回 `tf.data.Dataset` 对象，实现从子目录`class_a`和`class_b`生成批次文本的功能，同时生成标签0和1（0对应于`class_a`和1对应于`class_b`）。 



参数说明：

- **directory**：数据所在的**目录**。如果 `labels` 被设置为 “inferred” ，则它应包含子目录，每个子目录都包含一个类的文本文件。否则，目录结构将被忽略。
- **labels**：  “inferred” ，文本文件关联的类别标签（标签是从目录结构生成的），或者是整数标签的列表/元组，其大小与目录中找到的文本文件数量相同。标签的编号是根据文本文件路径中的字母和数字的顺序进行排序。
- **label_mode**：“int”  表示标签被编码为整数（例如，用于 `sparse_categorical_crossentropy` 类型的Loss）。“categorical” 是指标签被编码为分类矢量（例如，用于 `categorical_crossentropy` 类型的 Loss）。“binary” 表示将标签（只能有2个）编码为 `float32` 标量，其值为0或1（例如表示 `binary_crossentropy` 类型的 Loss）。“None”（无标签）。
- **class_names**：仅在 `labels` 被设置为 “inferred” 时有效。这是类别标签名称的明确列表（必须与子目录的名称匹配）。用于自定义控制类的顺序（否则使用字母数字顺序）。
- **batch_size**：数据批处理的大小。默认值：32
- **//TODO max_length**：文本字符串的最大大小。超过此长度的文本将被截断为 `max_length`。
- **shuffle**：是否随机打乱数据。默认值：True。如果设置为False，则按字母数字顺序对数据进行排序。
- **seed**：用于随机排列和转换的随机种子。
- **validation_split**：设置介于0和1之间的浮点数，用于分割出一部分数据供验证集使用。
- **subset**：“ training ”或“ validation ”之一。仅在 `validation_split` 设置时使用。





#### 10.3.3 优化器 Optimizers

在机器学习中，模型的优化算法可能会直接影响最终生成模型的性能。有时候效果不好，未必是特征数据的问题或者模型结构设计的问题，很可能就是优化算法的问题。

深度学习优化算法大概经历了 SGD -> SGDM -> NAG ->Adagrad -> Adadelta(RMSprop) -> Adam -> Nadam 这样的发展历程。其中，SGD 是最基础的入门级算法，也一直被学术界所推崇，而 Adam 和 Nadam 是目前最主流、最易使用的优化算法，收敛速度和效果也都不错，非常适合新手直接上手使用。

在 Keras 中的优化器 Optimizers 是搭配 compile() 和 fit() 这2个方法使用的，是模型训练必须要设置的两个参数之一（另外一个是 Losses）。



你可以先实例化 Optimizers 作为参数传给 model.compile() ，或者直接通过字符串标识符传递给 optimizer 参数，后一种方式将直接使用优化器的默认参数。示例如下：

```c#
// method 1# : compile(ILossFunc loss, OptimizerV2 optimizer, string[] metrics)
            var layers = keras.layers;
            model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });
            var optimizer = keras.optimizers.Adam(learning_rate : 0.01f);
            model.compile(optimizer: optimizer,
                loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                metrics: new[] { "accuracy" });


// method 2# : compile(string optimizer, string loss, string[] metrics)
            var layers = keras.layers;
            model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });
            model.compile("adam", "sparse_categorical_crossentropy", metrics: new[] { "accuracy" });
```



如果你不使用 compile() 方法，而是通过自定义方式编写训练循环，你也可以通过 tf.GradientTape() 来检索梯度，并调用 optimizer.apply_gradients() 方法实现权重的更新。示例如下：

```c#
var optimizer = keras.optimizers.SGD(learning_rate);

// Run training for the given number of steps.
foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
{
    // Wrap computation inside a GradientTape for automatic differentiation.
    using var g = tf.GradientTape();
    // Forward pass.
    var pred = neural_net.Apply(batch_x, is_training: true);
    var loss = cross_entropy_loss(pred, batch_y);

    // Compute gradients.
    var gradients = g.gradient(loss, neural_net.trainable_variables);

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables.Select(x => x as ResourceVariable)));
}

// Test model on validation set.
{
    var pred = neural_net.Apply(x_test, is_training: false);
    this.accuracy = (float)accuracy(pred, y_test);
    print($"Test Accuracy: {this.accuracy}");
}
```



//TODO 在深度学习中，学习率这一超参数 随着训练的深入迭代，可以逐渐衰减，这个可以更好地适应梯度下降的曲线（谷底区域 Loss 的下降逐渐放缓），往往可以取得更好的效果。在 Keras 中为了方便调试这类常见的情况，可以专门设置 学习率衰减规划，通过 keras.optimizers.schedules.ExponentialDecay() 方法实现学习率的动态规划。示例如下：//TODO 下述代码为Python，需要转换为 C#

```c#
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
```



**Keras 中可以使用的优化器如下：**

- SGD

  默认参数时为纯 SGD，设置 momentum 参数不为 0 时效果变成 SGDM，考虑了一阶动量。设置 nesterov 为 True 后效果变成 NAG，即 Nesterov Accelerated Gradient，在可以理解为在标准动量方法中添加了一个校正因子，以提前计算下一步的梯度来指导当前梯度。 

- RMSprop

  考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对 Adagrad 进行了优化，通过指数平滑的值实现只考虑一定窗口区间内的二阶动量。

- Adam

  同时考虑了一阶动量和二阶动量，可以看成 RMSprop 上进一步考虑了一阶动量。

- Adadelta

  考虑了二阶动量，与RMSprop类似，但是更加复杂一些，自适应性更强。

- Adagrad

  考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率。缺点是学习率单调下降，可能后期学习速率过慢会导致提前停止学习。

- Adamax

   这是基于无穷范数的 Adam 的变体 ，Adamax 有时候性能优于 Adam，特别是针对带有嵌入层 embeddings 的模型。

- Nadam

  在 Adam 基础上进一步考虑了 Nesterov Acceleration。

- Ftrl

  实现 FTRL （Follow-the-regularized-Leader）算法的优化器，同时支持 在线L2正则项 和 特征缩减L2正则项，广泛适用于在线学习（ Online Learning ）这类训练方式。





#### 10.3.4 损失函数 Losses

损失函数是模型在训练过程中不断优化降低其值的对象。

在 Keras 中的损失函数 Losses 是搭配 compile() 和 fit() 这2个方法使用的，是模型训练必须要设置的两个参数之一（另外一个是 Optimizers）。

对于回归模型，通常使用的损失函数是均方损失函数 mean_squared_error；对于二分类模型，通常使用的是二元交叉熵损失函数 binary_crossentropy；对于多分类模型，如果 label 是 one-hot 编码的，则使用类别交叉熵损失函数 categorical_crossentropy，如果 label 是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 sparse_categorical_crossentropy。



你可以先实例化 losses 作为参数传给 model.compile() ，或者直接通过字符串标识符传递给 loss 参数，后一种方式将直接使用损失函数的默认参数。示例如下：

```c#
// method 1# : compile(ILossFunc loss, OptimizerV2 optimizer, string[] metrics)
            var layers = keras.layers;
            model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });
            var loss = keras.losses.SparseCategoricalCrossentropy(from_logits: true);
            model.compile(optimizer: keras.optimizers.Adam(learning_rate : 0.01f),
                loss: loss,
                metrics: new[] { "accuracy" });


// method 2# : compile(string optimizer, string loss, string[] metrics)
            var layers = keras.layers;
            model = keras.Sequential(new List<ILayer>
            {
                layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
                layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation: keras.activations.Relu),
                layers.Dense(num_classes)
            });
            model.compile("adam", "sparse_categorical_crossentropy", metrics: new[] { "accuracy" });
```





**Keras 中的损失函数如下：**

**概率损失 Probabilistic losses**

- BinaryCrossentropy

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation.svg" alt="img" style="zoom:80%;" /> 

  二元交叉熵，针对的是二分类问题。

- CategoricalCrossentropy 

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418141984.svg" alt="img" style="zoom:80%;" /> 

  多类别交叉熵损失，针对的是多分类问题，真实值需要 one-hot 编码格式处理。

- SparseCategoricalCrossentropy 

  稀疏多类别交叉熵损失，原理 CategoricalCrossentropy 一样，不过直接支持整数编码类型的真实值。

- Poisson

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418176051.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = y_pred - y_true * log(y_pred)
  ```

  泊松损失，目标值为泊松分布的负对数似然损失 。

- KLDivergence

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418445938.svg" alt="img" style="zoom:80%;" /> 

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418457117.svg" alt="img" style="zoom:80%;" /> 

  Kullback-Leibler Divergence，KLD，KL散度，也叫做相对熵（Relative Entropy），它衡量的是相同事件空间里的两个概率分布的差异情况。

**回归损失 Regression losses**

- MeanSquaredError

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418494152.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = square(y_true - y_pred)
  ```

  均方差损失，MSE。

- MeanAbsoluteError

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418611895.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = abs(y_true - y_pred)
  ```

  平均绝对值误差，MAE。

- MeanAbsolutePercentageError

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418653153.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = 100 * abs(y_true - y_pred) / y_true
  ```

  平均绝对百分比误差，MAPE，注意分母不要除0。

- MeanSquaredLogarithmicError

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610418695237.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = square(log(y_true + 1.) - log(y_pred + 1.))
  ```

   MSLE，均方对数误差。

- CosineSimilarity

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420195932.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
  ```

  计算余弦相似度。

- Huber

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420269513.svg" alt="img" style="zoom:80%;" /> 

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420279413.svg" alt="img" style="zoom:80%;" /> 

  Huber Loss 是一个用于回归问题的带参损失函数, 优点是能增强平方误差损失函数对离群点的鲁棒性。当预测偏差小于 δ 时，它采用平方误差,当预测偏差大于 δ 时，采用的线性误差。相比于均方误差，HuberLoss降低了对离群点的惩罚程度，所以 HuberLoss 是一种常用的鲁棒的回归损失函数。 

- LogCosh

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420446722.svg" alt="img" style="zoom:80%;" /> 

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420456714.svg" alt="img" style="zoom:80%;" /> 

  ```
  logcosh = log((exp(x) + exp(-x))/2)，其中 x = y_pred - y_true。
  ```

  LogCosh是预测误差的双曲余弦的对数， 它比L2损失更平滑，与均方误差大致相同，但是不会受到偶尔疯狂的错误预测的强烈影响 

**最大间隔 (常见SVM) Hinge损失 Hinge losses for "maximum-margin" classification**

- Hinge

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420521133.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = maximum(1 - y_true * y_pred, 0)
  ```

  Hinge 损失常用于二分类。

- SquaredHinge

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/equation-1610420875670.svg" alt="img" style="zoom:80%;" /> 

  ```
  loss = square(maximum(1 - y_true * y_pred, 0))
  ```

  Hinge 损失的平方。

- CategoricalHinge

  ```
  loss = maximum(neg - pos + 1, 0) , neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)
  ```

  Hinge 损失的多类别形式。 





你也可以独立使用损失函数进行运算，通过调用 Tensorflow.Keras.Losses 的基类 ILossFunc 的 Call() 实现，方法如下：

```c#
Tensor Call(Tensor y_true, Tensor y_pred, Tensor sample_weight = null)
```

参数说明：

- y_true：真实值，张量类型，shape 为  (batch_size, d0, ... dN) ；
- y_pred：预测值，张量类型，shape 为  (batch_size, d0, ... dN) ；
- sample_weight： 每个样本的损失的缩放权重系数。如果提供了标量，则损失将简单地按照该给定值缩放；如果 `sample_weight` 是向量 `[batch_size]`，则批次中每个样本的总损失将由 `sample_weight` 向量中的相应元素缩放；如果 `sample_weight` 的形状为`(batch_size, d0, ... dN)`（或可以广播为该形状），则每个`y_pred` 的损失值均按 `sample_weight `的相应值进行缩放。 

返回值：

默认情况下，损失函数会为每个输入样本返回一个标量损失值。





损失类的实例化过程也可以传入构造函数参数，我们最后来看下构造函数的结构：

```c#
public Loss(string reduction = ReductionV2.AUTO, 
            string name = null,
            bool from_logits = false)
```

参数说明：

- reduction：默认  `"sum_over_batch_size"`（即平均值）。可枚举值为 “ sum_over_batch_size”，“ sum” 和 “ none”，区别如下：
  - “ sum_over_batch_size”：表示损失实例将返回批次中每个样本损失的平均值；
  - “ sum”：表示损失实例将返回批次中每个样本损失的总和；
  - “none”：表示损失实例将返回每个样本损失的完整数组；
- name：损失函数的自定义名称；
- from_logits：代表是否有经过Logistic函数，常见的 Logistic 函数包括 Sigmoid、Softmax 函数。





#### 10.3.5 评估指标 Metrics

最后，我们来看下 评估指标 Metrics，评估指标是判断模型性能的函数。

上一小节，我们了解了 损失函数 Losses。我们知道，损失函数除了作为模型训练时的优化目标（ Objective = Loss + Regularizatio ），也能够作为模型好坏的一种评价指标。但通常人们还会从其它角度评估模型的好坏，这就是评估指标。

大部分的损失函数都可以作为评估指标，但评估指标不一定可以作为损失函数，例如 AUC , Accuracy , Precision。这是因为训练模型的过程中不使用评估指标 Metrics 度量的结果，因此评估指标不要求连续可导，而损失函数是参与模型训练过程的，通常要求连续可导。



评估指标 Metrics 是 compile() 方法的可选参数，在模型编译时，可以通过列表形式指定多个评估指标。示例代码如下：

```c#
var layers = keras.layers;
model = keras.Sequential(new List<ILayer>
      {
            layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
            layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation: keras.activations.Relu),
            layers.Dense(num_classes)
       });

model.compile(optimizer: keras.optimizers.Adam(),
              loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
              metrics: new[] { "accuracy", "MeanSquaredError", "AUC" });

model.summary();
```



**Keras 中的评估指标如下：**

（注：评估指标较多，且和损失函数部分相似，这里只对部分进行说明。）

**准确性指标 Accuracy metrics**

- Accuracy

  准确性，计算预测与真实标签相同的频率；

- BinaryAccuracy

  二进制分类准确性，计算预测与真实二进制标签匹配的频率 ；

- CategoricalAccuracy

  多分类准确性，计算预测与真实 one-hot 标签匹配的频率 ；

- TopKCategoricalAccuracy

  多分类 TopK 准确率，计算目标在最高 `K` 个预测中的频率，要求 y_true(label) 为 one-hot 编码形式；

- SparseTopKCategoricalAccuracy

  稀疏多分类 TopK 准确率，计算整数值的目标在最高 `K` 个预测中的频率，要求 y_true(label) 为整数序号编码形式；

**概率指标 Probabilistic metrics**

（与损失函数 Loss 类似）

- BinaryCrossentropy
- CategoricalCrossentropy
- SparseCategoricalCrossentropy
- KLDivergence
- Poisson

**回归指标 Regression metrics**

（与损失函数 Loss 类似）

- MeanSquaredError

- RootMeanSquaredError

   <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/20190714113817886.png" alt="img" style="zoom: 67%;" /> 

  均方根误差指标，RMES；

- MeanAbsoluteError

- MeanAbsolutePercentageError

- MeanSquaredLogarithmicError

- CosineSimilarity

- LogCoshError

**二元分类指标 Classification metrics based on True/False positives & negatives**

- AUC

  通过黎曼求和公式求出近似的AUC（ ROC曲线 “TPR vs FPR” 下的面积 ），用于二分类，可直观解释为随机抽取一个正样本和一个负样本，正样本的预测值大于负样本的概率；

- Precision

  精确率，用于二分类，Precision = TP/(TP+FP)；

- Recall

  召回率，用于二分类，Recall = TP/(TP+FN)；

- TruePositives

  真正例，用于二分类；

- TrueNegatives

  真负例，用于二分类；

- FalsePositives

  假正例，用于二分类；

- FalseNegatives

  假负例，用于二分类；

- PrecisionAtRecall

  在召回率大于等于指定值的情况下计算最佳精度；

- SensitivityAtSpecificity

  当特异性 > = 指定值时，计算特定的最佳灵敏度；

- SpecificityAtSensitivity

  当灵敏度> =指定值时，计算特定的最佳特异性；

**图像分割指标  Image segmentation metrics**

- MeanIoU

  Mean Intersection-Over-Union 是语义图像分割的常用评估指标，它首先计算每个语义类的IOU，然后计算所有种类的平均值。IOU的定义如下：IOU = true_positive /（true_positive + false_positive + false_negative）；

**最大间隔 (常见SVM) Hinge指标 Hinge metrics for "maximum-margin" classification**

（与损失函数 Loss 类似）

- Hinge
- SquaredHinge
- CategoricalHinge



以上就是 Keras 中内置的评估指标，如果需要自定义评估指标，你可以对 Tensorflow.Keras.Metrics.Metric 进行子类化，重写构造函数、add_weight 方法、update_state 方法和 result 方法实现评估指标的计算逻辑，从而得到评估指标的自定义类的实现形式。 





### 10.4 Keras 建立模型的3种方式

在 Keras 中玩转神经网络模型一般有以下流程（5步走）：

- 准备训练数据（载入数据、数据预处理、生成批次）
- 搭建神经网络模型
- 配置训练过程和模型编译
- 训练模型
- 评估模型

如图所示：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1610502638177.png" alt="1610502638177" style="zoom: 67%;" />





可以使用以下3种方式构建模型：① 使用 Sequential 按层堆叠方式构建模型；② 使用函数式 API 构建任意结构模型；③ 继承 Model 基类构建完全自定义的模型。

对于结构相对简单和典型的神经网络（比如 MLP 和 CNN），并使用常规的手段进行训练 ，优先使用Sequential方法构建。

如果模型有多输入或者多输出，或者模型需要共享权重，或者模型具有残差连接等非顺序结构，推荐使用函数式API进行创建。这是一个易于使用的，全功能的API，支持任意模型的架构，对于大多数人和大多数用例都是足够的。

如果上述2种方式还无法满足模型需求，可以对 `Tensorflow.keras.Model` 类进行扩展以定义自己的新模型，同时手动编写了训练和评估模型的流程。这种模型子类化方式灵活度高，你可以从头开始实施所有操作，且与其他流行的深度学习框架共通，适合学术研究领域的模型开发探索。

具体3种方式的案例可以参考 “二、TensorFlow.NET API-8. 深度神经网络(DNN)入门”，链接如下：

https://github.com/SciSharp/TensorFlow.NET-Tutorials/blob/master/%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.md



接下来，我们通过示例代码片段，简单了解下 Keras 中的3种构建模型的方式。



#### 10.4.1 Sequential API

Sequential 模型适用于 **简单的层堆叠，** 其中每一层都有 **一个输入张量和一个输出张量**。

有2种方式可以将神经网络结构按特定顺序叠加起来，一种是直接提供一个层的列表，另外一种是通过层的 add 方式在末尾逐层添加。



**① layers 列表方式**

```c#
int num_classes = 5;
var layers = keras.layers;

model = keras.Sequential(new List<ILayer>
{
    layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)),
    layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation: keras.activations.Relu),
    layers.Dense(num_classes)
});
```



**② layers.add() 方式**

```c#
int num_classes = 5;
var layers = keras.layers;

model = keras.Sequential();
model.Layers.Add(layers.Rescaling(1.0f / 255, input_shape: (img_dim.dims[0], img_dim.dims[1], 3)));
model.Layers.Add(layers.Conv2D(16, 3, padding: "same", activation: keras.activations.Relu));
model.Layers.Add(layers.MaxPooling2D());
model.Layers.Add(layers.Flatten());
model.Layers.Add(layers.Dense(128, activation: keras.activations.Relu));
model.Layers.Add(layers.Dense(num_classes));
```





#### 10.4.2 Functional API

Keras Functional API 是一种创建比 Sequential API 更灵活的模型的方法。Functional API 可以处理具有非线性拓扑，共享层甚至多个输入或输出的模型。这种方式主要思想是，深度学习模型通常是层的有向无环图（DAG）。



考虑以下模型：

```
(input: 784-dimensional vectors)
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (64 units, relu activation)]
       ↧
[Dense (10 units, softmax activation)]
       ↧
(output: logits of a probability distribution over 10 classes)
```

这是一个基础的具有三层结构的网络。要使用 Functional API 方式构建此模型，我们先创建了一个输入节点 inputs 喂给模型，然后在此 inputs 对象上调用一个图层 outputs 节点，并依次添加剩余两个图层，最后把 输入 inputs 和 输出 outputs 传递给 keras.Model 的参数，创建模型。

示例代码片段如下：

```c#
// input layer
var inputs = keras.Input(shape: 784);

// 1st dense layer
var outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);

// 2nd dense layer
outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);

// output layer
outputs = layers.Dense(10).Apply(outputs);

// build keras model
model = keras.Model(inputs, outputs, name: "mnist_model");
```

我们可以通过 model.summary() 方法打印模型的摘要：

```c#
model.summary();
```

输出如下：

```
Model: "mnist_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 784)]             0         
_________________________________________________________________
dense (Dense)                (None, 64)                50240     
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 55,050
Trainable params: 55,050
Non-trainable params: 0
_________________________________________________________________
```





#### 10.4.3 Model Subclassing 

如果现有的这些层无法满足我的要求，我们不仅可以继承 Tensorflow.keras.Model 编写自己的模型类，也可以继承 Tensorflow.keras.layers.Layer 编写自己的层。Layer 类是 Keras 中的核心抽象之一，Layer 封装了状态（层的“权重”）和从输入到输出的转换（“Call”，即层的前向传递）。 

我们来用代码创建一个卷积神经网络，并自定义一个网络的参数为分类标签的种类数量。

首先，必要的引用添加如下：

```c#
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
```

接下来，子类化创建自定义CNN模型，参数为标签种类数，层使用 keras 中预定义的层（你也可以子类化 Layer 创建自定义的层）。我们在模型的构造函数中创建层，并在 Call() 方法中配置层的前向传播的输入输出顺序，模型类的代码如下：

```c#
public class ConvNet : Model
{
    Layer conv1;
    Layer maxpool1;
    Layer conv2;
    Layer maxpool2;
    Layer flatten;
    Layer fc1;
    Layer dropout;
    Layer output;

    public ConvNet(ConvNetArgs args)
        : base(args)
    {
        var layers = keras.layers;

        // Convolution Layer with 32 filters and a kernel size of 5.
        conv1 = layers.Conv2D(32, kernel_size: 5, activation: keras.activations.Relu);

        // Max Pooling (down-sampling) with kernel size of 2 and strides of 2.
        maxpool1 = layers.MaxPooling2D(2, strides: 2);

        // Convolution Layer with 64 filters and a kernel size of 3.
        conv2 = layers.Conv2D(64, kernel_size: 3, activation: keras.activations.Relu);
        // Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        maxpool2 = layers.MaxPooling2D(2, strides: 2);

        // Flatten the data to a 1-D vector for the fully connected layer.
        flatten = layers.Flatten();

        // Fully connected layer.
        fc1 = layers.Dense(1024);
        // Apply Dropout (if is_training is False, dropout is not applied).
        dropout = layers.Dropout(rate: 0.5f);

        // Output layer, class prediction.
        output = layers.Dense(args.NumClasses);

        StackLayers(conv1, maxpool1, conv2, maxpool2, flatten, fc1, dropout, output);
    }

    // Set forward pass.
    protected override Tensors Call(Tensors inputs, Tensor state = null, bool is_training = false)
    {
        inputs = tf.reshape(inputs, (-1, 28, 28, 1));
        inputs = conv1.Apply(inputs);
        inputs = maxpool1.Apply(inputs);
        inputs = conv2.Apply(inputs);
        inputs = maxpool2.Apply(inputs);
        inputs = flatten.Apply(inputs);
        inputs = fc1.Apply(inputs);
        inputs = dropout.Apply(inputs, is_training: is_training);
        inputs = output.Apply(inputs);

        if (!is_training)
            inputs = tf.nn.softmax(inputs);

        return inputs;
    }
}

public class ConvNetArgs : ModelArgs
{
    public int NumClasses { get; set; }
}
```

最后，我们输入训练数据，编译模型，利用梯度优化（optimizer.apply_gradients）迭代训练模型，训练完成后，输入测试数据集对模型效果进行评估。代码如下：

```c#
// MNIST dataset parameters.
int num_classes = 10;

// Training parameters.
float learning_rate = 0.001f;
int training_steps = 100;
int batch_size = 128;
int display_step = 10;
float accuracy_test = 0.0f;

IDatasetV2 train_data;
NDArray x_test, y_test, x_train, y_train;

public override void PrepareData()
{
    ((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data();
    // Convert to float32.
    // (x_train, x_test) = (np.array(x_train, np.float32), np.array(x_test, np.float32));
    // Normalize images value from [0, 255] to [0, 1].
    (x_train, x_test) = (x_train / 255.0f, x_test / 255.0f);

    train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
    train_data = train_data.repeat()
        .shuffle(5000)
        .batch(batch_size)
        .prefetch(1)
        .take(training_steps);
}

public void Run()
{
    tf.enable_eager_execution();

    PrepareData();

    // Build neural network model.
    var conv_net = new ConvNet(new ConvNetArgs
                               {
                                   NumClasses = num_classes
                               });

    // ADAM optimizer. 
    var optimizer = keras.optimizers.Adam(learning_rate);

    // Run training for the given number of steps.
    foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
    {
        // Run the optimization to update W and b values.
        run_optimization(conv_net, optimizer, batch_x, batch_y);

        if (step % display_step == 0)
        {
            var pred = conv_net.Apply(batch_x);
            var loss = cross_entropy_loss(pred, batch_y);
            var acc = accuracy(pred, batch_y);
            print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
        }
    }

    // Test model on validation set.
    {
        var pred = conv_net.Apply(x_test);
        accuracy_test = (float)accuracy(pred, y_test);
        print($"Test Accuracy: {accuracy_test}");
    }
}

void run_optimization(ConvNet conv_net, OptimizerV2 optimizer, Tensor x, Tensor y)
{
    using var g = tf.GradientTape();
    var pred = conv_net.Apply(x, is_training: true);
    var loss = cross_entropy_loss(pred, y);

    // Compute gradients.
    var gradients = g.gradient(loss, conv_net.trainable_variables);

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, conv_net.trainable_variables.Select(x => x as ResourceVariable)));
}

Tensor cross_entropy_loss(Tensor x, Tensor y)
{
    // Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64);
    // Apply softmax to logits and compute cross-entropy.
    var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
    // Average loss across the batch.
    return tf.reduce_mean(loss);
}

Tensor accuracy(Tensor y_pred, Tensor y_true)
{
    // # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
}
```





###  10.5 模型训练

模型建立完成后，TensorFlow 也内置了2种模型的训练方法：Keras 的 model.fit() 和 TensorFlow 的 optimizer.apply_gradients() ，我们简单地分别介绍一下。



**① 内置 fit 方法**

Keras 中的 model 内置了模型训练、评估和预测方法，分别为 model.fit() 、model.evaluate() 和 model.predict() 。

其中 fit 方法用来训练模型，该方法功能非常强大, 支持对 numpy array 、 Tensorflow.data.Dataset 和 Tensorflow.Keras.Datasets 数据进行训练，并且可以通过设置回调函数实现对训练过程的复杂控制逻辑。

我们通过 Functional API 方式来演示 MNIST数据集使用 fit 训练模型的方法。

第一步，引用类库。

```c#
using NumSharp;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
```

第二步，准备数据集。

```c#
NDArray x_train, y_train, x_test, y_test;

public void PrepareData()
{
    (x_train, y_train, x_test, y_test) = keras.datasets.mnist.load_data();
    x_train = x_train.reshape(60000, 784) / 255f;
    x_test = x_test.reshape(10000, 784) / 255f;
}
```

第三步，构建并编译模型。

```c#
Model model;
public void BuildModel()
{
    // input layer
    var inputs = keras.Input(shape: 784);

    // 1st dense layer
    var outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(inputs);

    // 2nd dense layer
    outputs = layers.Dense(64, activation: keras.activations.Relu).Apply(outputs);

    // output layer
    outputs = layers.Dense(10).Apply(outputs);

    // build keras model
    model = keras.Model(inputs, outputs, name: "mnist_model");
    // show model summary
    model.summary();

    // compile keras model into tensorflow's static graph
    model.compile(loss: keras.losses.SparseCategoricalCrossentropy(from_logits: true),
                  optimizer: keras.optimizers.RMSprop(),
                  metrics: new[] { "accuracy" });
}
```

最后一步，训练并评估模型（代码同时演示了模型的保存本地和本地载入）。

```c#
public override void Train()
{

    // train model by feeding data and labels.
    model.fit(x_train, y_train, batch_size: 64, epochs: 2, validation_split: 0.2f);

    // evluate the model
    model.evaluate(x_test, y_test, verbose: 2);

    // save and serialize model
    model.save("mnist_model");

    // recreate the exact same model purely from the file:
    // model = keras.models.load_model("path_to_my_model");
}
```



**② 利用梯度优化器 从头开始编写训练循环**

自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播进行参数迭代，拥有最高的灵活性。

代码参考本章上述 “10.4.3 Model Subclassing ” 一节，我们对其中的训练代码部分进行一下说明。

我们搭建一个3层的全连接神经网络（子类方式），网络层参数如下：

```c#
var neural_net = new NeuralNet(new NeuralNetArgs
{
    NumClasses = num_classes,
    NeuronOfHidden1 = 128,
    Activation1 = keras.activations.Relu,
    NeuronOfHidden2 = 256,
    Activation2 = keras.activations.Relu
});
```

模型中的变量和超参数如下：

```c#
// MNIST dataset parameters.
int num_classes = 10; // 0 to 9 digits
int num_features = 784; // 28*28

// Training parameters.
float learning_rate = 0.1f;
int display_step = 100;
int batch_size = 256;
int training_steps = 1000;

float accuracy;
IDatasetV2 train_data;
NDArray x_test, y_test, x_train, y_train;
```

然后，我们自定义交叉熵损失函数和准确度评估指标：

```c#
// Cross-Entropy Loss.
// Note that this will apply 'softmax' to the logits.
Func<Tensor, Tensor, Tensor> cross_entropy_loss = (x, y) =>
{
    // Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64);
    // Apply softmax to logits and compute cross-entropy.
    var loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels: y, logits: x);
    // Average loss across the batch.
    return tf.reduce_mean(loss);
};

// Accuracy metric.
Func<Tensor, Tensor, Tensor> accuracy = (y_pred, y_true) =>
{
    // Predicted class is the index of highest score in prediction vector (i.e. argmax).
    var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
};
```

接着，我们搭建随机梯度下降优化器，并应用梯度优化器于模型：

```c#
// Stochastic gradient descent optimizer.
var optimizer = keras.optimizers.SGD(learning_rate);

// Optimization process.
Action<Tensor, Tensor> run_optimization = (x, y) =>
{
    // Wrap computation inside a GradientTape for automatic differentiation.
    using var g = tf.GradientTape();
    // Forward pass.
    var pred = neural_net.Apply(x, is_training: true);
    var loss = cross_entropy_loss(pred, y);

    // Compute gradients.
    var gradients = g.gradient(loss, neural_net.trainable_variables);

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, neural_net.trainable_variables.Select(x => x as ResourceVariable)));
};
```

最后，使用自定义循环迭代进行模型训练，训练后的模型手动进行准确率评估：

```c#
// Run training for the given number of steps.
foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
{
    // Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y);

    if (step % display_step == 0)
    {
        var pred = neural_net.Apply(batch_x, is_training: true);
        var loss = cross_entropy_loss(pred, batch_y);
        var acc = accuracy(pred, batch_y);
        print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
    }
}

// Test model on validation set.
{
    var pred = neural_net.Apply(x_test, is_training: false);
    this.accuracy = (float)accuracy(pred, y_test);
    print($"Test Accuracy: {this.accuracy}");
}
```





至此，Keras 的模型搭建和模型训练就全部完成了，Keras 官方还有大量的完整解决方案的代码示例，涵盖计算机视觉、自然语言处理、数据结构化、推荐系统、时间序列、音频数据识别、Gan深度学习、强化学习和快速 Keras 模型等领域。

同时，你也可以直接调用 Keras 中的模型存储库 Keras Applications，其内置了 VGG、ResNet、InceptionNet、MobileNet 和 DenseNet 等大量成熟的深度学习模型，进行训练、直接推理使用或应用迁移学习。

因此，Keras 的简单、快速而不失灵活性的特点，使其成为 TensorFlow 推荐广大开发者使用，并且得到 TensorFlow 的官方内置和全面支持的核心 API 。



