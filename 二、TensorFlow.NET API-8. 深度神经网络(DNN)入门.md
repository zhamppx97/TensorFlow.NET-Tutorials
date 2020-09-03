# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 8. 深度神经网络(DNN)入门

### 8.1 深度神经网络(DNN)介绍

2006年， 深度学习鼻祖Hinton在《SCIENCE 》上发表了一篇论文” [Reducing the Dimensionality of Data with Neural Networks](http://www.cs.toronto.edu/~hinton/science.pdf) “，这篇论文揭开了深度学习的序幕。 这篇论文提出了两个主要观点：（1）、多层人工神经网络模型有很强的特征学习能力，深度学习模型学习得到的特征数据对原数据有更本质的代表性，这将大大便于分类和可视化问题；（2）、对于深度神经网络很难训练达到最优的问题，可以采用逐层训练方法解决，将上层训练好的结果作为下层训练过程中的初始化参数。 

深度神经网络(Deep Neural Networks，简称DNN)是深度学习的基础，想要学好深度学习，首先我们要理解DNN模型。 



**DNN模型结构：**

深度神经网络（Deep Neural Networks，DNN）可以理解为有很多隐藏层的神经网络，又被称为深度前馈网络（DFN），多层感知机（Multi-Layer Perceptron，MLP），其具有多层的网络结构，如下图所示： 

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1598604214665.png" alt="1598604214665" style="zoom:50%;" />

DNN模型按照层的位置不同，可以分为3种神经网络层：输入层、隐藏层和输出层。一般来说，第一层为输入层，最后一层为输出层，中间为单个或多个隐藏层（某些模型的结构可能不同）。

层与层之间是**全连接的**，也就是说，第i层的任意一个神经元一定与第i+1层的任意一个神经元相连。虽然DNN看起来很复杂，但是从小的局部结构来看，它还是和普通的感知机一样，即一个线性函数搭配一个激活函数。 



**DNN前向传播：**

 利用若干个权重系数矩阵W，偏置向量 b 来和输入向量 X 进行一系列线性运算和激活运算，从输入层开始，一层层地向后计算，一直到运算到输出层，得到输出结果的值。 



**DNN反向传播：**

深度学习的过程即找到网络各层中最优的线性系数矩阵 W 和偏置向量 b，让所有的输入样本通过网络计算后，预测的输出尽可能等于或者接近实际的输出样本，这样的过程一般称为反向传播。

我们需要定义一个损失函数，来衡量预测输出值和实际输出值之间的差异。接着对这个损失函数进行优化，通过不断迭代对线性系数矩阵 W 和 偏置向量 b 进行更新，让损失函数最小化，不断逼近最小极值 或 满足我们预期的需求。在DNN中， 损失函数优化极值求解的过程最常见的一般是通过梯度下降法来一步步迭代完成的。



**DNN在TensorFlow2.0中的一般流程：**

1. Step - 1：数据加载、归一化和预处理；
2. Step - 2：搭建深度神经网络模型；
3. Step - 3：定义损失函数和准确率函数；
4. Step - 4：模型训练；
5. Step - 5：模型预测推理，性能评估。



**DNN中的过拟合：**

随着网络的层数加深，模型的训练过程中会出现 梯度爆炸、梯度消失、欠拟合和过拟合，我们来说说比较常见的过拟合。过拟合一般是指模型的特征维度过多、参数过多，模型过于复杂，导致参数数量大大高于训练数据，训练出的网络层过于完美地适配训练集，但对新的未知的数据集的预测能力很差。即过度地拟合了训练数据，而没有考虑到模型的泛化能力。 

**一般的解决方法：**

-  获取更多数据：从数据源获得更多数据，或做数据增强；
- 数据预处理：清洗数据、减少特征维度、类别平衡；
- 正则化：限制权重过大、网络层数过多，避免模型过于复杂；
- 多种模型结合：集成学习的思想；
- Dropout：随机从网络中去掉一部分隐藏神经元；
- 中止方法：限制训练时间、次数，及早停止。 



接下来，我们通过2种 TensorFlow2.x 推荐的方式 Eager 和 Keras 的代码来演示 DNN 下的 MNIST 训练集的训练和推理，其中的线性函数和交叉熵损失函数等细节说明，请读者参考“[6. MNIST手写字符分类 Logistic Regression](<https://github.com/SciSharp/TensorFlow.NET-Tutorials/blob/master/%E4%BA%8C%E3%80%81TensorFlow.NET%20API-6.%20MNIST%E6%89%8B%E5%86%99%E5%AD%97%E7%AC%A6%E5%88%86%E7%B1%BB%20Logistic%20Regression.md>)”一章节，这里不再赘述。





### 8.2 TensorFlow.NET 代码实操 1 - DNN with Eager

Eager 模式下的 DNN 模型训练流程简述如下：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599098279490.png" alt="1599098279490" style="zoom:50%;" />



按照上述流程，我们进入代码实操阶段。



**① 新建项目，配置环境和引用：**

新建项目。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599110844722.png" alt="1599110844722" style="zoom:67%;" />

选择 .NET Core 框架。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599110989281.png" alt="1599110989281" style="zoom:67%;" />

输入项目名，DNN_Eager。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599111084062.png" alt="1599111084062" style="zoom:67%;" />

确认 .NET Core 版本为 3.0 及以上。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599111140613.png" alt="1599111140613" style="zoom:67%;" />

选择目标平台为 x64 。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599111174994.png" alt="1599111174994" style="zoom:67%;" />

使用 NuGet 安装 TensorFlow.NET 和 SciSharp.TensorFlow.Redist，如果需要使用 GPU，则安装 SciSharp.TensorFlow.Redist-Windows-GPU。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-8.%20%E6%B7%B1%E5%BA%A6%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(DNN)%E5%85%A5%E9%97%A8.assets/1599111399655.png" alt="1599111399655" style="zoom:80%;" />

添加项目引用。

```c#
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;
```



**② 定义网络权重变量和超参数：**

```c#
int num_classes = 10; // MNIST 的字符类别 0~9 总共 10 类
int num_features = 784; // 输入图像的特征尺寸，即像素28*28=784

// 超参数
float learning_rate = 0.001f;// 学习率
int training_steps = 1000;// 训练轮数
int batch_size = 256;// 批次大小
int display_step = 100;// 训练数据 显示周期

// 神经网络参数
int n_hidden_1 = 128; // 第1层隐藏层的神经元数量
int n_hidden_2 = 256; // 第2层隐藏层的神经元数量

IDatasetV2 train_data;// MNIST 数据集
NDArray x_test, y_test, x_train, y_train;// 数据集拆分为训练集和测试集
IVariableV1 h1, h2, wout, b1, b2, bout;// 待训练的权重变量
float accuracy_test = 0f;// 测试集准确率
```



**③ 载入MNIST数据，并进行预处理：**

数据下载 或 从本地加载 → 数据展平 → 归一化 → 转换 Dataset → 无限复制(方便后面take) / 乱序 / 生成批次 / 预加载 → 预处理后的数据提取需要的训练份数。

```c#
((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();// 下载 或 加载本地 MNIST
(x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));// 输入数据展平
(x_train, x_test) = (x_train / 255f, x_test / 255f);// 归一化

train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);//转换为 Dataset 格式
train_data = train_data.repeat()
    .shuffle(5000)
    .batch(batch_size)
    .prefetch(1)
    .take(training_steps);// 数据预处理
```



**④ 初始化网络权重变量和优化器：**

随机初始化网络权重变量，并打包成数组方便后续梯度求导作为参数。

```c#
// 随机初始化网络权重变量，并打包成数组方便后续梯度求导作为参数。
var random_normal = tf.initializers.random_normal_initializer();
h1 = tf.Variable(random_normal.Apply(new InitializerArgs((num_features, n_hidden_1))));
h2 = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_1, n_hidden_2))));
wout = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_2, num_classes))));
b1 = tf.Variable(tf.zeros(n_hidden_1));
b2 = tf.Variable(tf.zeros(n_hidden_2));
bout = tf.Variable(tf.zeros(num_classes));
var trainable_variables = new IVariableV1[] { h1, h2, wout, b1, b2, bout };
```



采用随机梯度下降优化器。

```c#
// 采用随机梯度下降优化器
var optimizer = tf.optimizers.SGD(learning_rate);
```



**⑤ 搭建DNN网络模型，训练并周期显示训练过程：**

搭建4层的全连接神经网络，隐藏层采用 sigmoid 激活函数，输出层采用 softmax 输出预测的概率分布。

```c#
// 搭建网络模型
Tensor neural_net(Tensor x)
{
    // 第1层隐藏层采用128个神经元。
    var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
    // 使用 sigmoid 激活函数，增加层输出的非线性特征
    layer_1 = tf.nn.sigmoid(layer_1);

    // 第2层隐藏层采用256个神经元。
    var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
    // 使用 sigmoid 激活函数，增加层输出的非线性特征
    layer_2 = tf.nn.sigmoid(layer_2);

    // 输出层的神经元数量和标签类型数量相同
    var out_layer = tf.matmul(layer_2, wout.AsTensor()) + bout.AsTensor();
    // 使用 Softmax 函数将输出类别转换为各类别的概率分布
    return tf.nn.softmax(out_layer);
}
```



创建交叉熵损失函数。

```c#
// 交叉熵损失函数
Tensor cross_entropy(Tensor y_pred, Tensor y_true)
{
    // 将标签转换为One-Hot格式
    y_true = tf.one_hot(y_true, depth: num_classes);
    // 保持预测值在 1e-9 和 1.0 之间，防止值下溢出现log(0)报错
    y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
    // 计算交叉熵损失
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)));
}
```



应用 TensorFlow 2.x 中的自动求导机制，创建梯度记录器，自动跟踪网络中的梯度，自动求导进行梯度下降和网络权重变量的更新优化。每隔一定周期，打印出当前轮次网络的训练性能数据 loss 和 accuracy 。关于自动求导机制，请参考“[6. MNIST手写字符分类 Logistic Regression](<https://github.com/SciSharp/TensorFlow.NET-Tutorials/blob/master/%E4%BA%8C%E3%80%81TensorFlow.NET%20API-6.%20MNIST%E6%89%8B%E5%86%99%E5%AD%97%E7%AC%A6%E5%88%86%E7%B1%BB%20Logistic%20Regression.md>)”一章节。

```c#
// 运行优化器
void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y, IVariableV1[] trainable_variables)
{
    using var g = tf.GradientTape();
    var pred = neural_net(x);
    var loss = cross_entropy(pred, y);

    // 计算梯度
    var gradients = g.gradient(loss, trainable_variables);

    // 更新模型权重 w 和 b 
    var a = zip(gradients, trainable_variables.Select(x => x as ResourceVariable));
    optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
}
```



```c#
// 模型预测准确度
Tensor accuracy(Tensor y_pred, Tensor y_true)
{
    // 使用 argmax 提取预测概率最大的标签，和实际值比较，计算模型预测的准确度
    var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
}
```



```c#
// 训练模型
foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
{
    // 运行优化器 进行模型权重 w 和 b 的更新
    run_optimization(optimizer, batch_x, batch_y, trainable_variables);

    if (step % display_step == 0)
    {
        var pred = neural_net(batch_x);
        var loss = cross_entropy(pred, batch_y);
        var acc = accuracy(pred, batch_y);
        print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
    }
}
```



**⑥ 测试集上性能评估：**

在测试集上对训练后的模型进行预测准确率性能评估。

```c#
// 在测试集上对训练后的模型进行预测准确率性能评估
{
    var pred = neural_net(x_test);
    accuracy_test = (float)accuracy(pred, y_test);
    print($"Test Accuracy: {accuracy_test}");
}
```



**完整的控制台运行代码如下：**

```c#
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;

namespace DNN_Eager
{
    class Program
    {
        static void Main(string[] args)
        {
            DNN_Eager dnn = new DNN_Eager();
            dnn.Main();
        }
    }

    class DNN_Eager
    {
        int num_classes = 10; // MNIST 的字符类别 0~9 总共 10 类
        int num_features = 784; // 输入图像的特征尺寸，即像素28*28=784

        // 超参数
        float learning_rate = 0.001f;// 学习率
        int training_steps = 1000;// 训练轮数
        int batch_size = 256;// 批次大小
        int display_step = 100;// 训练数据 显示周期

        // 神经网络参数
        int n_hidden_1 = 128; // 第1层隐藏层的神经元数量
        int n_hidden_2 = 256; // 第2层隐藏层的神经元数量

        IDatasetV2 train_data;// MNIST 数据集
        NDArray x_test, y_test, x_train, y_train;// 数据集拆分为训练集和测试集
        IVariableV1 h1, h2, wout, b1, b2, bout;// 待训练的权重变量
        float accuracy_test = 0f;// 测试集准确率

        public void Main()
        {
            ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();// 下载 或 加载本地 MNIST
            (x_train, x_test) = (x_train.reshape((-1, num_features)), x_test.reshape((-1, num_features)));// 输入数据展平
            (x_train, x_test) = (x_train / 255f, x_test / 255f);// 归一化

            train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);//转换为 Dataset 格式
            train_data = train_data.repeat()
                .shuffle(5000)
                .batch(batch_size)
                .prefetch(1)
                .take(training_steps);// 数据预处理

            // 随机初始化网络权重变量，并打包成数组方便后续梯度求导作为参数。
            var random_normal = tf.initializers.random_normal_initializer();
            h1 = tf.Variable(random_normal.Apply(new InitializerArgs((num_features, n_hidden_1))));
            h2 = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_1, n_hidden_2))));
            wout = tf.Variable(random_normal.Apply(new InitializerArgs((n_hidden_2, num_classes))));
            b1 = tf.Variable(tf.zeros(n_hidden_1));
            b2 = tf.Variable(tf.zeros(n_hidden_2));
            bout = tf.Variable(tf.zeros(num_classes));
            var trainable_variables = new IVariableV1[] { h1, h2, wout, b1, b2, bout };

            // 采用随机梯度下降优化器
            var optimizer = tf.optimizers.SGD(learning_rate);

            // 训练模型
            foreach (var (step, (batch_x, batch_y)) in enumerate(train_data, 1))
            {
                // 运行优化器 进行模型权重 w 和 b 的更新
                run_optimization(optimizer, batch_x, batch_y, trainable_variables);

                if (step % display_step == 0)
                {
                    var pred = neural_net(batch_x);
                    var loss = cross_entropy(pred, batch_y);
                    var acc = accuracy(pred, batch_y);
                    print($"step: {step}, loss: {(float)loss}, accuracy: {(float)acc}");
                }
            }

            // 在测试集上对训练后的模型进行预测准确率性能评估
            {
                var pred = neural_net(x_test);
                accuracy_test = (float)accuracy(pred, y_test);
                print($"Test Accuracy: {accuracy_test}");
            }

        }

        // 运行优化器
        void run_optimization(OptimizerV2 optimizer, Tensor x, Tensor y, IVariableV1[] trainable_variables)
        {
            using var g = tf.GradientTape();
            var pred = neural_net(x);
            var loss = cross_entropy(pred, y);

            // 计算梯度
            var gradients = g.gradient(loss, trainable_variables);

            // 更新模型权重 w 和 b 
            var a = zip(gradients, trainable_variables.Select(x => x as ResourceVariable));
            optimizer.apply_gradients(zip(gradients, trainable_variables.Select(x => x as ResourceVariable)));
        }

        // 模型预测准确度
        Tensor accuracy(Tensor y_pred, Tensor y_true)
        {
            // 使用 argmax 提取预测概率最大的标签，和实际值比较，计算模型预测的准确度
            var correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64));
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis: -1);
        }

        // 搭建网络模型
        Tensor neural_net(Tensor x)
        {
            // 第1层隐藏层采用128个神经元。
            var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
            // 使用 sigmoid 激活函数，增加层输出的非线性特征
            layer_1 = tf.nn.sigmoid(layer_1);

            // 第2层隐藏层采用256个神经元。
            var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
            // 使用 sigmoid 激活函数，增加层输出的非线性特征
            layer_2 = tf.nn.sigmoid(layer_2);

            // 输出层的神经元数量和标签类型数量相同
            var out_layer = tf.matmul(layer_2, wout.AsTensor()) + bout.AsTensor();
            // 使用 Softmax 函数将输出类别转换为各类别的概率分布
            return tf.nn.softmax(out_layer);
        }

        // 交叉熵损失函数
        Tensor cross_entropy(Tensor y_pred, Tensor y_true)
        {
            // 将标签转换为One-Hot格式
            y_true = tf.one_hot(y_true, depth: num_classes);
            // 保持预测值在 1e-9 和 1.0 之间，防止值下溢出现log(0)报错
            y_pred = tf.clip_by_value(y_pred, 1e-9f, 1.0f);
            // 计算交叉熵损失
            return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)));
        }

    }
}

```



**运行结果如下：**

```
step: 100, loss: 562.84094, accuracy: 0.2734375
step: 200, loss: 409.87466, accuracy: 0.51171875
step: 300, loss: 234.70618, accuracy: 0.70703125
step: 400, loss: 171.07526, accuracy: 0.8046875
step: 500, loss: 147.40372, accuracy: 0.86328125
step: 600, loss: 123.477295, accuracy: 0.8671875
step: 700, loss: 105.51019, accuracy: 0.8984375
step: 800, loss: 106.7933, accuracy: 0.87109375
step: 900, loss: 75.033554, accuracy: 0.921875
Test Accuracy: 0.8954
```



我们可以看到，loss 在不断地下降，accuracy 不断提高，最终的测试集的准确率为 0.8954，略低于 训练集上的准确率 0.9219，基本属于一个比较合理的训练结果。





### 8.3 TensorFlow.NET 代码实操 2 - DNN with Keras







### 8.4 视频教程

视频教程链接地址（或扫描下面的二维码）：





### 8.5 代码下载地址

代码下载链接地址（或扫描下面的二维码）：





