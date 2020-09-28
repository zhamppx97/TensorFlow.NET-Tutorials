# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 7. tf.data 数据集构建与预处理

### 7.1 tf.data 介绍

在工业现场的深度学习项目的开发过程中，我们一般会使用自己的数据集进行模型训练。但是，不同来源和不同格式的原始数据文件的类型十分复杂，有图像的，文本的，音频的，视频的，甚至3D点云的，将这些文件进行读取和预处理的过程十分繁琐，有时比模型的设计还要耗费精力。

比如，为了读入一批图像文件，我们可能需要纠结于 python 的各种图像处理类库，比如 `pillow` 、`openCV`，再创建自己的输入队列，然后自己设计 Batch 的生成方式，最后还可能在运行的效率上不尽如人意。

TensorFlow 一直有实用的异步队列机制，和多线程的能力，可以提高文件读取的效率，早期1.x版本的队列方式一般有4种：**FIFOQueue**、**PaddingFIFOQueue**、**PriorityQueue** 和 **RandomShuffleQueue**。自从Tensorflow1.4发布之后，Datasets就成为了新的给TensorFlow模型创建输入管道（input pipelines）的方法。这个API比用feed_dict（**可能是最慢的一种数据载入方法**）或者queue-based pipelines性能更好，也更易于使用。

为此，在TensorFlow 2.x开始，**TensorFlow 对各数据输入模块进行了整合，提供了 `tf.data` 这一模块并大力推荐开发者使用**，包括了一整套灵活的数据集构建 API，同时集成了map，reduce，batch，shuffle等数据预处理功能，能够帮助我们快速、高效地构建数据输入的流水线，尤其适用于数据量巨大的场景。

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200809152425297.png" alt="image-20200809152425297" style="zoom:50%;" />





### 7.2 tf.data 数据集构建

`tf.data` 的核心是 `tf.data.Dataset` 类，提供了对数据集的高层封装。`tf.data.Dataset` 由一系列的可迭代访问的元素（element）组成，每个元素包含一个或多个张量。比如说，对于一个由图像组成的数据集，每个元素可以是一个形状为 `长×宽×通道数` 的图片张量，也可以是由图片张量和图片标签张量组成的元组（Tuple）。

数据集的创建主要有以下几种方式，我们来介绍一下：

1. **tf.data.Dataset.from_tensor_slices**：最基础的数据集建立方式，也是最常用和推荐的方式，适用于数据集不是特别巨大，可以完整载入内存(PC内存 非GPU内存)中的数据；
2. **tf.data.Dataset.from_generator**：使用生成器generator来初始化Dataset，可以灵活地处理长度不同的元素（如序列）；
3. **tf.data.TFRecordDataset**：这是一种从TFRecord文件里面读取数据的接口，TFRecords也是TensorFlow推荐的数据存取方式，里面每一个元素都是一个tf.train.Example,一般需要先解码才可以使用。对于特别巨大而无法完整载入内存的数据集，我们可以先将数据集处理为 TFRecord 格式，然后使用 `tf.data.TFRocordDataset()` 进行载入；
4. **tf.data.TextLineDataset**：适用于文本数据输入的场景；
5. **tf.data.experimental.make_csv_dataset**：适用于CSV格式数据输入的场景。

以上5种方式是本篇文字在书写的时候，TensorFlow 2.3 推出的API，后续可能会有新的模块推出，或者对现有模块的整合优化，因此，请读者详见TensorFlow官网APII文档中的说明。

接下来，我们来详细说明下最基础的数据集构建方式  `tf.data.Dataset.from_tensor_slices()` 。比如说，我们的数据集的所有元素可以组成1个大的张量，张量的第0维为元素的数量，那么我们输入这样的1个张量，或者多个张量（数据，标签），即可通过 from_tensor_slices 方法构建dataset数据集，该数据集可按照0维进行迭代操作。



从Tensor构建，代码如下：

```c#
var X = tf.constant(new[] { 2013, 2014, 2015, 2016, 2017 });
var Y = tf.constant(new[] { 12000, 14000, 15000, 16500, 17500 });

var dataset = tf.data.Dataset.from_tensor_slices(X, Y);

foreach (var (item_x, item_y) in dataset)
{
    print($"x:{item_x.numpy()},y:{item_y.numpy()}");
}
```

输出如下：

```
x:2013,y:12000
x:2014,y:14000
x:2015,y:15000
x:2016,y:16500
x:2017,y:17500
```



也可以从Numpy数组构建，代码如下：

```c#
var X = np.array(new[] { 2013, 2014, 2015, 2016, 2017 });
var Y = np.array(new[] { 12000, 14000, 15000, 16500, 17500 });

var dataset = tf.data.Dataset.from_tensor_slices(X, Y);

foreach (var (item_x, item_y) in dataset)
{
    print($"x:{item_x.numpy()},y:{item_y.numpy()}");
}
```

和Tensor的方式一样的输出如下：

```
x:2013,y:12000
x:2014,y:14000
x:2015,y:15000
x:2016,y:16500
x:2017,y:17500
```



注意：当输入多个张量时，张量的第0维大小必须相同。



类似地，我们也可以通过这种方式载入 MNIST 数据集，代码如下（我们使用SciSharp的SharpCV进行图像的显示）：

```c#
using NumSharp;
using System;
using static Tensorflow.Binding;
using static SharpCV.Binding;

namespace TF.NET_Test_Core
{
    class Program
    {
        static void Main(string[] args)
        {
            var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
            x_train = np.expand_dims(x_train / 255f, -1);
            var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);

            mnist_dataset = mnist_dataset.take(1);
            foreach (var (image, label) in mnist_dataset)
            {
                cv2.imshow(label.ToString(), image.numpy());
            }
            Console.ReadKey();
        }
    }
}
```

输出结果如下（MNIST的第一个数据是手写字符5）：

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822092235194.png" alt="image-20200822092235194" style="zoom:67%;" />





### 7.3 tf.data 数据预处理

tf.data.Dataset 类提供了很多的数据集预处理和数据提取的方法，我们来介绍常用的几种数据预处理：

**① Dataset.batch() 数据分批**

使用 Dataset.batch() 可以将数据划分为固定大小的批次，方便模型训练过程中数据集采用batch的方式进行投入计算梯度和更新参数。

我们尝试生成大小为 4 的批次，并取 1 个批次显示下效果，代码如下：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
x_train = np.expand_dims(x_train / 255f, -1);
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
mnist_dataset = mnist_dataset.batch(4);
mnist_dataset = mnist_dataset.take(1);
foreach (var (image, label) in mnist_dataset)
{
    print(image.shape);
    print(label.shape);
    //use SharpCV to show the batch images
}
```

结果输出如下：

```
[4, 28, 28, 1]
[4]
```

可以看到，mnist_dataset 数据集的每个元素转变成了大小为 4 的 batch ，你也可以同样的方法使用 SciSharp 的 SharpCV 进行图像的显示，预期结果如下：

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822100912807.png" alt="image-20200822100912807" style="zoom:76%;" />



**② Dataset.shuffle() 数据乱序**

使用 Dataset.shuffle() 方法可以将数据的顺序随机打乱，可以同时组合 batch() 方法，将数据乱序后再设置批次，这样可以消除数据间的顺序关联，训练时非常常用。

shuffle 方法的主要参数为 buffer_size ，我们通过一组图示来看下这个参数的含义。我们举例假设 buffer_size 参数赋值为 6 。

**Step-1：取所有数据的前buffer_size个数据项，填充至 Buffer，如下所示。**

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822210802822.png" alt="image-20200822210802822" style="zoom:67%;" />



**Step-2：从 Buffer 数据区域随机取出 1 条数据进行输出，比如这里随机选择 item4 ，并从列表中输出了该数据，如下所示。**

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822211151918.png" alt="image-20200822211151918" style="zoom:67%;" />



**Step-3：从原来的所有数据中按照顺序选择最新的下 1 条数据（这里是 item7），填充至 Buffer 中上一步输出的那条数据的位置（这里是 item4），如下所示。**

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822211518660.png" alt="image-20200822211518660" style="zoom:67%;" />

然后，一直按照顺序循环执行上述的 Step-2 和 Step-3，就可以实现数据的不断输出（Step-2），期间 Buffer 的大小一直保持 buffer_size （或者 buffer_size-1 ）。

**注意点：这里的 item 数据，不一定为单条数据，如果组合 batch 方法，则取出的 1 条 item 中包含了 batch_size 条真实的数据。**

接下来，我们通过代码来看下数据打乱的效果，我们将缓存大小设置为10000，批次大小设置为4：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
x_train = np.expand_dims(x_train / 255f, -1);
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
mnist_dataset = mnist_dataset.shuffle(10000).batch(4);
mnist_dataset = mnist_dataset.take(1);
foreach (var (image, label) in mnist_dataset)
{
    cv2.imshow(label.ToString(), image.numpy());
}
```

我们多次运行这个程序，可以看到，每次的输出数据都是随机打乱的：

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822225858787.png" alt="image-20200822225858787" style="zoom:67%;" />

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200822225957434.png" alt="image-20200822225957434" style="zoom:67%;" />



**③ Dataset.repeat() 数据复制**

使用 Dataset.repeat() 可以对数据进行复制，它的参数 count 为复制的倍数，默认参数值 count = -1 为无限倍数的复制，我们使用 count = 2 来复制数据集为原来的 2倍进行测试，代码如下：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
x_train = np.expand_dims(x_train / 255f, -1);
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
int n = 0;
foreach (var (image, label) in mnist_dataset)
{
    n += 1;
}
print(n);

mnist_dataset = mnist_dataset.repeat(2);
n = 0;
foreach (var (image, label) in mnist_dataset)
{
    n += 1;
}
print(n);
```

运行上述代码，可以看到数据集的数量 从 60000 增加为 120000，增加了 2 倍，数据结果如下：

```
60001
120001
```



**④ Dataset.prefetch() 数据预取出（并行化策略）**

使用 Dataset.prefetch() 可以进行并行化的数据读取，充分利用计算资源，减少CPU数据加载和GPU训练之间的切换空载时间。我们用图示演示 Dataset.prefetch() 的效果，通过这个方法可以让数据集在训练的时候预先取出若干个元素，使得 GPU 训练的同时 CPU 可以并行地准备数据，从而提升训练的效率：

**普通模式：**

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200823231239910.png" alt="image-20200823231239910" style="zoom:50%;" />

**开启 prefetch 模式：**

<img src="二、TensorFlow.NET API-7. tf.data 数据集构建与预处理.assets/image-20200823231324083.png" alt="image-20200823231324083" style="zoom:50%;" />

prefetch() 方法的主要参数为 buffer_size ，表示将被加入缓冲器的元素的最大数量，这个参数会告诉 TensorFlow ，让其创建一个容纳至少 buffer_size 个元素的 buffer 区域，然后通过后台线程在后台并行地填充 buffer，从而提高运算性能。

我们来一起看下 prefetch() 方法的示例代码：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
x_train = np.expand_dims(x_train / 255f, -1);
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
mnist_dataset = mnist_dataset.prefetch(1);
foreach (var (image, label) in mnist_dataset)
{
    //do somthing here
}
```



**⑤ Dataset.map()**

Dataset.map(f) 对数据集中的每个元素应用函数 f （**自定义函数或者 Lambda 表达式**），得到一个新的数据集。这一功能在数据预处理中修改 dataset 中元素是很实用的，往往结合 tf.io 进行读写和解码文件，或者结合 tf.image 进行图像处理。

map() 方法的主参数为 回调方法 (map_func)，map_func 的参数和返回值均必须有对应 dataset 中元素类型。



接下来，我们通过几个示例来进行演示。



**示例1：传自定义函数的方式，我们使用 map 方法对 dataset 中的数据进行类型转换。**

我们先自定义类型转换的函数，代码如下：

```c#
static Tensor change_dtype(Tensor t)
{
    return tf.cast(t, dtype: TF_DataType.TF_INT32);
}
```

然后我们通过 map 方法，对一个 dataset 中的每个数据进行自定义函数 change_dtype 的应用。

```c#
var dataset = tf.data.Dataset.range(0, 3);
print("dataset:");
foreach (var item in dataset)
{
    print(item);
}

var dataset_map = dataset.map(change_dtype);
print("\r\ndataset_map:");
foreach (var item in dataset_map)
{
    print(item);
}
```

运行上述代码，我们可以得到正确的数据类型转换（ int64 → int32 ）的结果：

```
dataset:
(tf.Tensor: shape=(), dtype=int64, numpy=0, )
(tf.Tensor: shape=(), dtype=int64, numpy=1, )
(tf.Tensor: shape=(), dtype=int64, numpy=2, )

dataset_map:
(tf.Tensor: shape=(), dtype=int32, numpy=0, )
(tf.Tensor: shape=(), dtype=int32, numpy=1, )
(tf.Tensor: shape=(), dtype=int32, numpy=2, )
```



**示例2：传 Lambda 的方式，我们使用 map 方法对 dataset 中的数据进行数值运算。**

下述代码通过 map 方法，对一个 dataset 中的每个数据进行 Lambda 表达式（ x = x + 10 ）的操作。

```c#
var dataset = tf.data.Dataset.range(0, 3);
print("dataset:");
foreach (var item in dataset)
{
    print(item);
}

var dataset_map = dataset.map(x => x + 10);
print("\r\ndataset_map:");
foreach (var item in dataset_map)
{
    print(item);
}
```

运行上述代码，我们可以得到正确的数值运算（ x = x + 10 ）的结果：

```
dataset:
(tf.Tensor: shape=(), dtype=int64, numpy=0, )
(tf.Tensor: shape=(), dtype=int64, numpy=1, )
(tf.Tensor: shape=(), dtype=int64, numpy=2, )

dataset_map:
(tf.Tensor: shape=(), dtype=int64, numpy=10, )
(tf.Tensor: shape=(), dtype=int64, numpy=11, )
(tf.Tensor: shape=(), dtype=int64, numpy=12, )
```



**示例3：传自定义函数的方式，我们来尝试处理 tuple 类型的元素。**

对于 tuple 类型的元素处理，我们可以这样定义 map_func：





**示例4：使用 map 方法将所有 mnist 图片旋转90度。**

示例代码片段如下：





和 prefetch() 方法类似，map() 也可以利用 GPU 的性能并行化地对数据进行处理，提高效率。通过设置 Dataset.map() 的 use_inter_op_parallelism （默认为 true 开启），实现数据处理的并行化，如下图所示。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-7.%20tf.data%20%E6%95%B0%E6%8D%AE%E9%9B%86%E6%9E%84%E5%BB%BA%E4%B8%8E%E9%A2%84%E5%A4%84%E7%90%86.assets/1601282712428.png" alt="1601282712428" style="zoom:67%;" />





**⑥ 组合使用**

上述的一些 tf.data  的数据预处理，都可以进行任意地组合使用，例如在示例代码 MNIST 的 Logistics Regression 中，我们组合了数据复制 repeat() 、数据乱序 shuffle(5000) 、生成批次 batch(256) 和 数据预取 prefetch(1) ，代码如下：

```c#
var ((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data();
(x_train, x_test) = (x_train.reshape((-1, 784)), x_test.reshape((-1, 784)));
(x_train, x_test) = (x_train / 255f, x_test / 255f);
var train_data = tf.data.Dataset.from_tensor_slices(x_train, y_train);
train_data = train_data.repeat().shuffle(5000).batch(256).prefetch(1);
```



### 7.4 tf.data 数据使用

构建好数据并预处理后，我们需要从其中迭代获取数据以用于训练。`tf.data.Dataset` 是一个可迭代对象，因此可以使用 For 循环迭代获取数据，tf.data的数据提取输出和使用主要有3种方式：



**① 直接 For 循环**

直接 For 循环迭代提取数据使用，代码如下：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
foreach (var (image, label) in mnist_dataset)
{
    //do somthing here
}
```



**② Dataset.take() 取出部分或全部数据**

可以通过 Dataset.take() 取出 Dataset 中的部分或全部（参数设置为全部数据的大小）数据，代码如下：

```c#
var ((x_train, y_train), (_, _)) = tf.keras.datasets.mnist.load_data();
var mnist_dataset = tf.data.Dataset.from_tensor_slices(x_train, y_train);
mnist_dataset = mnist_dataset.take(1);//take one element
foreach (var (image, label) in mnist_dataset)
{
    //do somthing here
}
```



**//TODO:③ 直接输入至 keras 使用**

Keras 支持使用 tf.data.Dataset 直接作为输入。当调用 tf.keras.Model 的 fit() 和 evaluate() 方法时，可以将参数中的输入数据指定为一个元素格式为 (输入数据, 标签数据) 的 Dataset 。例如，对于 MNIST 数据集，我们可以直接传入 Dataset ：

```c#
model.fit(mnist_dataset, epochs=num_epochs);
```

由于已经通过 Dataset.batch() 方法划分了数据集的批次，所以这里也无需提供批次的大小。