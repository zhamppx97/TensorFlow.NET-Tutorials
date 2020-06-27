# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 2. 数据类型与张量详解

### 2.1 数据类型

TensorFlow本质上是一个深度学习的科学计算库，这个算法库的主要数据类型为张量，所有的运算也是基于张量数据进行的操作，更复杂的网络模型也只是一些基础运算的组合拼接，只有深入理解基本的张量运算，才能在各种深度学习的网络模型中游刃有余，开发出有价值有创意的算法模型。



TensorFlow中基本的数据类型，主要包含数值类型、字符串类型和布尔类型。下面简单举例一下：

① 数值类型：var x = tf.Variable(10, name: "x");

② 字符串类型：var mammal1 = tf.Variable("Elephant", name: "var1", dtype: tf.@string);

③ 布尔类型：var bo =  tf.Variable(true);



具体数据类型如下表：

```c#
public TF_DataType byte8 = TF_DataType.TF_UINT8;
public TF_DataType int8 = TF_DataType.TF_INT8;
public TF_DataType int16 = TF_DataType.TF_INT16;
public TF_DataType int32 = TF_DataType.TF_INT32;
public TF_DataType int64 = TF_DataType.TF_INT64;
public TF_DataType float16 = TF_DataType.TF_HALF;
public TF_DataType float32 = TF_DataType.TF_FLOAT;
public TF_DataType float64 = TF_DataType.TF_DOUBLE;
public TF_DataType @bool = TF_DataType.TF_BOOL;
public TF_DataType chars = TF_DataType.TF_STRING;
public TF_DataType @string = TF_DataType.TF_STRING;
```



### 2.2 张量详解

类似于 NumPy 中的 N 维数组对象 NDArray，TensorFlow中数据的基本单位为**张量**（Tensor），二者都是多维数组的概念，我们可以使用张量表示**标量**（0维数组）、**向量**（1维数组）、**矩阵**（2维数组）。

张量的主要特性为形状、类型和值，可以通过张量的 属性 shape、dtype 和 方法 numpy()来获取。举例如下：

**① 形状：**

```c#
var x = tf.constant(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
Console.WriteLine(Enumerable.SequenceEqual(new[] { 2, 4 }, x.shape));
```

上述代码执行返回结果 True，x.shape 通过返回整型一维数组的方式，显示Tensor的形状。

你也可以直接通过 Tensorflow.Binding 封装的 print() 方法直接输出形状：

```c#
var x = tf.constant(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
print(x.shape);
```

输出如下：

[2, 4]

**② 类型：**

```c#
var x = tf.constant(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
print(x.dtype);
```

输出如下：

TF_INT32

Tensor 的属性 dtype 返回 TF_DataType类型的枚举，可以很方便地通过 print()方法进行输出。

**③ 获取值**

```c#
var x = tf.constant(new[,] { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } });
print(x.numpy());
```

输出如下：

[[1, 2, 3, 4],
[5, 6, 7, 8]]

Tensor 的 numpy() 方法返回 NumSharp.NDArray 类型的返回值，内容为 Tensor 保存的值内容，可以很方便地通过 print()方法进行输出。



### 2.3 常量与变量

从行为特性来看，有两种类型的张量，常量 constant 和变量 Variable 。

常量的值在计算图中不可以被重新赋值，变量可以在计算图中用assign等算子重新赋值。

**① 常量**

一般的常量类型如下：

```c#
var i = tf.constant(1); // tf.int32 类型常量
var l = tf.constant(1, dtype: TF_DataType.TF_INT64); // tf.int64 类型常量
var f = tf.constant(1.23); //tf.float32 类型常量
var d = tf.constant(3.14, dtype: TF_DataType.TF_DOUBLE); // tf.double 类型常量
var s = tf.constant("hello world"); // tf.string类型常量
//TODO:var b = tf.constant(true); //tf.bool类型常量


print(i);
print(l);
print(f);
print(d);
print(s);
print(b);
```

输出如下：

tf.Tensor: shape=(), dtype=int32, numpy=1
tf.Tensor: shape=(), dtype=TF_INT64, numpy=1
tf.Tensor: shape=(), dtype=TF_DOUBLE, numpy=1.23
tf.Tensor: shape=(), dtype=TF_DOUBLE, numpy=3.14
tf.Tensor: shape=(), dtype=string, numpy=b'hello world'
//TODO:最后一行待确定 tf.Tensor: shape=(), dtype=tf@bool, numpy=true



我们来写一个4维的常量，并通过 tf.rank 函数返回张量的秩。

一般来说，标量为0维张量，向量为1维张量，矩阵为2维张量。

彩色图像有rgb三个通道，可以表示为3维张量。

视频再增加一个时间维，即可以表示为4维张量。

代码如下：

```c#
var tensor4 = tf.constant(new NDArray(new[, , ,]{ { { { 1.0, 1.0 }, { 2.0, 2.0 } },{ { 3.0, 3.0 },{4.0,4.0 } } },
                                                 {{{ 5.0,5.0 },{6.0,6.0 } },{{7.0,7.0 },{8.0,8.0 } } } })); // 4维张量
print(tensor4);
print(tf.rank(tensor4));
```

代码运行输出如下：

tf.Tensor: shape=(2,2,2,2), dtype=TF_DOUBLE, numpy=[[[[1, 1],
[2, 2]],
[[3, 3],
[4, 4]]],
[[[5, 5],
[6, 6]],
[[7, 7],
[8, 8]]]]
tf.Tensor: shape=(), dtype=int32, numpy=4



**② 变量**

深度学习模型中一般被训练的参数需要定义为变量，变量的值可以在模型训练过程中被修改。

我们简单测试一个二维数组的变量：

```c#
var v = tf.Variable(new[,] { { 1, 2 } }, name: "v");
print(v);
```

代码输出如下：

tf.Variable: 'v:0' shape=(1, 2), dtype=int32, numpy=[[1, 2]]



接下来我们一起看下常量和变量的差别，常量的值也可以参与运算，也可以被重新赋值，但是重新赋值或者运算后的结果会开辟新的内存空间，而变量的值可以通过 assign, assign_add 等方法给变量重新赋值，代码如下：

```c#
unsafe
{
    TypedReference r;
    long pointerToV;

    var V = tf.Variable(new[,] { { 1, 2 } });
    r = (__makeref(V));
    pointerToV = (long)*(IntPtr**)&r;
    print($"Value of the variable: {V}");
    Console.WriteLine($"Address of the variable: {pointerToV}");

    V.assign_add(tf.constant(new[,] { { 3, 4 } }));
    r = (__makeref(V));
    pointerToV = (long)*(IntPtr**)&r;
    print($"Value of the variable: {V}");
    Console.WriteLine($"Address of the variable: {pointerToV}");
}
```

输出结果如下：

Value of the variable: tf.Variable: 'Variable:0' shape=(1, 2), dtype=int32, numpy=[[1, 2]]
Address of the variable: 180185198040
Value of the variable: tf.Variable: 'Variable:0' shape=(1, 2), dtype=int32, numpy=[[4, 6]]
Address of the variable: 180185198040



### 2.4 常用数据操作

tf.cast 可以改变张量的数据类型：

```c#

```















### 2.5 广播











