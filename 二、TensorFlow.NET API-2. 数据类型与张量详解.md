# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 2. 数据类型与张量详解

### 2.1 数据类型

TensorFlow本质上是一个深度学习的科学计算库，这个算法库的主要数据类型为张量，所有的运算也是基于张量数据进行的操作，更复杂的网络模型也只是一些基础运算的组合拼接，只要深入理解基本的张量运算，才能在各种深度学习的网络模型中游刃有余，开发出有价值有创意的算法模型。



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

张量的主要属性为形状、类型和值，可以通过张量的 shape、dtype和numpy()来获取。举例如下：

① 形状：





### 2.3 常量与变量





### 2.4 常用数据操作（tensor？）





### 2.5 广播











