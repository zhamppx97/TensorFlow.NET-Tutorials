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



### 2.4 基本张量操作

**① tf.cast** 可以改变张量的数据类型：

```c#
//这个例子演示 将int32类型的值转换为float32类型的值
var h = tf.constant(new[] { 123, 456 }, dtype: tf.int32);
var f = tf.cast(h, tf.float32);
print(h);
print(f);
```

输出结果如下，通过 tf.cast 将int32类型的值转换为float32类型的值： 

tf.Tensor: shape=(2), dtype=int32, numpy=[123, 456]
tf.Tensor: shape=(2), dtype=float32, numpy=[123, 456]



**② tf.range** 创建区间张量值：

```c#
var b = tf.range(1, 10, delta: 2);
print(b);
```

常用参数说明：

参数1 start：区间初始值

参数2 limit：区间限定值，取值<limit（不等于limit）

参数3 delta：区间值递增的差量

输出结果如下：

tf.Tensor: shape=(5), dtype=int32, numpy=[1, 3, 5, 7, 9]



**③ tf.zeros / tf.ones** 创建0值和1值的张量：

下述例子创建一个 3×4 的0值张量 和 4×5的1值张量，一般可用于 张量的初始化。

```c#
var zeros = tf.zeros((3, 4));
print(zeros);
var ones = tf.ones((4, 5));
print(ones);
//TODO://var values = tf.fill((4, 5),6);
//print(values);
```

输出结果如下：

tf.Tensor: shape=(3,4), dtype=float32, numpy=[[0, 0, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0]]
tf.Tensor: shape=(4,5), dtype=float32, numpy=[[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1]]



**④ tf.random** 生成随机分布张量

tf.random.normal随机生成正态分布的张量；

tf.random.truncated_normal随机生成正态分布的张量，剔除2倍方差以外数据；

```c#
var normal1 = tf.random.normal((3, 4), mean: 100, stddev: 10.2f);
print(normal1);
var normal2 = tf.random.truncated_normal((3, 4), mean: 100, stddev: 10.2f);
print(normal2);
```

常用参数说明：

参数1 shape：生成的正态随机分布张量的形状

参数2 mean：正态分布的中心值

参数3 stddev：正态分布的标准差

输出结果如下：

tf.Tensor: shape=(3,4), dtype=float32, numpy=[[115.2682, 102.2946, 108.8016, 105.1554],
[94.24945, 88.12776, 70.64314, 105.8668],
[115.6427, 92.41293, 106.8677, 84.75417]]
tf.Tensor: shape=(3,4), dtype=float32, numpy=[[99.32899, 101.9571, 87.46071, 101.9749],
[101.2237, 105.6187, 105.9899, 98.18528],
[86.55171, 91.12146, 101.8604, 98.7331]]



**⑤ 张量的数学运算** 请参考章节 “二、TensorFlow.NET API-3. Eager Mode”



**⑥ 索引切片** 

可以通过张量的索引读取元素，对于Variable，可以通过索引对部分元素进行修改。

下述为张量的索引功能的演示：

```c#
var t = tf.constant(new NDArray(new[,] {
{11,12,13,14,15 },{ 21,22,23,24,25},{ 31,32,33,34,35},
{ 41,42,43,44,45},{ 51,52,53,54,55},{ 61,62,63,64,65} }));
print(t);

//取第0行
print(t[0]);

//取最后1行
print(t[-1]);

//取第1行第3列
print(t[1, 3]);
print(t[1][3]);

//TODO:
//gather,gather_nd,boolean_mask
//where,scatter_nd
```

输出如下：

tf.Tensor: shape=(6,5), dtype=int32, numpy=[[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55],
[61, 62, 63, 64, 65]]
tf.Tensor: shape=(5), dtype=int32, numpy=[11, 12, 13, 14, 15]
tf.Tensor: shape=(5), dtype=int32, numpy=[61, 62, 63, 64, 65]
tf.Tensor: shape=(), dtype=int32, numpy=24
tf.Tensor: shape=(), dtype=int32, numpy=24



下述为张量的切片功能的演示：

```c#
var t = tf.constant(new NDArray(new[,] {
{11,12,13,14,15 },{ 21,22,23,24,25},{ 31,32,33,34,35},
{ 41,42,43,44,45},{ 51,52,53,54,55},{ 61,62,63,64,65} }));
print(t);

//取第1行至第3行
print(t[new Slice(1, 4)]);

//2种方式：取第1行至最后1行，第0列至最后1列，每隔2列取1列
print(t[new Slice(1, 6), new Slice(0, 5, 2)]);
print(t[new Slice(1), new Slice(step: 2)]);
```

输出如下：

tf.Tensor: shape=(6,5), dtype=int32, numpy=[[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55],
[61, 62, 63, 64, 65]]
tf.Tensor: shape=(3,5), dtype=int32, numpy=[[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45]]
tf.Tensor: shape=(5,3), dtype=int32, numpy=[[21, 23, 25],
[31, 33, 35],
[41, 43, 45],
[51, 53, 55],
[61, 63, 65]]
tf.Tensor: shape=(5,3), dtype=int32, numpy=[[21, 23, 25],
[31, 33, 35],
[41, 43, 45],
[51, 53, 55],
[61, 63, 65]]



### 2.5 维度变换









### 2.6 合并分割







### 2.7 广播











