# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 9. AutoGraph机制

### 9.1 AutoGraph方法说明 

TensorFlow 2.x 有 3 种计算图的搭建模式，分别是 静态计算图（TensorFlow 1.x 主要使用的方式）、动态计算图和 AutoGraph 方式。在TensorFlow 2.x 中，官方主推使用 动态计算图方式 和 AutoGraph 方式。

动态计算图方便调试，代码可读性强，对于程序员来说编码效率高，但是运行效率不如静态计算图。而静态计算图的执行效率非常高，但是代码书写和可读性差，程序调试困难。

AutoGraph 则是兼顾了动态计算图的编码效率和静态计算图的执行效率这 2 大优点于一身，从字面上理解，AutoGraph机制 就是自动计算图转化功能，可以将按照动态计算图书写规则开发的代码，通过AutoGraph机制，转换成静态计算图并按照静态计算图的方式进行执行，达到 “动态计算图快速编写代码和调试代码” + “静态计算图内部高效执行代码” 的目的。

当然，AutoGraph机制能够转换的代码和使用的场景并非无所约束（要不然就会完全取代静态计算图和动态计算图），AutoGraph机制的正确使用，需要遵循一定的编码规则，同时需要深入理解AutoGraph进行计算图转换的内部运行过程，否则就会出现转换失败，或者出现异常的不符合预期的执行过程。

后面2节我们会详细说明 AutoGraph的内部运行机制原理和编码规范，接下来我们通过一个简单的例子来了解下AutoGraph机制的使用方法。



这是一个最简单的 Tensor 数值乘法运算：

```c#
public void Run()
{
    var a = tf.constant(2);
    var b = tf.constant(3);
    var output = Mul(a, b);
    print(output);

    Console.ReadKey();
}
Tensor Mul(Tensor a, Tensor b)
{
    return a * b;
}
```

运行后会输出数值相乘的结果 2 * 3 = 6：

```c#
tf.Tensor: shape=(), dtype=int32, numpy=6
```

这个过程是动态计算图方式进行的，即 Eager Mode 下的数值运算。

在 TensorFlow.NET 中我们通过 **tf.autograph.to_graph()** 方法将动态计算图转换为静态计算图，用法很简单，直接增加一句转换代码即可完成，我们来对上述的乘法运算进行 AutoGraph 转换测试下。

**加入 tf.autograph.to_graph() 进行 AutoGraph 转换：**

```c#
public void Run()
{
    var func = tf.autograph.to_graph(Mul);//AutoGraph
    var a = tf.constant(2);
    var b = tf.constant(3);
    var output = func(a, b);
    print(output);

    Console.ReadKey();
}
Tensor Mul(Tensor a, Tensor b)
{
    return a * b;
}
```

通过增加一行代码 "var func = tf.autograph.to_graph(Mul);" ，我们就可以实现 AutoGraph机制的计算图转换，乘法运算的运行结果不变：

```c#
tf.Tensor: shape=(), dtype=int32, numpy=6
```



**AutoGraph的运行效率**

一般来说，计算图模型由大量小的操作构成的时候，AutoGraph 的效率提升较大，如果模型操作数很少的时候，AutoGraph 的效率提升不大，有时候反而由于计算图转换产生的耗时，造成运行时间反而加长。

我们对上一个例子进行一下简单修改，增加一个循环测试，来测试下不同情况下的运行时间。

**情况1#：我们直接增加循环，将乘法运算循环运行一百万次**

代码如下：

```c#
public void Run()
{
    Stopwatch sw = new Stopwatch();
    sw.Restart();
    foreach (var item in range(1000000))
    {
        var a = tf.constant(2);
        var b = tf.constant(3);
        var output = Mul(a, b);
    }
    sw.Stop();
    print("Eager Mode：" + sw.Elapsed.TotalSeconds.ToString("0.0 s"));

    sw.Restart();
    var func = tf.autograph.to_graph(Mul);//AutoGraph
    foreach (var item in range(1000000))
    {
        var a = tf.constant(2);
        var b = tf.constant(3);
        var output = func(a, b);
    }
    sw.Stop();
    print("AutoGraph Mode：" + sw.Elapsed.TotalSeconds.ToString("0.0 s"));

    Console.ReadKey();
}
Tensor Mul(Tensor a, Tensor b)
{
    return a * b;
}
```

运行结果如下：

```c#
Eager Mode：20.2 s
AutoGraph Mode：41.8 s
```

我们可以看到 Eager 模式 20.2 s 反而比 AutoGraph 模式 41.8 s 快（不同PC配置差异可能运算结果略有不同）。



**情况2#：我们增加一点操作数，再将乘法运算同样循环运行一百万次**

增加了乘法运算的操作数，修改为 4 次连乘 "a * b * a * b * a * b * a * b"，代码如下：

```c#
public void Run()
{
    Stopwatch sw = new Stopwatch();
    sw.Restart();
    foreach (var item in range(1000000))
    {
        var a = tf.constant(2);
        var b = tf.constant(3);
        var output = Mul(a, b);
    }
    sw.Stop();
    print("Eager Mode：" + sw.Elapsed.TotalSeconds.ToString("0.0 s"));

    sw.Restart();
    var func = tf.autograph.to_graph(Mul);//AutoGraph
    foreach (var item in range(1000000))
    {
        var a = tf.constant(2);
        var b = tf.constant(3);
        var output = func(a, b);
    }
    sw.Stop();
    print("AutoGraph Mode：" + sw.Elapsed.TotalSeconds.ToString("0.0 s"));

    Console.ReadKey();
}
Tensor Mul(Tensor a, Tensor b)
{
    return a * b * a * b * a * b * a * b;
}
```

运行结果如下：

```c#
Eager Mode：78.3 s
AutoGraph Mode：47.1 s
```

这次，我们可以看到 Eager 模式 78.3 s 比 AutoGraph 模式 47.1 s 慢了许多（不同PC配置差异可能运算结果略有不同），AutoGraph的效率优势得到了体现。



接下来，我们看下 AutoGraph的其它简单的代码示例。























### 9.2 AutoGraph机制原理









### 9.3 AutoGraph编码规范

 使用静态编译将函数内的代码转换成计算图，因此对函数内可使用的语句有一定限制（仅支持 Python 语言的一个子集），且需要函数内的操作本身能够被构建为计算图。建议在函数内只使用 TensorFlow 的原生操作，不要使用过于复杂的 Python 语句，函数参数只包括 TensorFlow 张量或 NumPy 数组，并最好是能够按照计算图的思想去构建函数 

