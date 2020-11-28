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

**示例1：简单的条件比较运算**

一个简单的比较2个Tensor常量值大小并输出较小值的方法：

```c#
public void Run()
{
    var func = tf.autograph.to_graph(Condition_Min);
    var a = tf.constant(3);
    var b = tf.constant(2);
    var output = func(a, b);
    print(output);

    Console.ReadKey();
}
//TwoInputs_OneOutput_Condition
Tensor Condition_Min(Tensor a, Tensor b)
{
    return tf.cond(a < b, a, b);
}
```

运行代码，正确输出了较小的值 2：

```c#
tf.Tensor: shape=(), dtype=int32, numpy=2
```

**示例2：Lambda表达式的AutoGraph转换**

AutoGraph机制也同样支持Lambda表达式的静态计算图转换，简单示例代码如下：

```c#
public void Run()
{
    var func = tf.autograph.to_graph((x, y) => x * y);
    var output = func(tf.constant(3), tf.constant(2));
    print(output);

    Console.ReadKey();
}
```

运行代码，正确输出了 Lambda 乘法表达式的运算结果 6 ：

```c#
tf.Tensor: shape=(), dtype=int32, numpy=6
```





### 9.2 AutoGraph机制原理

当我们使用 tf.autograph.to_graph() 方法将一个函数转换为静态计算图的时候，程序内部到底发生了什么呢？

例如，我们测试如下代码：

```c#
public void Run()
{
    var func = tf.autograph.to_graph(Add);//AutoGraph
    print("First Run:Initialization");

    var a = tf.constant(2);
    var b = tf.constant(3);
    var output = func(a, b);
    print($"Second Run:{output.numpy()}");

    output = func(output, b);
    print($"Third Run:{output.numpy()}");

    Console.ReadKey();
}
Tensor Add(Tensor a, Tensor b)
{
    foreach (var i in range(3))
        print(i);
    var c = a + b;
    print(c);
    print("tracing");

    return c;
}
```

运行后，我们会看到如下的结果：

```c#
0
1
2
tf.Tensor 'add:0' shape=<unknown> dtype=int32
tracing
First Run:Initialization
Second Run:5
Third Run:8
```

我们来逐步解析这个过程，总的来说，发生了2件事情。

当我们写下 Add() 这个函数体的时候，什么都没有发生，只是在C#内存堆栈中记录下了这样一个函数的签名。

当我们通过 tf.autograph.to_graph() 方法第一次调用这个函数的时候，发生了第1件事情，程序通过解析 Add() 函数体的内容，创建了一个静态计算图。解析的过程会跟踪执行一遍函数体重的代码，确定各个变量的Tensor类型，并根据执行顺序将OP算子添加到计算图中，生成一个固定的静态计算图。

因此我们看到，终端按顺序输出了 循环变量i的值 0 1 2，函数内部Tensor add 算子 c 的初始内容，以及函数体末尾的打印测试字符串 tracing 。这里我们也可以通过函数体内部张量 c 的内容看到，c 只是一个静态计算图中的操作算子，并没有按照 Eager Mode 取得初始运算结果 0 。

当我们第2次调用函数，并传入待运算参数 Tensor 的时候，发生了第2件事情，程序自动调用出刚刚创建的静态计算图，传入待运算的 Tensor 并执行计算图，输出计算的结果。这个过程中，程序不再运行函数体的内部代码，直接调用上一步创建的计算图进行运算，因此函数体内部的诸多测试的 print() 方法不会被再次执行。

因此，程序仅仅输出了第2次的结果 “Second Run:5” 。

这个运算过程不再是 Eager Mode 方式，而是类似在 TensorFlow 1.x 中执行了下面的语句：

```c#
using (var sess = tf.Session(tf.get_default_graph()))
{
    var result = sess.run(tf.constant(2),tf.constant(3));
}
```

第3次调用函数的过程和第2次完全一致，程序依然调用原有的计算图进行重复地运算并输出结果 “Third Run:8” 。

我们可以看到，通过 AutoGraph 机制，可以提高重复运算的执行效率。

这里我通过图示的方式简单地演示该过程，如下图所示：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-9.%20AutoGraph%E6%9C%BA%E5%88%B6.assets/1606552061867.png" alt="1606552061867" style="zoom:50%;" />





### 9.3 AutoGraph编码规范

了解了以上 AutoGraph 的机制原理以后，我们来看下 AutoGraph 都有哪些编码规范。

1. **需要转换的函数体内尽量采用TensorFlow内部的函数，而不是C#中的其它函数**

   使用AutoGraph编译将函数体内部的代码转换成静态计算图，因此对函数体内可使用的语句有一定限制（仅支持 C# 语言的一个子集），且需要函数内的操作本身能够被构建为计算图。所以这里我们建议在函数内只使用 TensorFlow 的原生操作，不要使用过于复杂的 C#原生函数，函数参数只包括 TensorFlow 张量或 NumPy 数组，并最好是能够按照静态计算图的思想去构建函数。通过上述机制原理的演示，我们能够看到，函数体内部的非计算图算子相关的部分，都只会在第1次跟踪执行的时候被运行1次，而普通的C#函数是无法嵌入到TensorFlow计算图中的，后面重复执行的时候只会运行生成的静态计算图，因此非TensorFlow函数不会再被运行。

2. **避免在需要转换的函数体内部定义 Tensor 变量**

   如果在函数体内部定义了Tensor变量，那么创建变量的行为只会发生在第1次跟踪代码逻辑创建计算图时发生，且变量中途无法被调用赋值，失去创建的意义。如果函数体同时被 Eager 方式 和 AutoGraph 方式执行，会导致变量在2种情况下的输出不一致，实际上，很多情况下这时候TensorFlow会出现报错。

3. **需要转换的函数不可修改该函数外部的C#列表或者字典等数据结构变量**

   通过AutoGraph机制转换成的静态计算图是被编译成C++代码在TensorFlow内核中执行的。C#中的列表和字典等数据结构变量是无法嵌入到该计算图中的，它们仅仅能够在第1次的跟踪创建计算图时被读取，在后续重复执行计算图时是无法修改外部C#函数中的列表或字典这样的数据结构变量的。 





以上，就是AutoGraph机制的详细讲解，相信通过这种“神奇机制”，我们可以打造出更强大的兼顾开发速度和执行效率的深度学习算法程序。