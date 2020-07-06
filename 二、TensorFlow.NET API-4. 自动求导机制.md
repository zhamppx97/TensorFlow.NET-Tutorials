# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 4. 自动求导机制

### 4.1 机器学习中的求导

随机梯度下降法（SGD）是训练深度学习模型最常用的优化方法，通过梯度的定义我们可以发现，梯度的求解其实就是求函数偏导的问题，而导数在非严格意义上来说也就是一元的”偏导”。

常用的求导一般有3种方式，数值微分(Numerical Differentiation)，符号微分(Symbolic Differentiation)，前向模式（Forward Mode)和反向模式(Reverse Mode)数值微分。目前的深度学习框架大都实现了自动求梯度的功能，你只关注模型架构的设计，而不必关注模型背后的梯度是如何计算的。

Tensorflow的自动求导（automatic gradient / Automatic Differentiation）就是基于反向模式，就是我们常说的BP算法，其基于的原理是链式法则。实现的方式是利用反向传递与链式法则建立一张对应原计算图的梯度图，因为导数只是另外一张计算图，可以再次运行反向传播，对导数再进行求导以得到更高阶的导数。通过这种方式，我们仅需要一个前向过程和反向过程就可以计算所有参数的导数或者梯度，这对于拥有大量训练参数的神经网络模型梯度的计算特别适合。

下图分别展示了普通函数和神经网络的自动求导示意图：

<img src="二、TensorFlow.NET API-4. 自动求导机制.assets/image-20200705211534801.png" alt="image-20200705211534801" style="zoom:67%;" />

<img src="二、TensorFlow.NET API-4. 自动求导机制.assets/image-20200705211623944.png" alt="image-20200705211623944" style="zoom:67%;" />

接下来，我们通过代码演示下 TensorFlow 2 中的自动求导机制。



### 4.2 简单函数求导

在即时执行模式下，TensorFlow 引入了 `tf.GradientTape()` 这个 “求导记录器” 来实现自动求导。以下代码展示了如何使用 `tf.GradientTape()` 计算函数 y(x) = x² 在 x=3 处的导数：

```c#
using NumSharp;
using System;
using Tensorflow;
using Tensorflow.Gradients;
using static Tensorflow.Binding;

namespace TF.NET_Test_Core
{
    class Program
    {
        static void Main(string[] args)
        {
            var x = tf.Variable(3.0, dtype: TF_DataType.TF_FLOAT);
            using var tape = tf.GradientTape();//在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
            var y = tf.square(x);
            var y_grad = tape.gradient(y, x);//计算y关于x的导数
            print(y);
            print(y_grad);

            Console.ReadKey();
        }
    }
}
```

注意：版本需要 c# 8.0 及以上，.NET Core 3.0 及以上，目标平台 x64；

输出：

tf.Tensor: shape=(), dtype=float32, numpy=9
tf.Tensor: shape=(), dtype=float32, numpy=6



这里 `x` 是一个初始化为 3 的 **变量** （Variable），使用 `tf.Variable()` 声明。变量与普通张量的一个重要区别是其默认能够被 TensorFlow 的自动求导机制所求导，因此往往被用于定义机器学习模型的参数。

`tf.GradientTape()` 是一个自动求导的记录器。只要进入了 `using var tape = tf.GradientTape();` 的上下文环境，则在该环境中计算步骤都会被自动记录。比如在上面的示例中，计算步骤 `var y = tf.square(x);` 即被自动记录。离开上下文环境后，记录将停止，但记录器 `tape` 依然可用，因此可以通过 `var y_grad = tape.gradient(y, x);` 求张量 `y` 对变量 `x` 的导数。

最后程序正确输出了 函数 y=x² 在 x=3 时的结果为 3²=9，同时输出了 函数 y=x² 对 x 的导数 y=2\*x 在 x=3 时的结果为 2\*3=6 。



### 4.3 复杂函数求偏导













