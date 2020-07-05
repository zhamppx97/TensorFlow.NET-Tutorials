# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 4. 自动求导机制

### 4.1 机器学习中的求导

随机梯度下降法（SGD）是训练深度学习模型最常用的优化方法，通过梯度的定义我们可以发现，梯度的求解其实就是求函数偏导的问题，而导数在非严格意义上来说也就是一元的”偏导”。

常用的求导一般有3种方式，数值微分(Numerical Differentiation)，符号微分(Symbolic Differentiation)，前向模式（Forward Mode)和反向模式(Reverse Mode)数值微分。目前的深度学习框架大都实现了自动求梯度的功能，你只关注模型架构的设计，而不必关注模型背后的梯度是如何计算的。

Tensorflow的自动求导（automatic gradient / Automatic Differentiation）就是基于反向模式，就是我们常说的BP算法，其基于的原理是链式法则。实现的方式是利用反向传递与链式法则建立一张对应原计算图的梯度图，因为导数只是另外一张计算图，可以再次运行反向传播，对导数再进行求导以得到更高阶的导数。通过这种方式，我们仅需要一个前向过程和反向过程就可以计算所有参数的导数或者梯度，这对于拥有大量训练参数的神经网络模型梯度的计算特别适合。

<img src="二、TensorFlow.NET API-4. 自动求导机制.assets/image-20200705211534801.png" alt="image-20200705211534801" style="zoom:67%;" />

<img src="二、TensorFlow.NET API-4. 自动求导机制.assets/image-20200705211623944.png" alt="image-20200705211623944" style="zoom:67%;" />

接下来，我们通过代码演示下 TensorFlow 2 中的自动求导机制。



### 4.2 简单函数求导









### 4.3 复杂函数求偏导













