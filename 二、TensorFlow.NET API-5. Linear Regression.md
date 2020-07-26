# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 5. Linear Regression

### 5.1 线性回归问题

**5.1.1 问题描述** 

线性回归是回归问题中的一种，线性回归假设目标值与特征之间线性相关，即满足一个多元一次方程。通过构建损失函数，来求解损失函数最小时的参数 w 和 b 。通常我们可以表达成如下公式：

y^ = w x + b

y^ 为预测值，自变量 x 和因变量 y 是已知的，而我们想实现的是预测新增一个x，其对应的y是多少。因此，为了构建这个函数关系，目标是通过已知数据点，**求解线性模型中 w 和 b 两个参数**。

下图为 w = 1 , b = 0 的情况，其中红色的点为 自变量 x 和因变量 y 的实际值，红色线段即为误差值：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709221132967.png" alt="image-20200709221132967" style="zoom:100%;" />



**5.1.2 问题解析**

求解最佳参数，需要一个标准来对结果进行衡量，为此我们需要定量化一个目标函数式，使得计算机可以在求解过程中不断地优化。

针对任何模型求解问题，都是最终都是可以得到一组预测值y^ ，对比已有的真实值 y ，数据行数为 n ，可以将损失函数定义如下：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709220437954.png" alt="image-20200709220437954" style="zoom:80%;" />

即预测值与真实值之间的平均的平方距离，统计中一般称其为 MAE(mean square error) 均方误差。把之前的函数式代入损失函数，并且将需要求解的参数 w 和 b 看做是函数 L 的自变量，可得：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709220518861.png" alt="image-20200709220518861" style="zoom:80%;" />

现在的任务是求解最小化L时 w 和 b 的值，

即核心目标优化式为：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709220556086.png" alt="image-20200709220556086" style="zoom:80%;" />

**5.1.3 解决方案**

深度学习中一般采用梯度下降 (gradient descent) 的方法求解线性回归问题，梯度下降核心内容是对自变量进行不断的更新（针对w和b求偏导），使得目标函数不断逼近最小值的过程：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709220819205.png" alt="image-20200709220819205" style="zoom:80%;" />

使用梯度下降法，可以对凸问题求得最优解，对非凸问题，可以找到局部最优解。梯度下降法的算法思想如下图所示：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709221504649.png" alt="image-20200709221504649" style="zoom:80%;" />

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200709221515779.png" alt="image-20200709221515779" style="zoom:80%;" />

关于梯度下降的求解过程，这里就不再详细说明，接下来，让我们通过 TensorFlow 的代码来实现一个简单的线性回归。



### 5.2 TensorFlow 下的线性回归

考虑一组数据如下：

| train_X | 3.3  | 4.4  | 5.5  | 6.71 | 6.93  | 4.168 | 9.779 | 6.182 | 7.59 | 2.167 | 7.042 | 10.791 | 5.313 | 7.997 | 5.654 | 9.27 | 3.1  |
| ------- | ---- | ---- | ---- | ---- | ----- | ----- | ----- | ----- | ---- | ----- | ----- | ------ | ----- | ----- | ----- | ---- | ---- |
| train_Y | 1.7  | 2.76 | 2.09 | 3.19 | 1.694 | 1.573 | 3.366 | 2.596 | 2.53 | 1.221 | 2.827 | 3.465  | 1.65  | 2.904 | 2.42  | 2.94 | 1.3  |

现在，我们希望通过对该数据进行线性回归，即使用线性模型 Y = W * X + b 来拟合上述数据，此处 `W` 和 `b` 是待求的参数。

首先，我们使用 NDArray 接收输入数据，如有需要可以进行数据的归一化预处理。

然后，我们使用梯度下降方法来求线性模型中的两个参数  `W` 和 `b` 的值。

在TensorFlow中，我们建立梯度记录器 g = tf.GradientTape()，然后使用 g.gradient(loss, (W, b)) 自动计算梯度，并通过优化器 optimizer.apply_gradients(zip(gradients, (W, b))) 自动更新模型参数  `W` 和 `b` 。

我们来看下完整代码：

```c#
//1. Prepare data
NDArray train_X = np.array(3.3f, 4.4f, 5.5f, 6.71f, 6.93f, 4.168f, 9.779f, 6.182f, 7.59f, 2.167f,
                           7.042f, 10.791f, 5.313f, 7.997f, 5.654f, 9.27f, 3.1f);
NDArray train_Y = np.array(1.7f, 2.76f, 2.09f, 3.19f, 1.694f, 1.573f, 3.366f, 2.596f, 2.53f, 1.221f,
                           2.827f, 3.465f, 1.65f, 2.904f, 2.42f, 2.94f, 1.3f);
int n_samples = train_X.shape[0];

//2. Set weights
var W = tf.Variable(0f, name: "weight");
var b = tf.Variable(0f, name: "bias");
float learning_rate = 0.01f;
var optimizer = tf.optimizers.SGD(learning_rate);

//3. Run the optimization to update weights
int training_steps = 1000;
int display_step = 50;
foreach (var step in range(1, training_steps + 1))
{
    using var g = tf.GradientTape();
    // Linear regression (Wx + b).
    var pred = W * train_X + b;
    // MSE:Mean square error.
    var loss = tf.reduce_sum(tf.pow(pred - train_Y, 2)) / n_samples;
    var gradients = g.gradient(loss, (W, b));

    // Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, (W, b)));
    if (step % display_step == 0)
    {
        pred = W * train_X + b;
        loss = tf.reduce_sum(tf.pow(pred - train_Y, 2)) / n_samples;
        print($"step: {step}, loss: {loss.numpy()}, W: {W.numpy()}, b: {b.numpy()}");
    }
}
```

在这里，我们使用了前文的方式计算了损失函数关于参数的偏导数。同时，使用 `optimizer = tf.optimizers.SGD(learning_rate)` 声明了一个梯度下降 **优化器** （Optimizer），其学习率为 0.01。优化器可以帮助我们根据计算出的求导结果更新模型参数，从而最小化某个特定的损失函数，具体使用方式是调用其 `apply_gradients()` 方法。

在使用 apply_gradients() 方法的时候，需要传入参数 grads_and_vars ，这是一个列表，列表中的每个元素是一个 **(变量的偏导数，变量)** ，即 **[(grad_W, W), (grad_b, b)]** 。代码中的 var gradients = g.gradient(loss, (W, b)) 接收到的返回值是一个 [grad_W,grad_b] ，这里通过 zip() 方法将 变量的偏导数和变量 进行拼装，组合成需要的参数类型。

我们运行代码，可以打印输出正确的梯度下降和线性回归的运算过程：

```
step: 50, loss: 0.20823787, W: 0.3451206, b: 0.13602997
step: 100, loss: 0.19650392, W: 0.33442247, b: 0.2118748
step: 150, loss: 0.18730186, W: 0.3249486, b: 0.2790402
step: 200, loss: 0.18008542, W: 0.31655893, b: 0.33851948
step: 250, loss: 0.17442611, W: 0.30912927, b: 0.39119223
step: 300, loss: 0.16998795, W: 0.30254984, b: 0.43783733
step: 350, loss: 0.1665074, W: 0.29672337, b: 0.47914445
step: 400, loss: 0.16377792, W: 0.29156363, b: 0.51572484
step: 450, loss: 0.16163734, W: 0.28699434, b: 0.54811907
step: 500, loss: 0.15995868, W: 0.28294793, b: 0.57680625
step: 550, loss: 0.1586422, W: 0.2793646, b: 0.6022105
step: 600, loss: 0.15760982, W: 0.2761913, b: 0.6247077
step: 650, loss: 0.15680023, W: 0.27338117, b: 0.6446302
step: 700, loss: 0.15616529, W: 0.27089265, b: 0.6622727
step: 750, loss: 0.15566738, W: 0.26868886, b: 0.6778966
step: 800, loss: 0.15527686, W: 0.26673728, b: 0.69173247
step: 850, loss: 0.15497065, W: 0.26500902, b: 0.70398504
step: 900, loss: 0.15473045, W: 0.26347852, b: 0.7148355
step: 950, loss: 0.15454216, W: 0.2621232, b: 0.72444427
step: 1000, loss: 0.15439445, W: 0.26092294, b: 0.7329534
```

对应的线性回归拟合线如下：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/image-20200714222429798.png" alt="image-20200714222429798" style="zoom:100%;" />



### 5.3 C# 和 Python 的性能比较

通过一个相同数据集的1000轮的线性回归的例子的运行，我们对比 C# 和 Python 的运行速度和内存占用，发现 C# 的速度大约是 Python 的2倍，而内存的使用，C# 只占到 Python 的1/4 ，可以说 TensorFlow 的 C# 版本在速度和性能上同时超过了 Python 版本，因此，在工业现场或者实际应用时，TensorFlow.NET 除了部署上的便利，更有性能上的杰出优势。

下述2个图是具体的对比运行示意图：

<img src="二、TensorFlow.NET API-5. Linear Regression.assets/20200627154950.jpg" alt="20200627154950.jpg" style="zoom:80%;" />



<img src="二、TensorFlow.NET API-5. Linear Regression.assets/python-dotnet-comparision.gif" alt="python-dotnet-comparision" style="zoom:80%;" />



### 5.4 视频教程

视频教程链接地址（或扫描下面的二维码）：https://www.bilibili.com/video/BV1wf4y117qF?p=3
<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-5.%20Linear%20Regression.assets/image-20200721190633289.png" alt="image-20200721190633289" style="zoom:80%;" />