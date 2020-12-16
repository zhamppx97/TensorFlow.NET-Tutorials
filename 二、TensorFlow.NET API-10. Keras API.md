# [C# TensorFlow 2 入门教程](<https://github.com/SciSharp/TensorFlow.NET-Tutorials>)

# 二、TensorFlow.NET API

## 10. Keras API

### 10.1 Keras 简要介绍

2015年11月9日，Google 在 GitHub 上开源了 TensorFlow，同一年的3月，Keras 发布，支持 "TensorFlow", "Theano" 和"CNTK" 等 backend (后端)。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608022347132.png" alt="1608022347132" style="zoom:80%;" />

Keras 的出现大大降低了深度学习的入门门槛，易用性的同时也并不降低灵活性，我们从 Keras 官网的文字可以总结出 Keras 的主要特点：

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608104090066.png" alt="1608104090066" style="zoom:80%;" />

- 为人类习惯设计的深度学习

  Keras是为人类思维习惯而非机器逻辑设计的API。Keras 始终遵循最少学习成本的准则，它提供了简单易用的API，尽量减少经常使用的深度学习方法需要的开发量，并且提供了清晰可快速定位排查的错误消息机制，同时，它还具有大量完善的文档和开发指南。

- 创意的快速实现和灵活性

  Keras 是 [Kaggle](https://www.kaggle.com/) 上前5名获胜团队中最常用的深度学习框架，因为Keras的快速模型编写方式，使得进行创新的实验变得更加容易，所以它能使你比对手更快地尝试更多的想法。欧洲核子研究组织（CERN），美国国家航空航天局（NASA），美国国立卫生研究院（NIH）和世界上许多其他科学组织都在使用Keras（包括大型强子对撞机也在使用Keras）。Keras具有底层灵活性，可以开发任意的研究思路和创意想法，同时提供可选的高级便利功能，以加快实验的整体周期。

- 万亿级的机器学习部署

  [Keras](https://www.tensorflow.org/) 建立在 [TensorFlow 2.0](https://www.tensorflow.org/) 之上，是一个行业领先的框架，可以扩展到大型GPU集群或整个 [TPU云端](https://cloud.google.com/tpu)。这不仅是可能的，而且很容易实现。 

- 模型适用多平台部署

  充分利用 TensorFlow 框架的全面部署功能，你可以将 Keras 模型导出为 JavaScript 方式直接在浏览器中运行，也可以导出为 TF Lite 方式在iOS、Android 和嵌入式设备上运行，也可以通过 Web API  在 Web 上轻松部署 Keras 模型。 

- 丰富的生态系统

  Keras 紧密连接于 TensorFlow 2.x 生态系统的核心部分，涵盖了机器学习工作流程的每个步骤，从数据管理到超参数训练再到部署解决方案。 



2019年3月7日凌晨， 谷歌一年一度的 TensorFlow 开发者大会（TensorFlow DEV SUMMIT 2019）在加州举行。这次大会发布了  TensorFlow 2.0 Alpha 版，同时 TensorFlow 的 Logo 也变成了流行的扁平化设计。

 <img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/196414261.gif" alt="img" style="zoom:80%;" /> 

同时伴随 Alpha 版本的更新，TensorFlow 团队表达了对 Keras 更深的爱，Keras 整合进了 tf.keras 高层API，成为了 TensorFlow 的一部分。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608105508138.png" alt="1608105508138" style="zoom:80%;" />

2020年，Keras之父  **弗朗索瓦.肖莱(Francois Chollet)**  正式加入 Google 人工智能小组，同年的 TensorFlow 开发者峰会（TensorFlow DEV SUMMIT 2020）上，Keras（tf.keras） 正式内置为 TensorFlow 的核心 API，得到 TensorFlow 的全面支持，并成为 TensorFlow 官方推荐的首选深度学习建模工具。自此，TensorFlow 和 Keras 完美地结合，成为大家入门深度学习的首选API。

<img src="%E4%BA%8C%E3%80%81TensorFlow.NET%20API-10.%20Keras%20API.assets/1608106027998.png" alt="1608106027998" style="zoom:67%;" />





### 10.2 模型（Model）与层（Layer） API







### 10.3 Keras 常用训练和参数类





### 10.4 Keras 建立模型的3种方式





### 10.5 Keras完整示例





