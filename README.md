[TOC]

# Learning to Hash Naturally Sorts

## 模型简介

​		Learning to Hash Naturally Sorts发表在[IJCAI 2022(CCFA)]([arxiv.org](https://arxiv.org/abs/2201.13322))。NSH是一种无监督深度哈希学习方法，旨在解决数据相似性排序中的非可微问题。它采用SoftSort方法，通过对样本哈希码的汉明距离进行排序，并利用排序噪声对比估计（SortedNCE）损失，动态优化数据的排序检索候选列表。NSH在端到端训练中有效利用数据的语义信息，在提升检索性能的同时显著提高哈希码的质量和排序准确性。

<img src=\images\1.png alt="image-20240614090104863" width="700" />

**本人尝试对NSH模型进行复现，但代码有问题，模型并未收敛。**

## 模型效果

在论文中NSH的测试效果如下：

<img src=\images\2.png alt="image-20240614090104863" width="700" />

<img src=\images\3.png alt="image-20240614090145253" width="700" />

<img src=\images\4.png alt="image-20240614090211674" width="700" />

## 数据集

**cifar10有三种不同的配置**

- config[“dataset”]=“cifar10”将使用1000个图像（每个类100个图像）作为查询集，5000个图像（每类500个图像）用作训练集，其余54000个图像用作数据库。
- config[“dataset”]=“cifar10-1”将使用1000个图像（每个类100个图像）作为查询集，其余59000个图像用作数据库，5000个图像（每类500个图像）从数据库中随机采样作为训练集。
- config[“dataset”]=“cifar10-2”将使用10000个图像（每个类1000个图像）作为查询集，50000个图像（每类5000个图像）用作训练集和数据库。

你可以在[这里](https://github.com/TreezzZ/DSDH_PyTorch)下载NUS-WIDE，它使用data/nus-wide/code.py进行划分，每个类随机选择100幅图像作为查询集（共2100幅图像）。剩余的图像被用作数据库集，我们从中每个类随机采样500个图像作为训练集（总共10500个图像）。

你可以在[这里](https://github.com/thuml/HashNet)下载ImageNet、NUS-WIDE-m和COCO数据集，或者使用[百度云盘](https://pan.baidu.com)（密码：hash）。NUS-WIDE中有269648个图像，其中195834个图像分为21个常见类别。NUS-WIDE-m有223496个图像。

你可以在[这里](https://www.liacs.nl/~mirflickr)下载mirflickr，然后使用data/mirflickr/code.py划分，随机选择1000个图像作为测试查询集和4000个图像作为训练集。

## 如何运行

1. ### 配置运行环境

   在自己所使用的环境中安装依赖库。

   ```python
   pip install requirements.txt
   ```

2. ### 数据集

   若使用CIFAR数据集，则无需自己下载，若使用其他数据集，则将下载的数据集放置在dataset文件夹下。

3. ### 参数选择

   在train.py文件中修改运行参数。

   ```python
       configs = {
           "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
           "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
           "info": "NSH",
           "dataset": "mirflickr",
           "net": HashingModel,
           "resize_size": 224,
           "crop_size": 224,
           "batch_size": 50,
           "epoch": 200,
           "bit_list": [16, 32, 64],
           "positive_num": 2,
           "tau": 0.1,
           "num_workers": 4,
           "test_map": 1,
           "logs_path": "results",
       }
   ```

4. ### 运行

   在终端输入命令：

   ```python
   python train.py
   ```

   

