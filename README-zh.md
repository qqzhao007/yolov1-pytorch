# yolov1-pytorch(简体中文)

本仓库是基于pytorch的yolov1，可以视为论文的复现

* 本仓库参考了 https://github.com/ivanwhaf/yolov1-pytorch
  特此表示感谢
* Yolov1论文网址: [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640v5.pdf)

# Demo

## Train

在终端运行 `inference.py` 文件，可以指定部分参数

```bash
$ python inference.py -w weights/last.pth --source 0  # if 0, 使用摄像头作为图像输入
```
![Alt text](data/samples/test_predict.jpg)

# Usage

## Preparation

* 1.新建一个 `dataset` 文件夹用来存储训练、验证、测试数据
* 2.`dataset` 文件夹内部应该如下所示:
```
dataset/{dataset name}/
  ├──images
  |   ├──001.png
  |   ├──002.png
  |   └──003.jpg
  └──labels 
      ├──001.txt
      ├──002.txt
      └──003.txt
```
* 3.生成 '.txt' 文件

```
python modify_label.py
```
在运行上述命令前，你需要在`modify_label.py`中设置相关文件路径


## Train 

* 1.编辑 `cfg/yolov1.yaml` 配置文件, 设置相关参数
* 2.编辑 `cfg/dataset.yaml` 配置文件, 设置相关文件路径
* 3.如有需要，可以做`train.py` 中修改相关参数比如 **epochs** 等
* 4.运行 `train.py` 训练模型

```bash 
$ python train.py 
```

![!\[image\](https://github.com/ivanwhaf/yolov1-pytorch/blob/master/data/batch0.png)](data/batch0.png)

## Cautions

* 验证和测试过程均在`train.py` 文件中

## Program Structure Introduction

* cfg: 配置文件
* data: 部分图片以供readme显示使用
* dataset: 数据集
* models: 模型配置文件
* utils: 关键函数
* output: 输出文件夹
* weights: 模型权重

# Requirements

Python 3.X 

```bash
$ pip install -r requirements.txt
```
