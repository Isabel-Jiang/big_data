本项目是基于airbnb短租数据集的定价因素分析和文本挖掘。

# 数据

## 数据集

数据集来源见提交的论文。

其中我们所使用的数据以及其处理结果在data文件夹中。

论文中展示数据的代码在visualization_of_dataset.py中。

## 预处理

预处理过程在preprocess.py中。



# 挖掘算法

## 价格建模

价格建模及其预处理在classification.py中。

## 租金与地区的关系及其原因

租金与地区的关系可视化在location.py。

租金与地区和房间类型的可视化在room_type.py。

房间名称的分词挖掘在name_neighbor.py。

## 评论文本挖掘

评论文本先翻译成了英语，使用了google translate api，代码在translate.py

评论文本挖掘使用了google官方提供的代码，在bert_master文件夹中。

其中，我们的工作在00_our_model_use_bert.py中，其中借用了部分官方提供的代码，分为模型训练和使用两部分。

