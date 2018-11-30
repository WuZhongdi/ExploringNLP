# RNN_Exploring
##### This is a project about RNN realization, which could learn Chinese novels and write novels with given start word.
##### 这是一个关于RNN实现的项目，它可以学习中文小说，并根据给定开始词进行写作。

## 1、File-explaining 文件解释
##### There are two python files RNN_writing_articles.py and RNN_writing.py(forgive my pool naming)
##### 工程包含两个python文件，RNN_writing_articles.py和RNN_writing.py（原谅我蹩脚的起名）
##### RNN_writing_articles.py is where the RNN_model is trained, and it would save the items neccessary for novels writing.
##### RNN模型主要在RNN_writing_articles.py 中训练，并会保存所有写新的文本所需要的材料。
##### However, RNN_writing.py takes the responsibility to write new novels with the items saved.(currently, the learned novel is 鲁迅全集)
##### RNN_writing.py 则承担根据保存的模型等写心得文本的任务。（当前学习的文本是鲁迅全集）
##### Text Learned is stored in "data", if you want to make the algorithm learning some other novels, make sure it is in "data".
##### 被学习的文本保存在data文件夹中，如果你想更换学习的文本，请把它保存在data文件夹中。（这是个好习惯，不是吗:)）
##### When you run RNN_writing_articles.py, it would make some files as well as a folder named "saves".
##### 当你运行 RNN_writing_articles.py 的时候，会生成几个文件和一个叫"saves"的文件夹。
##### If you want to train new models, change parameters, delete the items in "saves" or delete the folder "saves".
##### 如果你想训练新的模型，更改参数，请将saves中的文件删除或者删除saves文件夹。

## 2、How to use 如何使用
##### Run RNN_writing_articles.py, after you see"Model Training Done and Saved!", Run RNN_writing.py. (EZ, isn't it?)
##### 先运行RNN_writing_articles.py，当你看到"Model Training Done and Saved!"后，运行RNN_writing.py。（简单，不是吗？）

## 3、Parameters and Functions in RNN_writing_articles.py 参数和方法
#####
