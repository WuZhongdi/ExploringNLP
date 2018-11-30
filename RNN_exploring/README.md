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
### Functions方法:
#####   name方法名             parameters参数                  function这个函数的作用
#####   __init__               None                             initializing初始化
#####   load_text              path                             read text with given path 从给定路径读取文本
#####   set_parameters         optional paramters很多可选参数    set all the parameters 设定全部参数
#####   pre_process            None                             pre processings:encode,puncs... 预处理
#####   get_inputs                                              used in train 在train方法中使用
#####   get_init_cell                                           used in train 在train方法中使用
#####   get_embed                                               used in train 在train方法中使用
#####   build_rnn                                               used in train 在train方法中使用
#####   build_nn                                                used in train 在train方法中使用
#####   get_batch                                               used in train 在train方法中使用
#####   train                  None                             train and save RNN model 训练和保存RNN模型

### Paramters of load_text参数:
##### name参数名                meaning含义
##### Path                      giving the path of your novel给定你的待学习文本的路径

### Optional parameters of set_parameters 可选参数
##### name参数名       default默认值          meaning含义
##### word_num          100000                num of word used for training 选取的学习文本长度（字）
##### epochs            200                   epochs of learning 学习轮次
##### batch_size        128                   size of batch batch的大小
##### RNN_size          256                   num of RNN cells RNN神经元数
##### embedding_size    256                   size of embedding 词向量维度
##### seq_len           32                    step of reading 学习步长
##### learning_rate     0.01                  learning rate 学习速率
##### output_rate       5                     frequency of outputing loss loss每多少次更新输出一次
##### layer_num         3                     num of layers 网络层数
##### dropout_keep_rate 0.8                   rate of connections kept during dropout dropout过程中保留比例

## 4、Parameters and Functions in RNN_writing.py 参数和方法
### Functions方法:
#####   name方法名             parameters参数                  function这个函数的作用
#####   __init__               None                             initializing初始化
#####   preparing              None                             load the saves 载入保存
#####   get_tensors                                             used in get_novel 在get_novel方法中使用
#####   choose_word                                             used in get_novel 在get_novel方法中使用  
#####   get_novel             start_word optional parameters    write new novels 写出新的文章

### Paramters of get_novel参数:
##### name参数名                meaning含义
##### start_word                start word of new novel 给定新文章的起始词

### Optional parameters of get_novel 可选参数
##### name参数名       default默认值          meaning含义
##### novel_len         500                   length of new novel 给定新写的文本的长度
##### end_at_punc       True                  whether to stop at "。" 是否在句号处停止
