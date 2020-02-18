
# NFFM+Microsoft Malware Prediction

这个项目主要用来记录练习Tensorflow2.0实现NFFM,并应用于实际数据集,编写网络和优化的过程.



NFM 论文地址: [https://arxiv.org/pdf/1511.06443.pdf](https://arxiv.org/pdf/1511.06443.pdf).  
PNN 论文地址: [https://arxiv.org/pdf/1611.00144.pdf](https://arxiv.org/pdf/1611.00144.pdf).  
xDeepFM 论文地址: [https://arxiv.org/pdf/1803.05170.pdf](https://arxiv.org/pdf/1803.05170.pdf).  
FGCNN 论文地址: [https://arxiv.org/pdf/1904.04447.pdf](https://arxiv.org/pdf/1904.04447.pdf). 

数据地址: [https://www.kaggle.com/c/microsoft-malware-prediction/data](https://www.kaggle.com/c/microsoft-malware-prediction/data). 

## 使用方式
   - 运行说明：需要先将数据下载在任意目录,先运行data_sample.py,因为数据量比较大,电脑内存较小,前期测试使用的是随机采样部分数据,不需要可以取消掉.运行完后,对生成的csv进行编码等处理,使用csv_to_tfrecord.py,这个脚本也是分块处理,内存占用很小,运行完生成tfrecord格式文件,利用这个格式的文件节省内存并加速训练.这两个文件的输入输出需要设置好路径.然后运行main.py文件即可，调用任意NN模型,按需设置参数,训练预测即可,模型默认保存在同一文件夹model里面,同时生成竞赛的结果csv文件.


## 更新记录
   - 2020-02-17 新增模型.  
        * 新增了PNN,xDeepFM模型,FGCN layer提取特征.目前来看，模型起到的优化效果有限，AUC表现上面看，NFFM>=xDeepFM>PNN_FGCN>PNN,但是数值也都在1个百分点以内。
      
   - 2020-02-10 网络优化.  
        * 测试NFFM,训练后提交结果LB auc 为0.674左右,然后将embedding和 Bi-Interaction concat后接入deep部分,增长了0.001,提升不大,继续尝试别的网络优化.
        
   - 2020-02-09 网络测试.  
        * 测试了不同的简单的特征编码方式，最终结果差别不大
        * 主要时间用来调整以下两个方面：
            * 网络结构：测试的时候发现，每次训练的过程中，网络几十个step就停止收敛了，而且收敛的效果很差，auc在0.6不到，看别人在别的竞赛数据集上应用的成绩很不错，所以仔细训练调试了几天.最后发现，在模型训练前面几个step，NFFM的浅层LR部分，每次很快就收敛了，LR部分的值特别大，并且基本上不变了，deep部分还没开始学习就不变了。所以经过多次测试，把LR的部分直接删掉了，然后训练完果然涨了7,8个点，收敛的情况也正常了，Loss下降比较平顺了。
            * 模型速度:尝试了两个方面，增加batch_size，优化脚本。前一种方式，受限于显卡内存，只能设置的比较小，而且tf2.0的版本感觉内存消耗特别大。上网查询后，尝试tf.float16精度，确实可以增大几十倍batch size，但是模型没法收敛，多方查询，貌似有bug，而且查了资料我这显卡本身也不包括在这个精度被支持的显卡范围内，以后换了计算力大于7.1的新显卡再试试。脚本基于tf2.0的版本编写，写的过程中发现确实bug很多，特别是内存和速度相对于1.1x烂了很多，后面发现2.0里面推荐tf.function装饰器来优化函数，把call函数用此装饰器声明，能够达到静态图的速度，所以后面试了下，确实速度相较不使用，一个step快了3-4倍，以后编写好以后记得使用此装饰器。
        
   - 2020-02-05 工程基本功能完成.  
        * 数据预处理脚本
        * 内存不足的情况下,编写了脚本,将csv数据转Tfrecord格式,保证有足够的内存运行程序.
        * NFFM网络基本结构,包括训练,训练,验证,预测等功能.
        * 模型效果测试,调试ing.......
