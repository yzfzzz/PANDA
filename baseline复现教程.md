# Gigavision挑战赛baseline复现教程

[TOC]

## 环境安装

1. 创建conda虚拟环境，原版baseline是python3.7，我用的是python3.8+pytorch1.11

2. 安装好python和pytorch环境

3. 安装mmcv-full，这个是mmdetection底层的支持包

   ```
   pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
   ```

4. 拷贝我处理好的PANDA文件夹，到自己的目录下

   ![image-20221031204418074](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031204418074.png)

5. 到mmdetection目录下，安装mmdetection其它库

   ```
   pip install-r requirements/build.txt
   pip install -v-e 或者 "python setup.py develop"
   ```

   我用的是`python setup.py develop`这个命令，另外一个好像不行



## 训练

mmdetection有以下几个部分

![image-20221031204657295](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031204657295.png)

- **首先修改一下mmdetection的两个文件，复制数据集到相应路径，下载好权重**

  原版baseline这些步骤我都帮你们处理好了，可以跳过这部分，放心使用

![image-20221031204755155](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031204755155.png)

- **然后，将官方==注释==转为coco格式，这个我也帮你们做了**

![image-20221031204849805](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031204849805.png)

- 去到`/home/yezifeng/PANDA/mmdetection/configs/panda`路径下，修改`cascade_rcnn_r50_fpn_1x_coco_round1_panda.py`文件

  定位到203行，修改为你的mmdection路径（把yezifeng换成你的名字就好啦）

  ![image-20221031210436702](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031210436702.png)

![image-20221031210337747](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031210337747.png)

还是在这个文件下，定位到264行的`lr=0.05`，这里lr=0.00125* gpu个数* samples_per_gpu，samples_per_gpu就是你的batchsize，比如我的GPU=8张，samples_per_gpu=5，那么lr=0.05，经过我的实验batchsize=5能最大地利用显存资源

![image-20221031210738808](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031210738808.png)

- **去到tools文件夹下，输入`nvidia-smi`，查看有多少张显卡空闲，因为我们要进行多卡训练，不然容易报错（8张TITAN RTX需要跑4个小时）**

  ![image-20221031205223562](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031205223562.png)

  ![image-20221031205600095](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031205600095.png)

- **输入以下命令，进行多卡训练**

  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  ./dist_train.sh ../configs/panda/cascade_rcnn_r50_fpn_1x_coco_round1_panda.py 8
  ```

  > `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`：使用的显卡号，对应上图的`GPU FAN`
  >
  > `8`：表示有8张卡训练

这里建议8卡一起跑，快速出结果~

需要经过漫长的等待......😪😪😪😭😭😭



## 测试

- **进入`data/panda_data`路径下，输入以下命令**

  ```
  python MulprocessApply.py
  ```

  ![image-20221031211303404](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031211303404.png)

​	注意这里要安装python包的话，统一用`pip install`，因为有一个库conda安装不了

- **推理测试集**

  ```
  ./dist_test.sh ../configs/panda/cascade_rcnn_r50_fpn_1x_coco_round1_panda.py work_dirs/cascade_rcnn_r50_fpn_1x_coco_round1_panda//epoch_120.pth 4 --format-only --options "jsonfile_prefix=../data/panda_data/panda_patches_results"
  ```

- **输入`python sloan_utils.py`命令，合成结果**

大功告成🎉🎉🎉



## 提交

新建一个`results`文件夹，下面有`det_results.json`文件，然后把这个results文件夹压缩成`zip格式`

![image-20221031212101266](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031212101266.png)

![image-20221031212159947](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031212159947.png)

提交官网上即可



## 代码修改对比

### 1.nms

| 修改前                                                       | 修改后                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221031212656196](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031212656196.png) | ![image-20221031212502606](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031212502606.png) |

### 2.Classes

| 修改前                                                       | 修改后                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221031213235990](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031213235990.png) | ![image-20221031213001264](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031213001264.png) |

### 3.bbox

| 修改前                                                       | 修改后                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20221031213442417](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031213442417.png) | ![image-20221031213409176](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031213409176.png) |

这个很重要，不然就会浪费一天的时间等待😭😭😭

<img src="https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031214059701.png" alt="image-20221031214059701" style="zoom:50%;" />

### 4.run

如果要把大图片的注释切分成小图片用于训练的话，就把第一部分代码打开，第二部分关闭

如果要把小图片的注释合成小图片用于提交的话，就把第二部分代码打开，第一部分关闭

![image-20221031213553773](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031213553773.png)

### 5.lr

将lr修改为适应GPU训练参数的数值

![image-20221031212743438](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221031212743438.png)

> 还有个地方是没引入一个包的，忘记在哪了，不过我的代码已经改好了，可以安心使用