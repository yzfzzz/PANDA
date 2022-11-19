# Gigavision2022

![image-20221107164559486](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221107164559486.png)

## 实验条件

- 每一个模型均采用resnet50为主干网络，均在COCO数据集上进行训练

- 最大训练epoch=120，batch_size不尽相同，取决于显存大小

- 优化器基本相似

  ```python
  optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0001)
  ```

- 学习率调整策略均相同

  ```python
  lr_config = dict(
      policy='step',
      warmup='linear',
      warmup_iters=500,
      warmup_ratio=0.001,
      step=[80, 110])
  ```

​		![image-20221119170455289](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221119170455289.png)

- 推理阶段，均采用最后一轮模型测试

## 实验结果

| 模型                                 | AP       | AR@500     | score     | COCO-boxAP | 日期  | 备注     |
| ------------------------------------ | -------- | ---------- | --------- | ---------- | ----- | -------- |
| cascade_rcnn_r50_fpn_1x_coco         | ==0.48== | 0.5705     | ==0.521== | 40.3       | 10.30 | baseline |
| cascade_rcnn_X-101-64x4d_FPN_1x_coco | 0.41     | 0.5242     | 0.461     | ==44.7==   | 11.04 |          |
| deformable_detr_r50_16x2_50e_coco    | 0.34     | 0.5314     | 0.418     | 44.5       | 11.07 |          |
| tood_r50_fpn_1x_coco                 | 0.46     | ==0.5720== | 0.511     | 42.4       | 11.18 | 2021     |
| vfnet_r50_fpn_1x_coco                | 0.44     | 0.5502     | 0.489     | 41.6       | 11.19 | 2021     |
| ddod_r50_fpn_1x_coco                 | 0.46     | 0.5587     | 0.503     | 41.7       | 11.19 | 2021     |
|                                      |          |            |           |            |       |          |

## 更新日志

- [x] 复现baseline(10.30)
- [x] 撰写baseline复现文档(11.1)
- [x] 划分验证集、测试集(11.7)
- [ ] 根据官方要求设置测试指标
- [ ] 测试`deformable_detr_r50_16x2_50e`模型，**使之达到最优效果**
- [x] 测试`cascade_rcnn_X-101-64x4d_FPN`(11.11)
- [x] 测试`tood_r50_fpn_1x_coco`(11.18)
- [x] 测试`vfnet_r50_fpn_1x_coco`(11.19)
- [x] 测试`ddod_r50_fpn_1x_coco`(11.19)
- [ ] 修复`GtBoxBasedCrop()`函数错误，以测试验证集

## 遇到的问题

### 无法验证模型

[Evaluation interval VS validation interval · Issue #6380 · open-mmlab/mmdetection (github.com)](https://github.com/open-mmlab/mmdetection/issues/6380)

[Questions about workflow and evaluation when setting up validation operations · Issue #9351 · open-mmlab/mmdetection (github.com)](https://github.com/open-mmlab/mmdetection/issues/9351)

