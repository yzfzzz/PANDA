![image-20221107164559486](https://yzfzzz.oss-cn-shenzhen.aliyuncs.com/image/image-20221107164559486.png)

## 实验结果

| 模型                                 | AP   | AR500  | score | 日期  | 备注                 |
| ------------------------------------ | ---- | ------ | ----- | ----- | -------------------- |
| cascade_rcnn_r50_fpn_1x_coco         | 0.47 | 0.57   | 0.514 | 10.30 |                      |
| cascade_rcnn_X-101-64x4d_FPN_1x_coco | 0.41 | 0.5242 | 0.461 | 11.04 | 可能存在过拟合的现象 |
| deformable_detr_r50_16x2_50e_coco    | 0.34 | 0.5314 | 0.418 | 11.07 |                      |

## 更新日志

- [x] 复现baseline(10.30)
- [x] 撰写baseline复现文档(11.1)
- [x] 划分验证集、测试集(11.7)
- [x] 成功实现训练、验证流程(11.8)
- [ ] 根据官方要求设置测试指标
- [ ] 测试`deformable_detr_r50_16x2_50e`模型，使之达到最优效果
- [ ] 测试`cascade_rcnn_X-101-64x4d_FPN`

