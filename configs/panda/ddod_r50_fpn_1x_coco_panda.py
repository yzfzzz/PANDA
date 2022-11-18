model = dict(
    type='DDOD',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/home/yezifeng/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='DDODHead',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        # assigner is mean cls_assigner
        assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
        reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

dataset_type = 'CocoDataset'
classes = ('person','visible car',)
cur_path = '/home/yezifeng/PANDA/mmdetection/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # 在线裁剪
    dict(type='GtBoxBasedCrop', crop_size=(3000,3000)),
    dict(type='Resize', img_scale=(1500, 1500), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1500, 1500),
        flip=False,
        flip_direction=['horizontal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=[
        dict(
        type=dataset_type,
        img_prefix=cur_path + 'data/panda_data/panda_round1_train_202104',
        classes=classes,
        ann_file=cur_path + 'data/panda_data/panda_round1_coco_full.json',
        pipeline=train_pipeline),
    ],
    val=dict(
        type=dataset_type,
        img_prefix=cur_path + 'data/panda_data/panda_round1_train_202104',
        classes=classes,
        ann_file=cur_path + 'data/panda_data/panda_round1_coco_full.json',
        pipeline=train_pipeline),
    test=dict(
        samples_per_gpu=5,
        type=dataset_type,
        img_prefix=cur_path + 'data/panda_data/panda_round1_test_202104_A_patches_3000_3000',
        classes=classes,
        ann_file=cur_path + 'data/panda_data/panda_round1_coco_full_patches_wh_3000_3000_testA.json',
        pipeline=test_pipeline))

evaluation = dict(interval=120, metric='bbox', classwise=True)

optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[80, 110])
total_epochs = 120

checkpoint_config = dict(interval=10)

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = cur_path + 'checkpoints/ddod_r50_fpn_1x_coco_20220523_223737-29b2fc67.pth'
resume_from = None
workflow = [('train', 1)]

