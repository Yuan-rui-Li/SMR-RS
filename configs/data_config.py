data_root = 'data/coco/'
data = dict(
    samples_per_gpu=7,
    workers_per_gpu=7,
    train=dict(
        type='CocoDataset',
        ann_file=data_root+'annotations/instances_val2017.json',
        img_prefix=data_root+'images/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])]
        )
)
gpu_ids = [0]