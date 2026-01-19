


# MAP-Net config for DA-Net dataset (professional, publication-ready)
_base_ = [
    '../_base_/default_runtime.py',
    './mapnet_runtime.py',
]

checkpoint_config = dict(interval=1)
total_iters = 40000  # Set to your desired number of iterations

exp_name = 'mapnet_danet_dataset_40k'

model = dict(
    type='MAP',
    generator=dict(
        type='MAPNet',
        backbone=dict(
            type='ConvNeXt',
            arch='tiny',  # Closest to DA-Net's small model
            out_indices=[0, 1, 2, 3],
            drop_path_rate=0.0,
            layer_scale_init_value=1.0,
            gap_before_final_norm=False,
            # No pretrained weights for strict comparison
            init_cfg=None,
        ),
        neck=dict(
            type='ProjectionHead',
            in_channels=[96, 192, 384, 768],
            out_channels=64,
            num_outs=4
        ),
        upsampler=dict(
            type='MAPUpsampler',
            embed_dim=32,
            num_feat=32,
        ),
        channels=32,
        num_trans_bins=32,
        align_depths=(1, 1, 2, 1),  # Match kernel_sizes length
        num_kv_frames=[1, 2, 3],
    ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
)

data = dict(
    workers_per_gpu=6,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2),

    train=dict(
        type='HWFolderMultipleGTDataset',
        lq_folder='dataset/RSID/train/hazy',
        gt_folder='dataset/RSID/train/GT',
        ann_file=None,  # If you have annotation files, set here
        num_input_frames=5,
        pipeline=[
            dict(type='LoadImageFromFileList', io_backend='disk', key='lq', flag='unchanged'),
            dict(type='LoadImageFromFileList', io_backend='disk', key='gt', flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='Normalize', keys=['lq'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
            dict(type='Normalize', keys=['gt'], mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True),
            dict(type='PairedRandomCrop', gt_patch_size=256),
            dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='horizontal'),
            dict(type='FramesToTensor', keys=['lq', 'gt']),
            dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
        ],
        test_mode=False),
    val=dict(
        type='HWFolderMultipleGTDataset',
        lq_folder='dataset/RSID/test/hazy',
        gt_folder='dataset/RSID/test/GT',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFileList', io_backend='disk', key='lq', flag='unchanged'),
            dict(type='LoadImageFromFileList', io_backend='disk', key='gt', flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='Normalize', keys=['lq'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
            dict(type='Normalize', keys=['gt'], mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True),
            dict(type='FramesToTensor', keys=['lq', 'gt']),
            dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
        ],
        test_mode=False),
    test=dict(
        type='HWFolderMultipleGTDataset',
        lq_folder='dataset/RSID/test/hazy',
        gt_folder='dataset/RSID/test/GT',
        ann_file=None,
        pipeline=[
            dict(type='LoadImageFromFileList', io_backend='disk', key='lq', flag='unchanged'),
            dict(type='LoadImageFromFileList', io_backend='disk', key='gt', flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(type='Normalize', keys=['lq'], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True),
            dict(type='Normalize', keys=['gt'], mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True),
            dict(type='FramesToTensor', keys=['lq', 'gt']),
            dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
        ],
        test_mode=True)
)

work_dir = f'./work_dirs/{exp_name}'
