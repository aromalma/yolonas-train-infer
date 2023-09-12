def get_names(path):
    with open(path, 'r') as f:
        t = f.read().split("\n")
    names = [x for x in filter(lambda i: bool(len(i)), t)]
    print(names)
    return names


def transform_config(dim):
    t = [
    {'DetectionMosaic':{'input_dim': dim, "prob":0.5 ,"enable_mosaic":False,'border_value':114 }},
                       {'DetectionRandomAffine': {'degrees': 6, 'translate': 0.05, 'scales': [0.7, 1.2], 'shear': 0.5,
                                                  'target_size': dim, 'filter_box_candidates': True, 'wh_thr': 2,
                                                  'area_thr': 0.1, 'ar_thr': 20}},
    #                    {'DetectionRandomAffine': {'degrees': 0.0, 'translate': 0.1, 'scales': [.7,1.3], 'shear': 0,
    #                                               'target_size': dim }},
                       # {'DetectionMixup': {'input_dim': dim, 'mixup_scale': [0.7, 1.3], 'prob': 0.2, 'flip_prob': 0.2}},
                       {'DetectionHSV': {'prob': 0.3, 'hgain': 5, 'sgain': 6, 'vgain': 6}},
                       # {'DetectionHorizontalFlip': {'prob': 0}},
                       {'DetectionPaddedRescale': {'input_dim': dim, 'pad_value': 114}},
                       {'DetectionTargetsFormatTransform': {'input_dim': dim, 'output_format': 'LABEL_CXCYWH'}}
        ]


    return t


config = {

    'model': 'yolo_nas_m',  # model
    'CHECKPOINT_DIR': 'checkpoints',
    'experiment_name': 'test',  # name of experiment

    'dataset_params': {
        'data_dir': '/home/groot/Aromal/yolonas-train-infer/nn/yolonas_train/',  # root of train and test folder
        'train_images_dir': 'train/',
        'train_labels_dir': 'train/',
        'test_images_dir': 'test/',
        'test_labels_dir': 'test/',
        'classes': get_names('/home/groot/Aromal/classes.txt'),  # path to classes.txt or obj.names
        'input_dim': [352, 640],
        'transforms': transform_config([352, 640])

    },
    'dataloader_params': {
        'batch_size': 8,  # adjust based on your GPU
        'num_workers': 4,
    },
    'train_parms': {
        'silent_mode': False,
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 10,
        "mixed_precision": True,
    }
}
