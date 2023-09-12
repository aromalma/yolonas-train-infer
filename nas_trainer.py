from super_gradients.training import Trainer
import torchvision.transforms as T
from super_gradients.training import Trainer
from super_gradients.training.metrics import Accuracy
from super_gradients.training import dataloaders
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, \
    coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import cv2
from opencv_draw_annotation import draw_bounding_box
import numpy as np
from config import config

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': config['dataset_params']['data_dir'],
        'images_dir': config['dataset_params']['train_images_dir'],
        'labels_dir': config['dataset_params']['train_labels_dir'],
        'classes': config['dataset_params']['classes'],
        'input_dim': config['dataset_params']['input_dim'],
        "transforms": config['dataset_params']["transforms"]
    },
    dataloader_params=config['dataloader_params']
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': config['dataset_params']['data_dir'],
        'images_dir': config['dataset_params']['test_images_dir'],
        'labels_dir': config['dataset_params']['test_labels_dir'],
        'classes': config['dataset_params']['classes'],
        'input_dim': config['dataset_params']['input_dim']
    },
    dataloader_params=config['dataloader_params']
)

print(".............TRAIN DATA TRANSFORMS..............")
# print(train_data.dataset)
print(train_data.dataset.transforms)

print()
train_data.dataset.plot()
test_data.dataset.plot()
model = models.get(config['model'],
                   num_classes=len(config['dataset_params']['classes']),
                   pretrained_weights="coco",
                   )

trainer = Trainer(experiment_name=config['experiment_name'], ckpt_root_dir=config['CHECKPOINT_DIR'])

config['train_parms']['loss'] = PPYoloELoss(
    use_static_assigner=False,
    # NOTE: num_classes needs to be defined here
    num_classes=len(config['dataset_params']['classes']),
    reg_max=16
)

config['train_parms']["valid_metrics_list"] = [
    DetectionMetrics_050(
        score_thres=0.1,
        top_k_predictions=300,
        # NOTE: num_classes needs to be defined here
        num_cls=len(config['dataset_params']['classes']),
        normalize_targets=True,
        post_prediction_callback=PPYoloEPostPredictionCallback(
            score_threshold=0.01,
            nms_top_k=1000,
            max_predictions=300,
            nms_threshold=0.7
        )
    )
]

config['train_parms']["metric_to_watch"] = 'mAP@0.50'

trainer.train(model=model,
              training_params=config['train_parms'],
              train_loader=train_data,
              valid_loader=test_data)

best_model = models.get('yolo_nas_m',
                        num_classes=len(config['dataset_params']['classes']),
                        checkpoint_path=f"{config['CHECKPOINT_DIR']}/{config['experiment_name']}/ckpt_best.pth").cuda()
trainer.test(model=best_model,
             test_loader=test_data,
             test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                    top_k_predictions=300,
                                                    num_cls=len(config['dataset_params']['classes']),
                                                    normalize_targets=True,
                                                    post_prediction_callback=PPYoloEPostPredictionCallback(
                                                        score_threshold=0.01,
                                                        nms_top_k=1000,
                                                        max_predictions=300,
                                                        nms_threshold=0.7)))
