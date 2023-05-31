import glob
import subprocess

from roboflow import Roboflow

from super_gradients.training import dataloaders, models, Trainer
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


def download_and_clear_dataset():
    rf = Roboflow(api_key="zwrl5fczBzjIyHK9WDL0")
    project = rf.workspace("squirrels-lyyup").project("hero_squirrels")
    datase = project.version(4).download("yolov5")

    # change labels
    for sub in ["train", "valid", "test"]:
        parent_dir = f"Hero_Squirrels-4/{sub}/labels/*.txt"
        lst = glob.glob(parent_dir)
        for filename in lst:
            command = f"sed -i 's/^0\ /14\ /' {filename}"
            subprocess.run(command, shell=True)
            command = f"sed -i 's/^1\ /80\ /' {filename}"
            subprocess.run(command, shell=True)


def train():
    dataset_params = {
        'data_dir':'Hero_Squirrels-4',
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'valid/images',
        'val_labels_dir':'valid/labels',
        'test_images_dir':'test/images',
        'test_labels_dir':'test/labels',
        'classes': [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush', 'squirrel'],
         'input_dim': [480, 480]
    }

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    # train_data.dataset.plot()

    model = models.get("yolo_nas_s", num_classes=len(dataset_params['classes']), pretrained_weights="coco")

    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': True,
        "average_best_models":True,
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
        # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
        "max_epochs": 100,
        "mixed_precision": False,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    trainer = Trainer(experiment_name="yolo_nas_squirrel", ckpt_root_dir="ckpt")
    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)


if __name__ == "__main__":
    #download_and_clear_dataset()
    train()
