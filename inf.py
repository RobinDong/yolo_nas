import super_gradients

from super_gradients.training import models

model = models.get("yolo_nas_s", num_classes=2, checkpoint_path="ckpt/yolo_nas_squirrel/ckpt_best.pth")
print(model)
