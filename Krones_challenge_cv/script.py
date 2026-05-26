import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import torch

model_path = r'C:\Users\Bakwowi Junior\Documents\school-documents\4th-semester-SS-26\Computer Vision\project\Krones_challenge_cv\results\11th_trial\last.pt'

# Load YOLO model
model = YOLO(model_path)

# Load checkpoint metadata
ckpt = torch.load(model_path, weights_only=False, map_location='cpu')
print(ckpt.get('model'))

# print(f"Best checkpoint epoch : {ckpt.get('epoch')}")
# print(f"Best fitness (mAP)    : {ckpt.get('best_fitness')}")

# Model object inside checkpoint
model_obj = ckpt.get('model')
# print(f"Model object type     : {type(model_obj)}")
# Model type e.g., 'YOLOv8n', 'YOLOv8s', etc.#
# print(f"Model name       : {model.model_name}")
# Model type
# print(f"Model type            : {type(model_obj).__name__}")

# # Task type
# print(f"Task                  : {model.task}")

# # Class names
# print(f"Classes               : {model.names}")

# # Number of classes
# print(f"Number of classes     : {len(model.names)}")