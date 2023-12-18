
from mrcnn.config import Config

from helpers import build_dataset, create_model

from datetime import datetime

from imgaug import augmenters as augs

class TrainingConfig(Config):
    NAME = "training"

    LEARNING_RATE=0.0015
    LEARNING_MOMENTUM = 0.85
    
    DETECTION_MIN_CONFIDENCE = 0.75

    STEPS_PER_EPOCH = 259
    NUM_CLASSES = 1 + 51

    VALIDATION_STEPS = 50

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [1, 1, 1.732]

    TRAIN_ROIS_PER_IMAGE = 150

SETS_PATH = "./dataset"

train_set = build_dataset("train", SETS_PATH)
val_set = build_dataset("val", SETS_PATH)

COCO_WEIGHTS_PATH = './mask_rcnn_coco.h5'

config = TrainingConfig()

model = create_model('training', config, COCO_WEIGHTS_PATH)

print('Model training starting.')

model.train(
    train_set, 
    val_set, 
    learning_rate=config.LEARNING_RATE, 
    epochs=100, 
    layers='heads', 
    augmentation=augs.SomeOf(
        2,
        [
            augs.Affine(rotate=(-45, 45)),
            augs.Affine(rotate=(-90, 90)),
            augs.MultiplyBrightness((0.5, 1.5)),
            augs.Sharpen(alpha=0.5)
        ]
    )
)

print('Model training completed.')

MODEL_WEIGHTS_PATH = f'./signal_plates_mask_rcnn_{datetime.now().strftime("%Y-%m-%dT%H-%M")}.h5'
model.keras_model.save_weights(MODEL_WEIGHTS_PATH)

print(f"Model weights saved at '{MODEL_WEIGHTS_PATH}'.")