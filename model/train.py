
from mrcnn.config import Config

from helpers import build_dataset, create_model

from datetime import datetime

from imgaug import augmenters as augs

class TrainingConfig(Config):
    NAME = "training"

    LEARNING_RATE=0.002
    LEARNING_MOMENTUM = 0.8
    
    WEIGHT_DECAY = 0.0001
    
    DETECTION_MIN_CONFIDENCE = 0.8

    STEPS_PER_EPOCH = 180
    NUM_CLASSES = 1 + 51

    VALIDATION_STEPS = 50

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    TRAIN_ROIS_PER_IMAGE = 300

    # ================= OLD =================

    # # number of training steps per epoch
    # STEPS_PER_EPOCH = 160
    # # learning rate and momentum
    # LEARNING_RATE=0.002
    # LEARNING_MOMENTUM = 0.8

    # # regularization penalty
    # WEIGHT_DECAY = 0.0001

    # # image size is controlled by this parameter
    # IMAGE_MIN_DIM = 512

    # # validation steps
    # VALIDATION_STEPS = 50

    # # number of Region of Interest generated per image
    # Train_ROIs_Per_Image = 200

    # # RPN Acnhor scales and ratios to find ROI
    # RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)
    # RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]
    # number of training steps per epoch

SETS_PATH = "/content/drive/MyDrive/TrabalhoA3"

train_set = build_dataset("train", SETS_PATH)
val_set = build_dataset("val", SETS_PATH)

COCO_WEIGHTS_PATH = SETS_PATH + '/mask_rcnn_coco.h5'

config = TrainingConfig()

model = create_model('training', config, COCO_WEIGHTS_PATH)

print('Model training starting.')

model.train(
    train_set, val_set, 
    learning_rate=config.LEARNING_RATE, 
    epochs=15, 
    layers='heads', 
    augmentation=augs.SomeOf(
        2, 
        [
            augs.Affine(rotate=(-45, 45)),
            augs.MultiplyBrightness((0.5, 1.5)),
            augs.Add(50, per_channel=True),
            augs.Sharpen(alpha=0.5)
        ],
        random_order=True
    )
)

print('Model training completed.')

MODEL_WEIGHTS_PATH = SETS_PATH + f'/signal_plates_mask_rcnn_{datetime.now().strftime("%Y-%m-%dT%H-%M")}.h5'
model.keras_model.save_weights(MODEL_WEIGHTS_PATH)

print(f"Model weights saved at '{MODEL_WEIGHTS_PATH}'.")