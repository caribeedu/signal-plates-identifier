# import mask rcnn libraries
from mrcnn.config import Config
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# import numpy libraries
from numpy import expand_dims
from numpy import mean

# import helpers
from helpers import build_dataset, create_model

class EvaluationConfig(Config):
    NAME = "evaluation"
    NUM_CLASSES = 1 + 51
    DETECTION_MIN_CONFIDENCE = 0.85
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def compute_mean_avarage_precision(dataset, model, config):
    APs = list()
    for image_id in dataset.image_ids:

        image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, config, image_id)

        scaled_image = mold_image(image, config)

        sample = expand_dims(scaled_image, 0)

        r = model.detect(sample, verbose=1)[0]

        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])

        APs.append(AP)
    mAP = mean(APs)
    return mAP

SETS_PATH = "/content/drive/MyDrive/TrabalhoA3"

val_set = build_dataset("val", SETS_PATH)

MODEL_WEIGHTS_PATH = SETS_PATH + '/signal_plates_mask_rcnn.h5'

config = EvaluationConfig()

model = create_model('inference', config, MODEL_WEIGHTS_PATH)

mAP = compute_mean_avarage_precision(val_set, model, config)

print("Mean Avarage Precision on validation dataset: %.3f" % mAP)