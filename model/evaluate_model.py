from mrcnn.config import Config
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from sklearn import metrics

from numpy import expand_dims, mean, arange

from helpers import build_dataset, create_model

class EvaluationConfig(Config):
    NAME = "evaluation"
    NUM_CLASSES = 1 + 51
    DETECTION_MIN_CONFIDENCE = 0.8
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def compute_ar(pred_boxes, gt_boxes, list_iou_thresholds):
    AR = []
    for iou_threshold in list_iou_thresholds:

        try:
            recall, _ = compute_recall(pred_boxes, gt_boxes, iou=iou_threshold)

            AR.append(recall)

        except:
          AR.append(0.0)
          pass

    AUC = 2 * (metrics.auc(list_iou_thresholds, AR))
    return AUC

def evaluate_model(dataset, model, cfg, list_iou_thresholds=None):

  if list_iou_thresholds is None: list_iou_thresholds = arange(0.5, 1.01, 0.1)

  APs = []
  ARs = []
  for image_id in dataset.image_ids:
		
    image, _, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
		
    scaled_image = mold_image(image, cfg)
		
    sample = expand_dims(scaled_image, 0)
		
    result = model.detect(sample, verbose=0)[0]
		
    AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, result["rois"], result["class_ids"], result["scores"], result['masks'], iou_threshold=0.5)
		
    AR = compute_ar(result['rois'], gt_bbox, list_iou_thresholds)
    ARs.append(AR)
    APs.append(AP)

  mAP = mean(APs)
  mAR = mean(ARs)
  f1_score = 2 * ((mAP * mAR) / (mAP + mAR))

  return mAP, mAR, f1_score

SETS_PATH = "./dataset"

val_set = build_dataset("val", SETS_PATH)

MODEL_WEIGHTS_PATH = './signal_plates_mask_rcnn.h5'

config = EvaluationConfig()

model = create_model('inference', config, MODEL_WEIGHTS_PATH)

mAP, mAR, f1 = evaluate_model(val_set, model, config)

print("Mean Avarage Precision on validation dataset: %.3f" % mAP)
print("Mean Avarage Recall on validation dataset: %.3f" % mAR)
print("F1 Score on validation dataset: %.3f" % f1)