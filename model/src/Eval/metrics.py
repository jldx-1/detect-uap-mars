"""
metrics.py

This module provides functions for computing evaluation metrics for the UAP detection model.
It includes functions to compute Intersection over Union (IoU), average precision (AP),
and mean Average Precision (mAP). Note that this is a simplified implementation;
for production-level evaluation, consider using specialized libraries like pycocotools.
"""

import numpy as np

def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        boxA, boxB (list or array): Each box is defined as [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_average_precision(detections, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision (AP) for a single class.
    
    Args:
        detections (list): List of detections as tuples (image_id, confidence, box),
                           where box is [x1, y1, x2, y2].
        ground_truths (dict): Mapping from image_id to list of ground truth boxes.
        iou_threshold (float): IoU threshold to consider a detection as true positive.
    
    Returns:
        float: Average precision value.
    """
    # Sort detections by descending confidence score
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    tp = []
    fp = []
    # Count total ground truths per image
    total_gts = {image_id: len(boxes) for image_id, boxes in ground_truths.items()}
    
    # Record which ground truth boxes have been detected per image
    detected = {}
    
    for det in detections:
        image_id, confidence, pred_box = det
        gt_boxes = ground_truths.get(image_id, [])
        ious = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
        max_iou = max(ious) if ious else 0
        
        if max_iou >= iou_threshold:
            if image_id not in detected:
                detected[image_id] = [False] * len(gt_boxes)
            max_index = np.argmax(ious)
            if not detected[image_id][max_index]:
                tp.append(1)
                fp.append(0)
                detected[image_id][max_index] = True
            else:
                tp.append(0)
                fp.append(1)
        else:
            tp.append(0)
            fp.append(1)
    
    tp = np.array(tp)
    fp = np.array(fp)
    
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    
    recalls = cum_tp / (sum(total_gts.values()) + 1e-6)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-6)
    
    # Compute AP as the area under the precision-recall curve
    average_precision = np.trapz(precisions, recalls)
    return average_precision

def compute_mAP(detections_per_class, ground_truths_per_class, iou_threshold=0.5):
    """
    Computes mean Average Precision (mAP) over all classes.
    
    Args:
        detections_per_class (dict): Keys are class indices; values are lists of detections.
        ground_truths_per_class (dict): Keys are class indices; values are dictionaries
                                        mapping image_ids to ground truth boxes.
        iou_threshold (float): IoU threshold for determining true positives.
    
    Returns:
        float: The mean Average Precision.
    """
    aps = []
    for cls in detections_per_class:
        detections = detections_per_class[cls]
        ground_truths = ground_truths_per_class.get(cls, {})
        ap = compute_average_precision(detections, ground_truths, iou_threshold)
        aps.append(ap)
    mAP = np.mean(aps) if aps else 0.0
    return mAP

if __name__ == "__main__":
    # Quick test of compute_iou
    boxA = [50, 50, 150, 150]
    boxB = [100, 100, 200, 200]
    print("IoU:", compute_iou(boxA, boxB))
    
    # Dummy detections: (image_id, confidence, box)
    detections = [
        ("img1", 0.9, [50, 50, 150, 150]),
        ("img1", 0.8, [55, 60, 148, 152]),
        ("img2", 0.85, [30, 30, 120, 120])
    ]
    # Dummy ground truths: image_id -> list of boxes
    ground_truths = {
        "img1": [[50, 50, 150, 150]],
        "img2": [[30, 30, 120, 120]]
    }
    
    ap = compute_average_precision(detections, ground_truths)
    print("Average Precision:", ap)
    
    # For mAP, assuming a single class (class 0)
    detections_per_class = {0: detections}
    ground_truths_per_class = {0: ground_truths}
    mAP = compute_mAP(detections_per_class, ground_truths_per_class)
    print("mAP:", mAP)
