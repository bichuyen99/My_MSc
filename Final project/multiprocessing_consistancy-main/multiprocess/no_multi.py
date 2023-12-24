import numpy as np


# functions for IOUs calculations
def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) > 0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou

def get_IOU(curr_masks, next_masks):
    IOUs = np.zeros((len(curr_masks), len(next_masks)))

    for i, mask1 in enumerate(curr_masks):
        for j, mask2 in enumerate(next_masks):

            mask1 = (mask1 > 0).astype(int)
            mask2 = (mask2 > 0).astype(int)

            iou = calculate_iou(mask1, mask2)
            # if iou > threshold:
            IOUs[i][j] = iou

    return IOUs
