import numpy as np
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool, freeze_support
import os


# functions for IOUs calculations
def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) > 0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou
    
def task(i, j, curr_masks_shape, next_masks_shape):
    curr_sm = SharedMemory('curr_masks_memory')
    next_sm = SharedMemory('next_masks_memory')

    task_curr_sm = np.ndarray(curr_masks_shape, dtype=np.int32, buffer=curr_sm.buf)
    task_next_sm = np.ndarray(next_masks_shape, dtype=np.int32, buffer=next_sm.buf)
    result = calculate_iou(task_curr_sm[i], task_next_sm[j])
    del task_next_sm
    del task_curr_sm

    curr_sm.close()
    next_sm.close()

    return result


def multi_get_IOU(curr_masks, next_masks):
    freeze_support()
    curr_masks = np.array(curr_masks, dtype=np.int32)
    curr_masks_shape = curr_masks.shape
    next_masks = np.array(next_masks, dtype=np.int32)
    next_masks_shape = next_masks.shape

    curr_sm = SharedMemory(name='curr_masks_memory', create=True, size=curr_masks.nbytes)
    next_sm = SharedMemory(name='next_masks_memory', create=True, size=next_masks.nbytes)

    # create a new numpy array that uses the shared memory
    curr_data = np.ndarray(curr_masks_shape, dtype=np.int32, buffer=curr_sm.buf)
    curr_data[:] = curr_masks

    next_data = np.ndarray(next_masks_shape, dtype=np.int32, buffer=next_sm.buf)
    next_data[:] = next_masks

    # # create a child process
    with Pool() as pool:
        results=pool.starmap(task, [(i, j, curr_masks_shape, next_masks_shape) for i in range(curr_masks_shape[0]) for j in range(next_masks_shape[0])])

    # close the shared memory and release the shared memory
    curr_sm.close()
    next_sm.close()

    curr_sm.unlink()
    next_sm.unlink()

    return np.array(results).reshape(curr_masks_shape[0], next_masks_shape[0])