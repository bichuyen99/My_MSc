import numpy as np
from mpi4py import MPI

# functions for IOUs calculations
def calculate_iou(gt_mask, pred_mask):
    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask) > 0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou

def mpi_get_IOU(curr_masks, next_masks):
    '''
    Create process for every IOU(mask_1, mask_2) and gather all results into matrix of IOUs

    Input: list of np.arrays
    Output: np.array (len(curr_masks) * len(next_masks))
    '''
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    IOUs = np.zeros((len(curr_masks), len(next_masks)))

    # Divide the work among processes
    chunk_size = len(curr_masks) // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < size - 1 else len(curr_masks)

    for i in range(start, end):
        mask1 = (curr_masks[i] > 0).astype(int)
        for j, mask2 in enumerate(next_masks):
            mask2 = (mask2 > 0).astype(int)
            iou = calculate_iou(mask1, mask2)
            IOUs[i][j] = iou

    # Gather results from all processes
    all_IOUs = comm.gather(IOUs, root=0)

    if rank == 0:
        # Combine results from all processes
        final_IOUs = np.vstack(all_IOUs)
        return final_IOUs
