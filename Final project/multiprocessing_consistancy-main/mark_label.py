import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from multiprocess.no_multi import get_IOU, calculate_iou
from multiprocess.multi_module import multi_get_IOU
from multiprocess.mpi4py_module import mpi_get_IOU
from multiprocess.cupy_module import cupy_get_IOU
import time


# preprocessing
def load_masks(name):
    next_frame = ANNOTATIONS[name]
    next_mask_path = [mask['segmentation'] for mask in next_frame]
    next_masks = [np.load(os.path.join(INPUT_PATH, mask_path)) for mask_path in next_mask_path]
    return next_masks

def get_iou_function(multiprocess_type):
    if multiprocess_type=='no':
        return get_IOU
    
    elif multiprocess_type=='multi':
        return multi_get_IOU
    
    elif multiprocess_type=='mpi':
        return mpi_get_IOU

    elif multiprocess_type=='cupy':
        return cupy_get_IOU
    

def remove_duplicates(masks, frame_name, get_IOU_function):
    dup_ious = get_IOU_function(masks, masks)
    dup_ious = dup_ious - np.eye(*dup_ious.shape)

    rows_nonzero, col_nonzero = np.nonzero(dup_ious > 0.5)
    unique_masks = np.arange(dup_ious.shape[1]).tolist()

    for row, col in zip(rows_nonzero, col_nonzero):
        if (row in unique_masks) and (col in unique_masks):
            unique_masks.remove(col)

    uni_masks = [mask for ind, mask in enumerate(masks) if ind in unique_masks]
    ANNOTATIONS[frame_name] = [val for ind, val in enumerate(ANNOTATIONS[frame_name]) if ind in unique_masks]

    del dup_ious
    del masks

    return uni_masks



# lebel generator
def label():
    mark_label = 2
    while  True:
        yield mark_label
        mark_label+=1


# label first frame
def initialize(first_frame_masks, labels_gen):
    initializes_frame = []
    while len(first_frame_masks) > 0:
        mask = first_frame_masks.pop(0)
        mark = next(labels_gen)
        initializes_frame.append(mark * mask)

    return initializes_frame


#  given current frame label next frame
def make_frame_consistent(IOUs, curr_masks, next_masks, labels_gen, threshold=0.3):

    consistensy_index = IOUs.argmax(axis=1)
    for m1_ind, m2_ind in enumerate(consistensy_index):
        
        scale_condition = pow(((curr_masks[m1_ind] > 0).sum() / next_masks[m2_ind].sum() - 1), 2)

        marked_curr_mask = curr_masks[m1_ind]
        next_mask_to_mark = next_masks[m2_ind].astype(int)

        if scale_condition < threshold:
            curr_object_label = np.max(marked_curr_mask)
            next_masks[m2_ind] = curr_object_label * next_mask_to_mark
            del next_mask_to_mark

    del curr_masks # memory efficient
    
    index_new_objects = np.where(np.array(next_masks).max(axis=(1, 2)) == 1)[0]

    for ind in index_new_objects:
        mark = next(labels_gen)
        next_masks[ind] = mark * next_masks[ind]

    return next_masks


def save_masks(frame_name, frame_masks):
    masks_ann = ANNOTATIONS[frame_name]
    # path for numpy masks
    path_frame = os.path.join(SAVE_PATH, 'numpy_masks', frame_name.split('.')[0])
    os.makedirs(path_frame, exist_ok=True)
           
    for ind_mask, mask in enumerate(frame_masks):
        # save masks
        path_marked_mask = os.path.join(path_frame, str(ind_mask) +'.npy')
        np.save(path_marked_mask, mask)

        # change path to marked masks

        del masks_ann[ind_mask]['segmentation']
        masks_ann[ind_mask]['segmentation'] =  path_marked_mask

# parsing args
def parse_args():

    parser = argparse.ArgumentParser(description ='args for algorithm which makes frame consistant')

    parser.add_argument('--data-path', type=str, default='data',  help='Path to the data.')
    parser.add_argument('--exp-name', type=str, help='Here you can specify the name of the experiment.')
    parser.add_argument('--multi', type=str, default='no', help='Here you can specify the multiprocess type.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print('Multiprocess type: ', args.multi)
    
    INPUT_PATH = args.data_path
    SAVE_PATH = os.path.join('experiments', args.exp_name)
    OUTPUT_NAME = args.exp_name
    json_file = 'sk_masks.json'

    IOU_function = get_iou_function(args.multi)

    os.makedirs(SAVE_PATH, exist_ok=True)

    with open(os.path.join(INPUT_PATH, json_file)) as f:
        ANNOTATIONS = json.load(f)

    for key, frame in ANNOTATIONS.items():
        ANNOTATIONS[key] = sorted(frame, key=lambda x: x['area'], reverse=True)

    labels_gen = label()
    annotations = sorted(tuple(ANNOTATIONS.keys()))
    first_frame = annotations[0]

    current_masks = load_masks(first_frame)
    current_masks = remove_duplicates(current_masks, first_frame, IOU_function)

    current_masks = initialize(current_masks, labels_gen)
    save_masks(first_frame, current_masks)
    
    time_spend = []
    for next_frame in tqdm(annotations[1:]):
        next_masks = load_masks(next_frame)
        next_masks = remove_duplicates(next_masks, next_frame, IOU_function)

        start = time.time()
        IOUs = IOU_function(current_masks, next_masks)
        end = time.time()
        time_spend.append(end - start)

        current_masks = make_frame_consistent(IOUs, current_masks, next_masks, labels_gen)
        save_masks(next_frame, current_masks)

    with open(os.path.join(SAVE_PATH, f'{OUTPUT_NAME}.json') , 'w', encoding='utf-8') as f:
        json.dump(ANNOTATIONS, f, ensure_ascii=False, indent=4)

    # save time spend
    os.makedirs('running_time', exist_ok=True)
    with open(f'running_time/{OUTPUT_NAME}.json', 'w', encoding='utf8') as output:
        json.dump(time_spend, output)
