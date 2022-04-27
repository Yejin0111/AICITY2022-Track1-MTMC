import os, sys, pdb
import numpy as np
import pickle
import random
from tqdm import tqdm


def is_valid_position(x, y, w, h, cam_id):
    """
    To judge if the position is out of the image

    Args:
        None
    """
    x_max = 1280
    y_max = 720 if cam_id == 45 or cam_id == 46 else 960
    x2 = x + w
    y2 = y + h
    if x < 0 or y < 0 or x2 >= x_max or y2 >= y_max:
        return False
    return True


def process_border(x, y, w, h, cam_id):
    x_max = 1280
    y_max = 720 if cam_id == 45 or cam_id == 46 else 960

    dw, dh = 0, 0
    if x < 1:
        dw = -x
        x = 1
    if y < 1:
        dh = -y
        y = 1
    x2, y2 = x + w, y + h
    w = x_max - x if x2 >= x_max else w - dw
    h = y_max - y if y2 >= y_max else h - dh
    return (x, y, w, h)


def get_static_10_color():
    """ For visualization
    """
    color_list = [[]] * 10
    color_list[0] = (255, 0, 0)
    color_list[1] = (0, 255, 0)
    color_list[2] = (0, 0, 255)
    color_list[3] = (255, 255, 0);
    color_list[4] = (255, 0, 255)
    color_list[5] = (0, 255, 255)
    color_list[6] = (128, 0, 0)
    color_list[7] = (0, 128, 0)
    color_list[8] = (0, 0, 128)
    color_list[9] = (138, 43, 226)
    return color_list


def get_color(num_color=365):
    """
    For visualization

    Args:
        num_color: The number of colors will be involved in visualization
    """
    num_ch = np.power(num_color, 1/3.)
    frag = int(np.floor(256/num_ch))
    color_ch = range(0, 255, frag)
    color_list = []
    for color_r in color_ch:
        for color_g in color_ch:
            for color_b in color_ch:
                color_list.append((color_r, color_g, color_b))
    random.shuffle(color_list)
    # color_list[0] = (255, 0, 0); color_list[1] = (0, 255, 0); color_list[2] = (0, 0, 255); color_list[3] = (255, 255, 0); color_list[4] = (138, 43, 226)
    return color_list

def track_file_format_transfer(src_root, dst_root):
    """ Transfer file format of track results.
    Single camera format (the direct output file of single camera algorithm) -> Multi camera format (the submission format)
    All files must be named as "c04x.txt"

    Args:
        src_root:
        dst_root:
    """
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    cam_list = os.listdir(src_root)
    cam_list.sort()
    for cam_file in cam_list:
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) # c04x.txt -> 4x
        dst_obj = open(os.path.join(dst_root, cam_file), 'w')
        f_dict = {}
        with open(os.path.join(src_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(float(x)) for x in line.rstrip().split(',')] # [frame_id, track_id, x, y, w, h, ...]
                x, y, w, h = s[2:6]
                if not is_valid_position(x, y, w, h, cam_id): # to drop those frames that are beyond of the image
                    x, y, w, h = process_border(x, y, w, h, cam_id)
                    if w <= 0 or h <= 0:
                        continue
                    s[2:6] = x, y, w, h
                fr_id = s[0]
                line = '{} {} {} {} -1 -1\n'.format(cam_id, s[1], s[0], ' '.join(map(str, s[2:6]))) # [camera_id, track_id, frame_id, x, y, w, h, -1, -1]
                if fr_id not in f_dict:
                    f_dict[fr_id] = []
                f_dict[fr_id].append(line)

            fr_ids = sorted(f_dict.keys())
            for fr_id in fr_ids:
                for line in f_dict[fr_id]:
                    dst_obj.write(line)
        dst_obj.close()

def load_feat_from_pickle(src_root, trun_root):
    """ This is the latest version.
    Load the original pickle files for the purpose of constructing the feature array for final matching.
    It is saved to Pickle format. 
    """
    feat_dict = {}
    fr_dict = {}
    res_trun_dict = {}
    src_list = sorted(os.listdir(src_root))
    trun_list = sorted(os.listdir(trun_root))
    # for src_file in tqdm(src_list):
    for src_file, trun_file in zip(src_list, trun_list):
        src_path = os.path.join(src_root, src_file)
        f_dict = pickle.load(open(src_path, 'rb'), encoding='latin1')
        trun_path = os.path.join(trun_root, trun_file)
        trun_dict = pickle.load(open(trun_path, 'rb'), encoding='latin1')
        for k, v in f_dict.items():
            if v is None or v.size < 30 or np.all(v == 0): # v.size < 30: to prevent some bad features
                continue
            s = k.split('_')
            track_id = int(s[-1])
            cam_id = int(s[0][1:])
            fr_id = int(s[1])

            if cam_id not in feat_dict:
                feat_dict[cam_id] = {}
                fr_dict[cam_id] = {}
                res_trun_dict[cam_id] = {}
            if track_id not in feat_dict[cam_id]:
                feat_dict[cam_id][track_id] = []
                fr_dict[cam_id][track_id] = []
                res_trun_dict[cam_id][track_id] = []
            feat_dict[cam_id][track_id].append(v)
            fr_dict[cam_id][track_id].append(fr_id)
            res_trun_dict[cam_id][track_id].append(trun_dict[k])
    return feat_dict, fr_dict, res_trun_dict
    
