import os, sys, pdb
import numpy as np
import pickle
import shutil
import argparse
from utils import track_file_format_transfer, load_feat_from_pickle
from tracklet import Tracklet

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_root', type=str, default="./data/track_results/", 
            help='the root path of single camera tracking results')
    parser.add_argument('--dst_root', type=str, default="./data/preprocessed_data/", 
            help='the root path of preprocessed results')
    parser.add_argument('--feat_root', type=str, default="./data/track_feats/", 
            help='the root path of features of each tracklet')
    parser.add_argument('--trun_root', type=str, default="./data/truncation_rates/", 
            help='the root path of truncation rates of each tracklet')
    return parser

def generate_all_preprocessed_track_info(cam_dict, dst_root):
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root)

    cam_list = []
    track_list = []
    in_out_info_obj = open(os.path.join(dst_root, 'in_out_all.txt'), 'w')
    for cam_id, tracklet_dict in cam_dict.items():
        for track_id, tracklet in tracklet_dict.items():
            assert (tracklet.cam_id == cam_id)
            cam_list.append(cam_id)
            track_list.append(track_id)
            in_out_info_obj.write('{} {} {} {} {} {}\n'.format(cam_id, track_id, tracklet.st_id, tracklet.en_id, 
                tracklet.frames[0], tracklet.frames[-1])) # [cam_id, track_id, start_id, end_id, start_frame, end_frame]
    in_out_info_obj.close()
    cam_arr = np.array(cam_list)
    track_arr = np.array(track_list)
    np.save(os.path.join(dst_root, 'cam_vec.npy'), cam_arr)
    np.save(os.path.join(dst_root, 'track_vec.npy'), track_arr)


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    # tmp_root = os.path.join('tmp', args.src_root.split('/')[-1])
    tmp_root = 'tmp'
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)
    os.makedirs(tmp_root)
    print ('* transfering the format of track files...')
    track_file_format_transfer(args.src_root, tmp_root)

    print ('* calculating the "in port" and "out port" of each tracklet...')
    cam_list = os.listdir(tmp_root)
    cam_list.sort()
    cam_dict = {}
    for cam_file in cam_list:
        print ('processing: {}'.format(cam_file))
        cam_id = int(cam_file[1:4]) # c04x.txt -> 4x
        tracklet_dict = {}
        with open(os.path.join(tmp_root, cam_file), 'r') as fid:
            for line in fid.readlines():
                s = [int(x) for x in line.rstrip().split()] # [cam_id, track_id, frame_id, x, y, w, h, -1, -1]
                c_id, track_id, fr_id, x, y, w, h = s[:-2]
                assert (c_id == cam_id)
                xc = (x + w / 2.)
                yc = (y + h / 2.)
                if s[1] not in tracklet_dict:
                    tracklet_dict[track_id] = Tracklet(c_id, xc, yc, fr_id, -1, 4) # initialized theta=-1 and angle id=4
                else:
                    tracklet_dict[track_id].add_element(xc, yc, fr_id)
        assert cam_id not in cam_dict
        cam_dict[cam_id] = tracklet_dict

    print ('* generating all preprocessed track information...')
    generate_all_preprocessed_track_info(cam_dict, args.dst_root)

    print ('* processing features...')
    feat_dict, fr_dict, trun_dict = load_feat_from_pickle(args.feat_root, args.trun_root)
    with open(os.path.join(args.dst_root, 'feat_all_vec.pkl'), 'wb') as fid:
        pickle.dump(feat_dict, fid)
    with open(os.path.join(args.dst_root, 'frame_all_vec.pkl'), 'wb') as fid:
        pickle.dump(fr_dict, fid)
    with open(os.path.join(args.dst_root, 'truncation.pkl'), 'wb') as fid:
        pickle.dump(trun_dict, fid)

