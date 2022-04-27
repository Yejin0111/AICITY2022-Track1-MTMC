import os, sys, pdb
import numpy as np
import cv2
import pickle
import argparse
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from rerank import re_ranking


# Those dicts are adopted to preserve all valid crossing vehicles. 
# For example: if one vehicle leaves port 1 at current camera, it should drive into port 1 at next camera.
in_valid_positive_directions  = ((1, 1), (1, 0), (1, 2))
out_valid_positive_directions = ((1, 1), (0, 1), (2, 1))
in_valid_negative_directions  = ((3, 3), (3, 0), (3, 2))
out_valid_negative_directions = ((3, 3), (0, 3), (2, 3))

# Those dicts are used to preserve the vehicles that must be in two successive cameras.
in_positive_direction_time_thre_dict  = {42: 550, 43: 180, 44: 440, 45: 240, 46: 360}
out_positive_direction_time_thre_dict = {41: 1111, 42: 1640, 43: 1610, 44: 1450, 45: 1610}
in_negative_direction_time_thre_dict  = {41: 600, 42: 350, 43: 560, 44: 290, 45: 760}
out_negative_direction_time_thre_dict = {42: 1240, 43: 1710, 44: 1520, 45: 1582, 46: 1430}

# The purpose of this dict is to preserve the vehicles whose travel time is possible between two ports.
# hard constraint for refine distance matrix
two_track_valid_pass_time_for_mask_dict = {(41, 42): [450, 1080], (42, 43): [130, 250], (43, 44): [340, 545], 
                                  (44, 45): [110, 635], (45, 46): [150, 900],
                                  (46, 45): [200, 730], (45, 44): [120, 700], (44, 43): [95, 1005], 
                                  (43, 42): [195, 530], (42, 41): [410, 570]}
# loose constraint for post-process
two_track_valid_pass_time_dict = {(41, 42): [300, 1500], (42, 43): [130, 800], (43, 44): [280, 680], 
                                  (44, 45): [50, 800], (45, 46): [80, 1000],
                                  (46, 45): [150, 850], (45, 44): [80, 800], (44, 43): [80, 1090], 
                                  (43, 42): [150, 600], (42, 41): [350, 850]}

# hyper parameters
args_params_dict = {(41, 42): {'topk': 13, 'r_rate': 0.5, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 0.8, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 2},
                    (42, 43): {'topk': 15, 'r_rate': 0.5, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 0.8, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 1},
                    (43, 44): {'topk': 7, 'r_rate': 0.8, 'k1': 13, 'k2': 8, 'lambda_value': 0.4, 
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 1},
                    (44, 45): {'topk': 15, 'r_rate': 0.5, 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'num_search_times': 1},
                    (45, 46): {'topk': 7, 'r_rate': 0.5, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 1},
                    (46, 45): {'topk': 7, 'r_rate': 0.3, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 0.8, 'long_time_t': 1000, 'short_time_t': 1000, 'num_search_times': 1},
                    (45, 44): {'topk': 15, 'r_rate': 0.5, 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'num_search_times': 1},
                    (44, 43): {'topk': 15, 'r_rate': 0.5, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 1},
                    (43, 42): {'topk': 7, 'r_rate': 0.8, 'k1': 13, 'k2': 5, 'lambda_value': 0.7, 
                     'alpha': 1.1, 'long_time_t': 500, 'short_time_t': 1000, 'num_search_times': 1},
                    (42, 41): {'topk': 15, 'r_rate': 0.5, 'k1': 12, 'k2': 7, 'lambda_value': 0.6, 
                     'alpha': 1.1, 'long_time_t': 1000, 'short_time_t': 1000, 'num_search_times': 1},}
          

def argument_parser():
    """ Argument parser
    Receive args

    Args:
        None
        
    Returns:
        parser: An argument object that contains all args

    Raises:
        None
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--src_root', type=str, default="data/preprocessed_data/", 
            help='the root path of tracked files of single camera with submission format')
    parser.add_argument('--dst_root', type=str, default="submit/", 
            help='the root path of the generated file to submit')
    parser.add_argument('--mode', type=str, default='linear')
    parser.add_argument('--st_dim', type=int, default=0)
    parser.add_argument('--en_dim', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--r_rate', type=float, default=0.5)

    parser.add_argument('--k1', type=int, default=12)
    parser.add_argument('--k2', type=int, default=7)
    parser.add_argument('--lambda_value', type=float, default=0.6)

    parser.add_argument('--alpha', type=float, default=1.1)
    parser.add_argument('--long_time_t', type=float, default=1000)
    parser.add_argument('--short_time_t', type=float, default=1000)

    parser.add_argument('--num_search_times', type=int, default=1)

    parser.add_argument('--occ_rate', type=float, default=1.0)
    parser.add_argument('--occ_alpha', type=float, default=0.)

    return parser

class MultiCameraMatching(object):
    """ This class is used to match tracklets among all different cameras.

    Attributes:
        cam_arr: camera id array
        track_arr: tracklet array
        in_dir_arr: the "in zone" in our paper
        out_dir_arr: the "out zone" in our paper
        in_time_arr: the time when a tracklet enter the "in zone"
        out_time_arr: the time when a tracklet exit the "out zone"
        feat_dict: the tracklet features
        feat_arr: it has been deprecated
    """

    def __init__(self, cam_arr, track_arr, in_dir_arr, out_dir_arr, 
            in_time_arr, out_time_arr, feat_dict,
            topk=5, r_rate=0.5, 
            k1=12, k2=7, lambda_value=0.6,
            alpha=1.1, long_time_t=1000, short_time_t=1000,
            num_search_times=1,
            trun_dict=None, occ_rate=1., occ_alpha=0.):
        self.cam_arr = cam_arr
        self.track_arr = track_arr
        self.in_dir_arr = in_dir_arr
        self.out_dir_arr = out_dir_arr
        self.in_time_arr = in_time_arr
        self.out_time_arr = out_time_arr

        self.feat_dict = feat_dict
        self.trun_dict = trun_dict

        ### params
        self.topk = topk
        self.r_rate = r_rate

        self.k1 = k1
        self.k2 = k2
        self.lambda_value = lambda_value

        self.alpha = alpha
        self.long_time_t = long_time_t
        self.short_time_t = short_time_t

        self.num_search_times = num_search_times

        self.occ_rate = occ_rate
        self.occ_alpha = occ_alpha


        self.global_id_arr = np.zeros_like(cam_arr) - 1

    def select_map_arr(self, cam_id, is_out, direction=True):
        """ Select valid vehicles...
        Args:
            cam_id: camera id
            is_out: The track is out of the camera, else in
            direction: True is positive and False is negative
        """
        map_arr = (self.cam_arr == cam_id)
        dir_map = np.zeros_like(map_arr)
        if is_out: # 保留正确的进/出口车辆
            valid_directions = out_valid_positive_directions if direction else out_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == -1) & (self.out_dir_arr == valid_directions[0][1]) # for those tracks at the beginning
            dir_map |= tmp_map
            if cam_id == 46: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_thre = out_positive_direction_time_thre_dict[cam_id] if direction \
                    else out_negative_direction_time_thre_dict[cam_id] # time threshold
            tmp_map = (self.out_dir_arr == valid_directions[0][1]) & (self.out_time_arr < t_thre)
            dir_map &= tmp_map
            # dir_map = (self.out_time_arr < t_thre)
        else:
            valid_directions = in_valid_positive_directions if direction else in_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.out_dir_arr == -1) # for those tracks in the last time
            dir_map |= tmp_map
            if cam_id == 46: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            t_thre = in_positive_direction_time_thre_dict[cam_id] if direction \
                    else in_negative_direction_time_thre_dict[cam_id] # time threshold
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.in_time_arr > t_thre)
            dir_map &= tmp_map
            # dir_map = (self.in_time_arr > t_thre)

        map_arr &= dir_map
        return map_arr

    def select_map_arr_interval(self, cam_id, is_out, interval=[0, 2001], direction=True):
        """ Select valid vehicles...
        Args:
            cam_id: camera id
            is_out: The track is out of the camera, else in
            direction: True is positive and False is negative
        """
        
        map_arr = (self.cam_arr == cam_id)
        dir_map = np.zeros_like(map_arr)
        if is_out: # 保留正确的进/出口车辆
            valid_directions = out_valid_positive_directions if direction else out_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == -1) & (self.out_dir_arr == valid_directions[0][1]) # for those tracks at the beginning
            dir_map |= tmp_map
            if cam_id == 46 or cam_id == 43: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            # t_thre = out_positive_direction_time_thre_dict[cam_id] if direction else out_negative_direction_time_thre_dict[cam_id] # time threshold
            t_min_thre, t_max_thre = interval
            tmp_map = (self.out_dir_arr == valid_directions[0][1]) & (t_min_thre < self.out_time_arr) & \
                    (self.out_time_arr < t_max_thre)
            dir_map &= tmp_map
            # dir_map = (self.out_time_arr < t_thre)
        else:
            valid_directions = in_valid_positive_directions if direction else in_valid_negative_directions
            for i in range(len(valid_directions)):
                tmp_map = (self.in_dir_arr == valid_directions[i][0]) & (self.out_dir_arr == valid_directions[i][1])
                dir_map |= tmp_map
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (self.out_dir_arr == -1) # for those tracks in the last time
            dir_map |= tmp_map
            if cam_id == 46 or cam_id == 43: # for the special case!
                tmp_map = (self.in_dir_arr == 1) & (self.out_dir_arr == 3) # for the special case in camera 46
                dir_map |= tmp_map

            # t_thre = in_positive_direction_time_thre_dict[cam_id] if direction else in_negative_direction_time_thre_dict[cam_id] # time threshold
            t_min_thre, t_max_thre = interval
            tmp_map = (self.in_dir_arr == valid_directions[0][0]) & (t_min_thre < self.in_time_arr) & \
                    (self.in_time_arr < t_max_thre) 
            dir_map &= tmp_map
            # dir_map = (self.in_time_arr > t_thre)

        map_arr &= dir_map
        return map_arr

    def do_matching(self, cam_out_arr, cam_in_arr, track_out_arr, track_in_arr, 
            out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id, st_dim=0, en_dim=2048):
        n_out = cam_out_arr.shape[0]
        cam_out_feat_list = []
        track_out_feat_list = []
        index_out_list = []
        feat_out_list = []
        trun_out_list = []
        for i in range(n_out):
            f_out = np.array(self.feat_dict[cam_out_arr[i]][track_out_arr[i]])[:, st_dim:en_dim]
            trun_out = np.array(self.trun_dict[cam_out_arr[i]][track_out_arr[i]])
#             cam_out_feat_list.append(np.ones(f_out.shape[0]) * cam_out_arr[i])
#             track_out_feat_list.append(np.ones(f_out.shape[0]) * track_out_arr[i])
            index_out_list.append(np.ones(f_out.shape[0], dtype=np.int64) * i)
            feat_out_list.append(f_out)
            trun_out_list.append(trun_out)
#         cam_out_feat_arr = np.concatenate(cam_out_feat_list) # n
#         track_out_feat_arr = np.concatenate(track_out_feat_arr) # n
        index_out_arr = np.concatenate(index_out_list)
        feat_out_arr = np.concatenate(feat_out_list) # nxc
        trun_out_arr = np.concatenate(trun_out_list) # n
        print ('done for preparing feat_out_arr')

        n_in = cam_in_arr.shape[0]
        cam_in_feat_list = []
        track_in_feat_list = []
        index_in_list = []
        feat_in_list = []
        trun_in_list = []
        for j in range(n_in):
            f_in = np.array(self.feat_dict[cam_in_arr[j]][track_in_arr[j]])[:, st_dim:en_dim]
            trun_in = np.array(self.trun_dict[cam_in_arr[j]][track_in_arr[j]])
#             cam_in_feat_list.append(np.ones(f_in.shape[0]) * cam_in_arr[j])
#             track_in_feat_list.append(np.ones(f_in.shape[0]) * track_in_arr[j])
            index_in_list.append(np.ones(f_in.shape[0], dtype=np.int64) * j)
            feat_in_list.append(f_in)
            trun_in_list.append(trun_in)
#         cam_in_feat_arr = np.concatenate(cam_in_feat_list)
#         track_in_feat_arr = np.concatenate(track_in_feat_arr)
        index_in_arr = np.concatenate(index_in_list)
        feat_in_arr = np.concatenate(feat_in_list) # mxc
        trun_in_arr = np.concatenate(trun_in_list) # m
        print ('done for preparing feat_in_arr')

        print ('start to compute distance matrix...')
        dist_mat = self.compute_distance_matrix(feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                    cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id, 
                    trun_out_arr, trun_in_arr) # nxm

        print ('start to find matched pairs...')
        matched_i, matched_j, matched_d = self.find_pairs(dist_mat, index_out_arr, index_in_arr, 
                track_out_arr, track_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id)

        return matched_i, matched_j
    
    def compute_distance_matrix(self, feat_out_arr, feat_in_arr, index_out_arr, index_in_arr,
                    cam_out_arr, cam_in_arr, out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id,
                    trun_out_arr, trun_in_arr):
        # dist_mat = np.matmul(feat_out_arr, feat_in_arr.T) # baseline for cosine
        # dist_mat = distance.cdist(feat_out_arr, feat_in_arr, 'euclidean') # baseline for L2

        # rerank
        q_q_sim = np.matmul(feat_out_arr, feat_out_arr.T)
        g_g_sim = np.matmul(feat_in_arr, feat_in_arr.T)
        q_g_sim = np.matmul(feat_out_arr, feat_in_arr.T)
        k1 = self.k1
        k2 = self.k2
        lambda_value = self.lambda_value
        dist_mat = re_ranking(q_g_sim, q_q_sim, g_g_sim, k1=k1, k2=k2, lambda_value=lambda_value) # nxm
        
        # mask with intervals
        tth_min, tth_max = two_track_valid_pass_time_for_mask_dict[(cam_out_id, cam_in_id)]
        out_time_out_box_arr = out_time_out_arr[index_out_arr] # n
        in_time_in_box_arr = in_time_in_arr[index_in_arr] # m
        n_out_box = out_time_out_box_arr.shape[0]
        n_in_box = in_time_in_box_arr.shape[0]
        out_time_out_box_mat = np.expand_dims(out_time_out_box_arr, 1).repeat(n_in_box, 1) # nxm
        in_time_in_box_mat = np.expand_dims(in_time_in_box_arr, 0).repeat(n_out_box, 0) # nxm
        
        alpha = self.alpha # param need to be adapted
        long_time_t = self.long_time_t # param need to be adapted
        short_time_t = self.short_time_t # param need to be adapted
        travel_time_mat = in_time_in_box_mat - out_time_out_box_mat
        travel_time_mask = np.ones_like(travel_time_mat)
        too_short_pairs_indices = (travel_time_mat < tth_min)
        too_long_pairs_indices = (travel_time_mat > tth_max)
        travel_time_mask[too_short_pairs_indices] = np.exp(alpha * (tth_min - \
                                                travel_time_mat[too_short_pairs_indices]) / short_time_t)
        travel_time_mask[too_long_pairs_indices] = np.exp(alpha * (travel_time_mat[too_long_pairs_indices] \
                                                        - tth_max) / long_time_t)

        dist_mat *= travel_time_mask

        # mask with occlusion
        occ_rate = self.occ_rate
        occ_alpha = self.occ_alpha
        trun_out_arr = np.expand_dims(trun_out_arr, 1).repeat(n_in_box, 1) # nxm
        trun_out_mask_arr = (trun_out_arr > occ_rate)
        trun_out_weight_arr = np.ones_like(trun_out_arr)
        trun_out_weight_arr[trun_out_mask_arr] = np.exp(occ_alpha * (1 + trun_out_arr[trun_out_mask_arr]))

        trun_in_arr = np.expand_dims(trun_in_arr, 0).repeat(n_out_box, 0) # nxm
        trun_in_mask_arr = (trun_in_arr > occ_rate)
        trun_in_weight_arr = np.ones_like(trun_in_arr)
        trun_in_weight_arr[trun_in_mask_arr] = np.exp(occ_alpha * (1 + trun_in_arr[trun_in_mask_arr]))

        dist_mat *= trun_out_weight_arr
        dist_mat *= trun_in_weight_arr

        return dist_mat

    def find_pairs(self, dist_mat, index_out_arr, index_in_arr, track_out_arr, track_in_arr,
                        out_time_out_arr, in_time_in_arr, cam_out_id, cam_in_id):
        sorted_out_index_dist_mat = dist_mat.argsort(1) # nxm
        sorted_in_index_dist_mat = dist_mat.argsort(0) # nxm
        topk = self.topk # param need to be adapted
        r_rate = self.r_rate # param need to be adapted
        matched_box_dict = {}
        for i in range(dist_mat.shape[0]): # iter out_port
            forward_candidate_track_index_arr = index_in_arr[sorted_out_index_dist_mat[i]][:topk] # track_in_arr的index
            bin_count = np.bincount(forward_candidate_track_index_arr)
            forward_matched_track_id = np.argmax(bin_count) # 匹配次数最多的tracklet的index
            forward_matched_track_id_count = bin_count[forward_matched_track_id] # 该tracklet的出现次数
            if forward_matched_track_id_count < 2:
                continue
            indices = np.where(forward_candidate_track_index_arr == forward_matched_track_id)[0]
            forward_indices = sorted_out_index_dist_mat[i, indices]
            reverse_pair_track_count = 0
            for j in forward_indices:
                if i in sorted_in_index_dist_mat[:topk, j]:
                    reverse_pair_track_count += 1
            if reverse_pair_track_count / forward_matched_track_id_count > r_rate:
                track_index = index_out_arr[i]
                if track_index not in matched_box_dict:
                    matched_box_dict[track_index] = []
                matched_box_dict[track_index].append(forward_matched_track_id)

        # find pairs
        matched_i = []
        matched_j = []
        matched_d = []
        for track_out_index in sorted(matched_box_dict.keys()):
            bin_count = np.bincount(np.array(matched_box_dict[track_out_index]))
            track_in_index = np.argmax(bin_count)
            track_in_index_count = bin_count[track_in_index]
            if track_in_index_count > 1:
                matched_i.append(track_out_index)
                matched_j.append(track_in_index)
                matched_d.append(track_in_index_count)
        matched_i = np.array(matched_i, dtype=np.int16)
        matched_j = np.array(matched_j, dtype=np.int16)
        matched_d = np.array(matched_d, dtype=np.int16)

        # filter repeated pairs
        unique_matched_j = np.unique(matched_j)
        filtered_matched_i = []
        filtered_matched_j = []
        filtered_matched_d = []
        for mj in unique_matched_j:
            mj_arr = np.where(matched_j == mj)[0]
            if mj_arr.shape[0] > 1:
                md_arr = matched_d[mj_arr]
                md_index = md_arr.argmax() # need to be optimized
                mj_index = mj_arr[md_index]
                filtered_matched_i.append(matched_i[mj_index])
                filtered_matched_j.append(matched_j[mj_index])
                filtered_matched_d.append(matched_d[mj_index])
            else:
                filtered_matched_i.append(matched_i[mj_arr[0]])
                filtered_matched_j.append(matched_j[mj_arr[0]])
                filtered_matched_d.append(matched_d[mj_arr[0]])
        filtered_matched_i = np.array(filtered_matched_i, dtype=np.int16)
        filtered_matched_j = np.array(filtered_matched_j, dtype=np.int16)
        filtered_matched_d = np.array(filtered_matched_d, dtype=np.int16)
        return filtered_matched_i, filtered_matched_j, filtered_matched_d

    def drop_invalid_matched_pairs(self, matched_i, matched_j, cam_out_id, cam_in_id, out_time_out_arr, in_time_in_arr):
        """
        Args:
            matched_i: 
            matched_j:
        """
        tth_min, tth_max = two_track_valid_pass_time_dict[(cam_out_id, cam_in_id)]
        keep_ids = []
        for idx, (i, j) in enumerate(zip(matched_i, matched_j)):
            travel_time = in_time_in_arr[j] - out_time_out_arr[i]
            if travel_time < tth_min or travel_time > tth_max:
                continue
            keep_ids.append(idx)
        matched_i = matched_i[keep_ids]
        matched_j = matched_j[keep_ids]
        return matched_i, matched_j

    def matching(self, cam_in_id, cam_out_id, interval_out=[0, 2001], interval_in=[0, 2001], \
                direction=True, mode='linear', st_dim=0, en_dim=2048, is_params=True):
        if cam_in_id > cam_out_id:
            assert direction == True
        else:
            assert direction == False

        if is_params:
            map_out_arr = self.select_map_arr_interval(cam_out_id, is_out=True, \
                            interval=interval_out, direction=direction)
        else:
            map_out_arr = self.select_map_arr(cam_out_id, is_out=True, direction=direction)
        cam_out_arr = self.cam_arr[map_out_arr]
        track_out_arr = self.track_arr[map_out_arr]
        # in_dir_out_arr = self.in_dir_arr[map_out_arr]
        # out_dir_out_arr = self.out_dir_arr[map_out_arr]
        in_time_out_arr = self.in_time_arr[map_out_arr]
        out_time_out_arr = self.out_time_arr[map_out_arr]

        if is_params:
            map_in_arr = self.select_map_arr_interval(cam_in_id, is_out=False, \
                            interval=interval_in, direction=direction)
        else:
            map_in_arr = self.select_map_arr(cam_in_id, is_out=False, direction=direction)
        cam_in_arr = self.cam_arr[map_in_arr]
        track_in_arr = self.track_arr[map_in_arr]
        # in_dir_in_arr = self.in_dir_arr[map_in_arr]
        # out_dir_in_arr = self.out_dir_arr[map_in_arr]
        in_time_in_arr = self.in_time_arr[map_in_arr]
        out_time_in_arr = self.out_time_arr[map_in_arr]
        print ('cam: {}; tracks: {} \t track_ids: {}'.format(cam_out_id, len(track_out_arr), np.sort(track_out_arr)))
        print ('cam: {}; tracks: {} \t track_ids: {}'.format(cam_in_id, len(track_in_arr), np.sort(track_in_arr)))

        #### Search results circularly
        all_matched_i = []
        all_matched_j = []
        print ('* Start matching...')
        topk = self.topk
        r_rate = self.r_rate
        num_search_times = self.num_search_times
        for i in range(num_search_times):
            print ('** Iter {}...'.format(i))
            sub_track_out_arr = np.setdiff1d(track_out_arr, track_out_arr[all_matched_i]) # sorted. need to be readjust
            sub_track_in_arr = np.setdiff1d(track_in_arr, track_in_arr[all_matched_j]) # sorted. need to be readjust
            
            num_candidates = 4
            if sub_track_out_arr.shape[0] < num_candidates or sub_track_in_arr.shape[0] < num_candidates:
                break

            map_sub_out_arr = np.isin(track_out_arr, sub_track_out_arr, True) # accelerate
            map_sub_in_arr = np.isin(track_in_arr, sub_track_in_arr, True) # accelerate

            sub_track_out_arr = track_out_arr[map_sub_out_arr] # original order
            sub_track_in_arr = track_in_arr[map_sub_in_arr] # original order

            sub_cam_out_arr = cam_out_arr[map_sub_out_arr]
            sub_cam_in_arr = cam_in_arr[map_sub_in_arr]

            sub_out_time_out_arr = out_time_out_arr[map_sub_out_arr]
            sub_in_time_in_arr = in_time_in_arr[map_sub_in_arr]

            r = min(sub_track_out_arr.shape[0] / float(track_out_arr.shape[0]), 
                    sub_track_in_arr.shape[0] / float(track_in_arr.shape[0]))
            self.topk = int(topk * r)
            if self.topk < 3:
                self.topk = 3
        
            sub_matched_i, sub_matched_j = self.do_matching(sub_cam_out_arr, sub_cam_in_arr, sub_track_out_arr, 
                                                    sub_track_in_arr, sub_out_time_out_arr, sub_in_time_in_arr, 
                                                    cam_out_id, cam_in_id, st_dim=st_dim, en_dim=en_dim)
            sub_matched_i, sub_matched_j = self.drop_invalid_matched_pairs(sub_matched_i, sub_matched_j, 
                                                cam_out_id, cam_in_id, sub_out_time_out_arr, sub_in_time_in_arr)

            for smi, smj in zip(sub_matched_i, sub_matched_j):
                # assert mi, mj only match one item in track_arr
                mi = np.where(track_out_arr == sub_track_out_arr[smi])[0].item()
                mj = np.where(track_in_arr == sub_track_in_arr[smj])[0].item()
                
                assert (mi not in all_matched_i)
                assert (mj not in all_matched_j)
                all_matched_i.append(mi)
                all_matched_j.append(mj)
        matched_i = np.array(all_matched_i)
        matched_j = np.array(all_matched_j)

        matched_track_out_arr = track_out_arr[matched_i]
        sorted_ids = np.argsort(matched_track_out_arr)
        matched_i = matched_i[sorted_ids] # for print
        matched_j = matched_j[sorted_ids] # for print

        print ('number of matched pairs: {}'.format(len(matched_i)))
        global_max_id = self.global_id_arr.max() + 1
        for i, j in zip(matched_i, matched_j):
            track_out_id = track_out_arr[i]
            track_in_id = track_in_arr[j]
            idx_i = (self.cam_arr == cam_out_id) & (self.track_arr == track_out_id)
            idx_j = (self.cam_arr == cam_in_id) & (self.track_arr == track_in_id)

            try:
                assert (self.global_id_arr[idx_j].item() == -1)
            except:
                pdb.set_trace()
            if self.global_id_arr[idx_i].item() != -1:
                self.global_id_arr[idx_j] = self.global_id_arr[idx_i]
            else:
                self.global_id_arr[idx_i] = global_max_id
                self.global_id_arr[idx_j] = global_max_id
                global_max_id += 1
            all_g_ids = np.where(self.global_id_arr == self.global_id_arr[idx_i].item())[0]
            all_matched_cams = self.cam_arr[all_g_ids]
            all_matched_tracks = self.track_arr[all_g_ids]
            print ('{:3d}: ({:3d}, {:3d}) \t interval: {:4d} \t all_matched_cams: {:18s} \t '
                    'all_matched_tracks: {}'.format(self.global_id_arr[idx_i].item(), 
                    track_out_id, track_in_id, self.in_time_arr[idx_j].item() - self.out_time_arr[idx_i].item(), 
                    ', '.join(map(str, all_matched_cams)), ', '.join(map(str, all_matched_tracks))))
            

    def forward_matching(self, mode='linear', st_dim=0, en_dim=2048):
        # positve matching
        for cam_id in range(41, 46):
            cam_out_id = cam_id
            cam_in_id = cam_id + 1
            print ('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            key = (cam_out_id, cam_in_id)
            print ('params: {}'.format(args_params_dict[key]))
            self.topk = args_params_dict[key]['topk']
            self.r_rate = args_params_dict[key]['r_rate']
            self.k1 = args_params_dict[key]['k1']
            self.k2 = args_params_dict[key]['k2']
            self.lambda_value = args_params_dict[key]['lambda_value']
            self.alpha = args_params_dict[key]['alpha']
            self.long_time_t = args_params_dict[key]['long_time_t']
            self.short_time_t = args_params_dict[key]['short_time_t']
            self.num_search_times = args_params_dict[key]['num_search_times']

            self.matching(cam_in_id, cam_out_id, direction=True, mode=mode, \
                    st_dim=st_dim, en_dim=en_dim, is_params=False)

        # negative matching
        for cam_id in range(46, 41, -1):
            cam_out_id = cam_id
            cam_in_id = cam_id - 1
            print ('out: {}; in: {}'.format(cam_out_id, cam_in_id))

            key = (cam_out_id, cam_in_id)
            print ('params: {}'.format(args_params_dict[key]))
            self.topk = args_params_dict[key]['topk']
            self.r_rate = args_params_dict[key]['r_rate']
            self.k1 = args_params_dict[key]['k1']
            self.k2 = args_params_dict[key]['k2']
            self.lambda_value = args_params_dict[key]['lambda_value']
            self.alpha = args_params_dict[key]['alpha']
            self.long_time_t = args_params_dict[key]['long_time_t']
            self.short_time_t = args_params_dict[key]['short_time_t']

            self.matching(cam_in_id, cam_out_id, direction=False, mode=mode, \
                    st_dim=st_dim, en_dim=en_dim, is_params=False)

    def write_output(self, src_path, dst_path):
        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))

        print ('* writing output...')
        dst_obj = open(dst_path, 'w')
        with open(src_path, 'r') as fid:
            for line in fid.readlines():
                s = [int(i) for i in line.rstrip().split()]

                if s[0] == 45 or s[0] == 46:
                    h = 720
                else:
                    h = 960
                w = 1280
                if s[3] < 0 or s[4] < 0 or (s[5]+s[3]) > w or (s[6]+s[4]) > h:
                    continue

                idx = ((self.cam_arr == s[0]) & (self.track_arr == s[1]))
                g_id = self.global_id_arr[idx].item()
                if g_id != -1:
                    s[1] = g_id
                    dst_obj.write('{}\n'.format(' '.join(map(str, s)))) # [camera_id, track_id, frame_id, x, y, w, h, -1, -1]
        dst_obj.close()


def prepare_data(cam_path, track_path, in_out_all_path, feat_path=None):
    cam_arr = np.load(cam_path)
    track_arr = np.load(track_path)
    feat_arr = np.load(feat_path) if feat_path is not None else np.zeros((cam_arr.shape[0], 8))
    c_dict = {}
    with open(in_out_all_path, 'r') as fid:
        for line in fid.readlines():
            s = [int(i) for i in line.rstrip().split()]
            if s[0] not in c_dict:
                c_dict[s[0]] = {}
            c_dict[s[0]][s[1]] = s[2:]

    sorted_ids = np.argsort(cam_arr)
    cam_arr = cam_arr[sorted_ids]
    track_arr = track_arr[sorted_ids]
    feat_arr = feat_arr[sorted_ids]
    in_dir_arr = np.zeros_like(cam_arr)
    out_dir_arr = np.zeros_like(cam_arr)
    in_time_arr = np.zeros_like(cam_arr)
    out_time_arr = np.zeros_like(cam_arr)
    for i in range(len(cam_arr)):
        in_dir_arr[i], out_dir_arr[i], in_time_arr[i], out_time_arr[i] = c_dict[cam_arr[i]][track_arr[i]]
    return cam_arr, track_arr, in_dir_arr, out_dir_arr, in_time_arr, out_time_arr, feat_arr,

def load_feat_dict(feat_path):
    feat_dict = pickle.load(open(feat_path, 'rb'))
    return feat_dict

def load_pickle_dict(pickle_path):
    pickle_dict = pickle.load(open(pickle_path, 'rb'))
    return pickle_dict

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    cam_path = os.path.join(args.src_root, 'cam_vec.npy')
    track_path = os.path.join(args.src_root, 'track_vec.npy')
    in_out_all_path = os.path.join(args.src_root, 'in_out_all.txt')
    feat_path = None # the original version, it has been deprecated.
    feat_path2 = os.path.join(args.src_root, 'feat_all_vec.pkl')
    
    src_path = os.path.join(args.src_root, 'all_cameras.txt')
    dst_path = os.path.join(args.dst_root, 'track1.txt')
    if not os.path.exists(args.dst_root):
        os.makedirs(args.dst_root)

    feat_dict = load_feat_dict(feat_path2) # load features

    trun_path = os.path.join(args.src_root, 'truncation.pkl')
    trun_dict = load_pickle_dict(trun_path) # load truncation rates

    preprocessed_data = prepare_data(cam_path, track_path, in_out_all_path) # load all preprocessed data
    cam_arr, track_arr, in_dir_arr, out_dir_arr, in_time_arr, out_time_arr, feat_arr = preprocessed_data
    matcher = MultiCameraMatching(cam_arr, track_arr, in_dir_arr, out_dir_arr, 
                                in_time_arr, out_time_arr, feat_dict,
                                topk=args.topk, r_rate=args.r_rate, 
                                k1=args.k1, k2=args.k2, lambda_value=args.lambda_value,
                                alpha=args.alpha, long_time_t=args.long_time_t, short_time_t=args.short_time_t,
                                num_search_times=args.num_search_times,
                                trun_dict=trun_dict, occ_rate=args.occ_rate, occ_alpha=args.occ_alpha)

    matcher.forward_matching(mode=args.mode, st_dim=args.st_dim, en_dim=args.en_dim)

    print ('* matching done.')
    matcher.write_output(src_path, dst_path)

