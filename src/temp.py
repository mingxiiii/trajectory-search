# # from rtree.index import Rtree
# from src.features.helper import *
#
# train = 'gps_20161002_trajectory'
# qgram_size = 5
# rtree_id_dict_path = './data/interim/%s/rtree_id_dict_q_%d.txt' % (train, qgram_size)
# # print(rtree_id_dict_path)
# rtree_id_dict = read_pickle(rtree_id_dict_path)
# # print(rtree_id_dict.keys())
# rtree_id_dict = swap_k_v(rtree_id_dict)
# # print(rtree_id_dict.keys())
# save_pickle(rtree_id_dict, rtree_id_dict_path)


import os
import sys
print(os.getcwd())
print(os.path.abspath(os.path.join(os.getcwd(), '..', 'traj-dist-master')))