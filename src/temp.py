from rtree.index import Rtree
from src.features.build_bbox import *

train = 'gps_20161002_trajectory'
rtree_id_dict_path = './data/interim/%s/rtree_id_dict.txt' % train
rtree_id_dict = read_pickle(rtree_id_dict_path)
save_pickle(rtree_id_dict, rtree_id_dict_path)