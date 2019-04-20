import numpy as np
from rtree.index import Rtree
from src.features.helper import *
import sys
import logging
import pickle
import time

def main(train):
    logger = logging.getLogger('build_rtree')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./log/%s' % train)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    start_time = time.time()
    train_path = './data/processed/%s.txt' % train
    data = load_trajectory(train_path)
    trajectory, id_list = build_qgram(data)
    order_id_dict = build_id_dict(id_list)
    order_key_dict = build_order_dict(id_list)

    #save orderId-key mapping
    #key: trajectory id in string, value: encoded key
    filename = '../data/processed/order_id_dict.txt'
    outfile = open(filename,'wb')
    pickle.dump(order_id_dict,outfile)
    outfile.close()

    #key: key, value: trajectory id in string
    filename = '../data/processed/order_key_dict.txt'
    outfile = open(filename,'wb')
    pickle.dump(order_key_dict,outfile)
    outfile.close()


    # R-tree constructor
    # parameter: 'data_full' is the filename of R-tree storage
    #            2 files are created: data_full.dat, data_full.idx
    # return: r-tree index
    rtree_path = './data/interim/%s/my_rtree' % train
    data_idx = Rtree(rtree_path)
    logger.info('Output R-tree: %s' % rtree_path)
    # put all trajectories into r-tree in the form of bounding box
    node_id = 0
    for key, qgrams in trajectory.items():
        for qgram in qgrams:
        # paremeters:
        #    1. node id
        #    2. bounding box(point): (x,y,x,y)
        #    3. data inside each node: trajectory's key from order_dict
        data_idx.insert(node_id, (qgram[0],qgram[1],qgram[0],qgram[1]), obj=(order_id_dict[key]))
        node_id += 1

    del data_idx
    end_time = time.time()
    print("exec time: "+str(end_time-start_time))
    print("Finished...")


if __name__ == '__main__':
    train_data = sys.argv[1]
    main(train_data)
