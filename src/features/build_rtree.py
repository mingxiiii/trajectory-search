from rtree.index import Rtree
from src.features.helper import *
import sys
import logging
import time


def main(train, qgram_size):
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

    logger.info('---------------------------- Build R-tree ----------------------------')
    qgram_tag = 'q_%d' % qgram_size
    train_path = './data/processed/%s.txt' % train
    data = load_trajectory(train_path)
    logger.info('Load train trajectory: %s' % train_path)

    trajectory, id_list = build_qgram(data, qgram_size)
    id_to_key_dict = build_id_to_key(id_list)
    # order_key_dict = build_order_dict(id_list)

    #save orderId-key mapping
    #key: trajectory id in string, value: encoded key

    rtree_id_dict_path = './data/interim/%s/rtree_id_dict_%s.txt' % (train, qgram_tag)
    save_pickle(id_to_key_dict, rtree_id_dict_path)
    logger.info('Output rtree_id_dict: %s' % rtree_id_dict_path)

    #key: key, value: trajectory id in string
    # filename = '../data/processed/order_key_dict.txt'
    # outfile = open(filename,'wb')
    # pickle.dump(order_key_dict,outfile)
    # outfile.close()


    # R-tree constructor
    # parameter: 'data_full' is the filename of R-tree storage
    #            2 files are created: data_full.dat, data_full.idx
    # return: r-tree index
    rtree_path = './data/interim/%s/my_rtree_%s' % (train, qgram_tag)
    data_idx = Rtree(rtree_path)
    logger.info('Output R-tree: %s' % rtree_path)
    # put all trajectories into r-tree in the form of bounding box
    node_id = 0
    start_time = time.time()
    for key, qgrams in trajectory.items():
        for qgram in qgrams:
        #    parameters:
        #    1. node id
        #    2. bounding box(point): (x,y,x,y)
        #    3. data inside each node: trajectory's key from order_dict
            data_idx.insert(node_id, (qgram[0], qgram[1], qgram[0], qgram[1]), obj=(id_to_key_dict[key]))
            node_id += 1

    del data_idx
    end_time = time.time()
    logger.info("exec time: "+str(end_time-start_time))
    logger.info('Finished building R-tree')


if __name__ == '__main__':
    train_data = sys.argv[1]
    q_size = sys.argv[2]
    main(train_data, q_size)
