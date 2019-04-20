from os import path
from src.features.build_bbox import *
import sys
sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))
import traj_dist.distance as tdist
import numpy as np
import logging
import os
import gc


def main(query, train, query_num):
    logger = logging.getLogger('build_truth')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('./log/%s' % query)
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

    logger.info('---------------------------- Search for ground truth ----------------------------')

    query_path = './data/processed/%s.txt' % query
    train_path = './data/processed/%s.txt' % train
    query_id_dict_path = './data/interim/%s/%s/query_id_dict.txt' % (query, train)
    rtree_id_dict_path = './data/interim/%s/rtree_id_dict.txt' % train
    result_path = './data/truth/%s/%s' % (query, train)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    query_id_dict = read_pickle(query_id_dict_path)  # fakeID -> realID
    logger.info('Load query id dictionary: %s' % query_id_dict_path)

    train_id_dict = read_pickle(rtree_id_dict_path)
    logger.info('Load train id dictionary: %s' % rtree_id_dict_path)

    query_data = load_trajectory(query_path, n=query_num)
    logger.info('Load %d query trajectory: %s' % (query_num, query_path))

    train_data = load_trajectory(train_path)  # realID -> vectors
    logger.info('Load train trajectory: %s' % train_path)

    train_id_dict = {v: k for k, v in train_id_dict.items()}
    # print(train_id_dict)
    # result = []
    for query_id, query_trajectory in query_data.items():
        query_key = query_id_dict[query_id]
        print(query_key)
        if query_key in [16, 17, 18, 19, 22, 25, 43, 45, 47]:
            distance_list = []
            train_key_list = []
            for train_id, train_trajectory in train_data.items():
                try:
                    train_key = train_id_dict[train_id]
                    distance = tdist.edr(np.array(train_trajectory), np.array(query_trajectory), "spherical")*max(len(train_trajectory),len(query_trajectory))
                    distance_list.append(distance)
                    train_key_list.append(train_key)
                except KeyError:
                    pass
            ix = sorted(range(len(distance_list)), key=lambda k: distance_list[k])
            distance_list_sorted = [distance_list[i] for i in ix]
            train_key_sorted = [train_key_list[i] for i in ix]
            trajectory_result = [(e1, e2) for e1, e2 in zip(train_key_sorted, distance_list_sorted)]
            with open(result_path + "/query_%s.txt" % query_key, 'w') as f:
                f.write(query_id + '\n')
                f.write('\n'.join('{} {}'.format(item[0], item[1]) for item in trajectory_result))
            f.close()
            gc.collect()
        # result.append([query_key, trajectory_result])
    logger.info('Finished building ground truth')
    return


if __name__ == '__main__':
    query_trajectory = sys.argv[1]
    train_trajectory = sys.argv[2]
    n = int(sys.argv[3])
    main(query=query_trajectory, train=train_trajectory, query_num=n)
