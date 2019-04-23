from os import path
import sys
# sys.path.append(path.abspath('/Users/mingxidai/Documents/project/trajectory-search/'))
sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))
import pickle
import time
import traj_dist.distance as tdist
from src.features.helper import load_trajectory
import numpy as np
import logging
import sys
import os
import gc
from src.features.helper import swap_k_v, read_pickle

#
# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold


def searchResult(query, train, query_num, user_k, qgram_size):
    logger = logging.getLogger('predict')
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

    # loading the files:
    logger.info('---------------------------- Predict the top-k similar trajectories ----------------------------')
    qgram_tag = 'q_%d' % qgram_size
    query_path = './data/processed/%s.txt' % query
    train_path = './data/processed/%s.txt' % train
    candidate_traj_path = './data/interim/%s/%s/candidate_trajectory_%s.txt' % (query, train, qgram_tag)
    query_id_dict_path = './data/interim/%s/%s/query_id_dict_%s.txt' % (query, train, qgram_tag)
    rtree_id_dict_path = './data/interim/%s/rtree_id_dict_%s.txt' % (train, qgram_tag)
    result_path = './data/result/%s/%s/%s/' % (query, train, qgram_tag)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    candidateList = read_pickle(candidate_traj_path)  # candidateList => [[queryID_1,[(traID1, count1),(traID2, count2)]], [...]]
    logger.info('Load candidate trajectory: %s' % candidate_traj_path)

    query_id_to_key = read_pickle(query_id_dict_path)
    logger.info('Load query id dictionary: %s' % query_id_dict_path)

    rtree_id_to_key = read_pickle(rtree_id_dict_path)
    logger.info('Load rtree id dictionary: %s' % rtree_id_dict_path)

    trajectory_dict = load_trajectory(train_path)
    logger.info('Load train trajectory: %s' % train_path)

    real_query_dict = load_trajectory(query_path, n=query_num)
    logger.info('Load %d query trajectory: %s' % (query_num, query_path))

    query_key_to_id = swap_k_v(query_id_to_key)  # key: encoded key; value: trajectory id in string
    rtree_key_to_id = swap_k_v(rtree_id_to_key)  # key: encoded key; value: trajectory id in string

    logger.info('Start finding top K')
    for index in range(len(candidateList)):  # start to calculate
        print(index)
        k = min(user_k, len(candidateList[index][1]))
        topK = candidateList[index][1][0:k]
        queryID = candidateList[index][0]
        pre_result = list(map(lambda x: x[0], topK))  # get the candidate trajectory IDs from top k
        # print(queryID)
        # print(pre_result)
        result_map = {}  # build a map to save the result
        for t in pre_result:
             # result_map[t] = calculateEdr(trajectory_dict[rtree_id_dict[t]], real_query_dict[query_id_dict[queryID]])
            result_map[t] = tdist.edr(np.array(trajectory_dict[rtree_key_to_id[t]]), np.array(real_query_dict[query_key_to_id[queryID]]), "spherical")*max(len(trajectory_dict[rtree_key_to_id[t]]),len(real_query_dict[query_key_to_id[queryID]]))
        # print(result_map)
        fullCandidates = candidateList[index][1]  # list of [ID, count]
        i = k
        query_tra = real_query_dict[query_key_to_id[queryID]]
        lengthQ = len(query_tra)
        bestSoFar = result_map[topK[i-1][0]]
        while i < len(fullCandidates):
            candidate = fullCandidates[i]
            candidateID = candidate[0]
            # try:
            tra_s = trajectory_dict[rtree_key_to_id[candidateID]]
            # Mingxi: you can delete the "try except" part if you think there will no "miss match" in the dict anymore:
            # except KeyError:
            #     pass
            countValue = candidate[1]
            lengthS = len(tra_s)
            if countValue >= (max(lengthQ, lengthS) - (bestSoFar+1)*qgram_size):
                # pointedByCounts = filter(lambda e:e[1]==countValue, fullCandidates)
                # for s in pointedByCounts:
                    realDist = tdist.edr(np.array(tra_s), np.array(query_tra), "spherical")*max(len(tra_s), len(query_tra))
                    if realDist < bestSoFar:
                        result_map[candidateID] = realDist
                        bestSoFar = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[k-1][1]  # update the best so far
            else:
                break
            i += 1
        finalResult = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[0:k]
        with open(result_path + "/query_%s.txt" % queryID, 'w') as f:
            f.write(query_key_to_id[queryID] + '\n')
            f.write('\n'.join('{} {}'.format(item[0], item[1]) for item in finalResult))
        f.close()
        gc.collect()
    logger.info('Finished')


if __name__ == "__main__":
    query_trajectory = sys.argv[1]
    train_trajectory = sys.argv[2]
    query_n = int(sys.argv[3])
    top_k = int(sys.argv[4])
    qgram_n = int(sys.argv[5])
    searchResult(query=query_trajectory, train=train_trajectory, query_num=query_n, user_k=top_k, qgram_size=qgram_n)
