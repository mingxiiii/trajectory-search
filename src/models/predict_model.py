from os import path
import sys
# sys.path.append(path.abspath('/Users/mingxidai/Documents/project/trajectory-search/'))
sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))
import pickle
import time
import traj_dist.distance as tdist
from src.features.build_bbox import load_trajectory
import numpy as np
import logging
import sys
import os
import gc

#
# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold
threshold = 0.05
qGramSize = 20


def match(coor1, coor2):
    return abs(coor1[0]-coor2[0]) <= threshold and abs(coor1[1]-coor2[1]) <= threshold


def calculateEdr(trajectory1, trajectory2):
    if len(trajectory1) == 0:
        return len(trajectory2)
    elif len(trajectory2) == 0:
        return len(trajectory1)
    else:
        return min(calculateEdr(trajectory1[1:], trajectory2[1:]) + subcost(trajectory1[0], trajectory2[0]),
                   calculateEdr(trajectory1[1:], trajectory2)+1,
                   calculateEdr(trajectory1, trajectory2[1:])+1)


def subcost(t1, t2):
    if match(t1, t2):
        return 1
    else:
        return 0


def searchResult(query, train, query_num, k):
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
    query_path = './data/processed/%s.txt' % query
    train_path = './data/processed/%s.txt' % train
    candidate_traj_path = './data/interim/%s/%s/candidate_trajectory.txt' % (query, train)
    query_id_dict_path = './data/interim/%s/%s/query_id_dict.txt' % (query, train)
    rtree_id_dict_path = './data/interim/%s/rtree_id_dict.txt' % train
    result_path = './data/result/%s/%s' % (query, train)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(candidate_traj_path, "rb") as f:
        candidateList = pickle.load(f) # candidateList => [[queryID_1,[(traID1, count1),(traID2, count2)]], [...]]
    logger.info('Load candidate trajectory: %s' % candidate_traj_path)

    with open(query_id_dict_path, "rb") as f:
        query_id_dict = pickle.load(f)
    logger.info('Load query id dictionary: %s' % query_id_dict_path)

    with open(rtree_id_dict_path, "rb") as f:
        rtree_id_dict = pickle.load(f)
    logger.info('Load rtree id dictionary: %s' % rtree_id_dict_path)

    trajectory_dict = load_trajectory(train_path)
    logger.info('Load train trajectory: %s' % train_path)

    real_query_dict = load_trajectory(query_path, n=query_num)
    logger.info('Load %d query trajectory: %s' % (query_num, query_path))
    query_id_dict = {v: k for k, v in query_id_dict.items()}  # reverse the query Dict: fakeID -> realID

    logger.info('Start finding top K')
    for index in range(len(candidateList)):  # start to calculate
        topK = candidateList[index][1][0:k]
        queryID = candidateList[index][0]
        pre_result = list(map(lambda x: x[0], topK))  # get the candidate trajectory IDs from top k
        result_map = {}  # build a map to save the result
        for t in pre_result:
             # result_map[t] = calculateEdr(trajectory_dict[rtree_id_dict[t]], real_query_dict[query_id_dict[queryID]])
            result_map[t] = tdist.edr(np.array(trajectory_dict[rtree_id_dict[t]]),np.array(real_query_dict[query_id_dict[queryID]]), "spherical")*max(len(trajectory_dict[rtree_id_dict[t]]),len(real_query_dict[query_id_dict[queryID]]))
        # print(result_map)
        fullCandidates = candidateList[index][1]  # list of [ID, count]
        i = k
        query_tra = real_query_dict[query_id_dict[queryID]]
        lengthQ = len(query_tra)
        bestSoFar = result_map[topK[i-1][0]]
        while i < len(fullCandidates):
            candidate = fullCandidates[i]
            candidateID = candidate[0]
            try:
                tra_s = trajectory_dict[rtree_id_dict[candidateID]]
            # Mingxi: you can delete the "try except" part if you think there will no "miss match" in the dict anymore:
            except KeyError:
                pass
            countValue = candidate[1]
            lengthS = len(tra_s)
            if countValue >= (max(lengthQ,lengthS) - (bestSoFar+1)*qGramSize):
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
        with open(result_path + "/query_%s.txt" % index, 'w') as f:
            f.write('\n'.join('{} {}'.format(item[0], item[1]) for item in finalResult))
        f.close()
        gc.collect()
    logger.info('Finished')


if __name__ == "__main__":
    query_trajectory = sys.argv[1]
    train_trajectory = sys.argv[2]
    n = int(sys.argv[3])
    top_k = int(sys.argv[4])
    searchResult(query=query_trajectory, train=train_trajectory, query_num=n, k=top_k)
