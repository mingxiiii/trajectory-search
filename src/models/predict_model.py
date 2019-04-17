from os import path
import sys
# sys.path.append(path.abspath('/Users/mingxidai/Documents/project/trajectory-search/'))
sys.path.append(path.abspath('/Users/mingxidai/Documents/Master/traj-dist-master'))

import pickle
import time
import traj_dist.distance as tdist
from src.features.build_bbox import load_trajectory
import numpy as np
#
# RDD: (test-key,Iter[(train_key,count),(train_key,count)])
# change threshold
threshold = 0.5
qGramSize = 20

def match(coor1,coor2):
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


def searchResult(k):
    with open("./data/processed/candidate_trajectory.txt", "rb") as f:
        candidateList = pickle.load(f)
    print(candidateList[0][1][0:10])
    topK = candidateList[0][1][0:k]
    queryID = candidateList[0][0]
    with open("./data/processed/query_id_dict.txt", "rb") as f:
        query_id_dict = pickle.load(f)
    with open("./data/processed/rtree_id_dict2.txt", "rb") as f:
        rtree_id_dict = pickle.load(f)
    trajectory_dict = load_trajectory("./data/processed/gps_20161001_trajectory.txt")
    real_query_dict = load_trajectory("./data/processed/gps_20161002_query.txt")
    query_id_dict = {v: k for k, v in query_id_dict.items()}
    result = list(map(lambda x: x[0], topK))
    #print(query_id_dict[queryID])
    #print(queryID)
    #print(trajectory_dict[rtree_id_dict[result[0]]])

    #print(real_query_dict[query_id_dict[queryID]])
    result_map = {}
    for t in result:
         # result_map[t] = calculateEdr(trajectory_dict[rtree_id_dict[t]], real_query_dict[query_id_dict[queryID]])
        result_map[t] = tdist.edr(np.array(trajectory_dict[rtree_id_dict[t]]),np.array(real_query_dict[query_id_dict[queryID]]), "spherical")*max(len(trajectory_dict[rtree_id_dict[t]]),len(real_query_dict[query_id_dict[queryID]]))
    #print(result_map)
    fullCandidates = candidateList[0][1] # list of [ID, count]
    i = k
    query_tra = real_query_dict[query_id_dict[queryID]]
    lengthQ = len(query_tra)
    bestSoFar = result_map[topK[i-1][0]]
    while i < len(fullCandidates):
        candidate = fullCandidates[i]
        try:
            tra_s = trajectory_dict[rtree_id_dict[candidate[0]]]
        except KeyError:
            pass
        countValue = candidate[1]
        lengthS = len(tra_s)
        if countValue >= (max(lengthQ,lengthS) - (bestSoFar+1)*qGramSize):
            # pointedByCounts = filter(lambda e:e[1]==countValue, fullCandidates)
            # for s in pointedByCounts:
                realDist = tdist.edr(np.array(tra_s), np.array(query_tra), "spherical")*max(len(tra_s), len(query_tra))
                print(realDist)
                if(realDist<bestSoFar):
                    result_map[candidate[0]] = realDist
                    bestSoFar = sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[k-1][1]
        else:
            break
        i+=1

    return sorted(result_map.items(), key=lambda kv: (kv[1], kv[0]))[0:k]


if __name__ == "__main__":
    print(searchResult(7))
